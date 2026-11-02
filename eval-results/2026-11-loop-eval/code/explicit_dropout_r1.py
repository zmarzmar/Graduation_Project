from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ValueRegularizerMode(str, Enum):
    TOKEN = "token"
    ATTENTION = "attention"


@dataclass(frozen=True)
class RegularizationCoefficients:
    # Final objective: J_final = J_task + λ_q J_q + λ_k J_k + λ_v J_v + λ_ff J_ff
    lambda_q: float = 0.0
    lambda_k: float = 0.0
    lambda_v: float = 0.0
    lambda_ff: float = 0.0
    dropout_p: float = 0.0
    value_mode: ValueRegularizerMode = ValueRegularizerMode.TOKEN


class ExplicitDropoutRegularizer:
    @staticmethod
    def _apply_token_mask(x: Tensor, token_mask: Optional[Tensor]) -> Tensor:
        # token_mask: [B, S] with 1/True for valid tokens, 0/False for padding
        if token_mask is None:
            return x
        mask = token_mask.to(dtype=x.dtype).unsqueeze(-1)  # [B, S, 1]
        return x * mask

    @staticmethod
    def _feature_gram(x: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        # X^T X for each batch item:
        # x: [B, S, D] -> gram: [B, D, D]
        x = ExplicitDropoutRegularizer._apply_token_mask(x, token_mask)
        return torch.einsum("bsd,bse->bde", x, x)

    @staticmethod
    def _attention_gram(attn_weights: Tensor) -> Tensor:
        # Y = A^T A for each batch item
        # attn_weights: [B, H, S, S] or [B, S, S]
        if attn_weights.dim() == 4:
            attn = attn_weights.mean(dim=1)  # [B, S, S], head-averaged attention
        else:
            attn = attn_weights
        return torch.matmul(attn.transpose(-1, -2), attn)  # [B, S, S]

    @staticmethod
    def _batch_trace_product(a: Tensor, b: Tensor) -> Tensor:
        # Tr(A B) for each batch item
        return torch.einsum("bij,bji->b", a, b)

    @staticmethod
    def query_regularizer(
        x: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        dropout_p: float,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # J_q = 1/2 Tr[((X^T X) ⊙ p^2) Λ_q]
        # Λ_q = W_q^T W_k X^T X W_k^T W_q
        gram = ExplicitDropoutRegularizer._feature_gram(x, token_mask)  # [B, D, D]
        b = gram * (dropout_p**2)

        wtq_wk = w_q.t() @ w_k
        wk_t_wq = w_k.t() @ w_q

        lambda_q = torch.einsum("ij,bjk,kl->bil", wtq_wk, gram, wk_t_wq)  # [B, D, D]
        reg = 0.5 * ExplicitDropoutRegularizer._batch_trace_product(b, lambda_q)
        return reg.mean()

    @staticmethod
    def key_regularizer(
        x: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        dropout_p: float,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # J_k = 1/2 Tr[((X^T X) ⊙ p^2) Λ_k]
        # Λ_k = W_k^T W_q X^T X W_q^T W_k
        gram = ExplicitDropoutRegularizer._feature_gram(x, token_mask)  # [B, D, D]
        b = gram * (dropout_p**2)

        wtk_wq = w_k.t() @ w_q
        wq_t_wk = w_q.t() @ w_k

        lambda_k = torch.einsum("ij,bjk,kl->bil", wtk_wq, gram, wq_t_wk)  # [B, D, D]
        reg = 0.5 * ExplicitDropoutRegularizer._batch_trace_product(b, lambda_k)
        return reg.mean()

    @staticmethod
    def value_regularizer_token(
        x: Tensor,
        w_v: Tensor,
        dropout_p: float,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # J_v = 1/2 Tr[((X^T X) ⊙ p^2) (W_v^T W_v)]
        gram = ExplicitDropoutRegularizer._feature_gram(x, token_mask)  # [B, D, D]
        b = gram * (dropout_p**2)
        wtw = w_v.t() @ w_v  # [D, D]
        reg = 0.5 * ExplicitDropoutRegularizer._batch_trace_product(b, wtw.expand_as(b))
        return reg.mean()

    @staticmethod
    def value_regularizer_attention(
        x: Tensor,
        attn_weights: Tensor,
        w_v: Tensor,
        dropout_p: float,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # J_av = 1/2 Tr[((X^T A^T A X) ⊙ p^2) (W_v^T W_v)]
        # ψ = X^T Y X, where Y = A^T A
        x = ExplicitDropoutRegularizer._apply_token_mask(x, token_mask)
        y = ExplicitDropoutRegularizer._attention_gram(attn_weights)  # [B, S, S]
        psi = torch.einsum("bsd,bse,bef->bdf", x, y, x) * (dropout_p**2)  # [B, D, D]
        wtw = w_v.t() @ w_v  # [D, D]
        reg = 0.5 * ExplicitDropoutRegularizer._batch_trace_product(psi, wtw.expand_as(psi))
        return reg.mean()

    @staticmethod
    def ffn_regularizer(
        x: Tensor,
        w_ff_1: Tensor,
        w_ff_2: Tensor,
        dropout_p: float,
        token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # J_ff = sum_{m=1}^2 1/2 Tr[((X^T X) ⊙ p^2) (W_ff,m^T W_ff,m)]
        gram = ExplicitDropoutRegularizer._feature_gram(x, token_mask)  # [B, D, D]
        b = gram * (dropout_p**2)

        wtw1 = w_ff_1.t() @ w_ff_1
        wtw2 = w_ff_2.t() @ w_ff_2

        reg1 = 0.5 * ExplicitDropoutRegularizer._batch_trace_product(b, wtw1.expand_as(b))
        reg2 = 0.5 * ExplicitDropoutRegularizer._batch_trace_product(b, wtw2.expand_as(b))
        return (reg1 + reg2).mean()


class ExplicitSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # x: [B, S, D]
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head format: [B, H, S, Hd]
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # S = Q K^T / sqrt(d_h)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, S, S]

        if attention_mask is not None:
            # attention_mask: [B, S], True/1 for valid tokens
            mask = attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # A = softmax(S)
        attn = F.softmax(scores, dim=-1)

        # Attention(Q, K, V) = A V
        context = torch.matmul(attn, v)  # [B, H, S, Hd]
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(context)
        return out, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # FF(X) = σ(X W_ff,1) W_ff,2
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ExplicitTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        reg_coeffs: RegularizationCoefficients,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = ExplicitSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = PositionwiseFeedForward(embed_dim=embed_dim, hidden_dim=ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.reg_coeffs = reg_coeffs

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Keep a copy for regularization terms that depend on the layer input X
        x_in = x

        attn_out, attn_weights = self.self_attn(x, attention_mask=attention_mask)
        x = self.norm1(x + attn_out)

        ffn_in = x
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        reg_loss = x.new_tensor(0.0)

        # J_q = 1/2 Tr[((X^T X) ⊙ p^2) Λ_q]
        if self.reg_coeffs.lambda_q > 0.0:
            reg_loss = reg_loss + self.reg_coeffs.lambda_q * ExplicitDropoutRegularizer.query_regularizer(
                x=x_in,
                w_q=self.self_attn.q_proj.weight,
                w_k=self.self_attn.k_proj.weight,
                dropout_p=self.reg_coeffs.dropout_p,
                token_mask=attention_mask,
            )

        # J_k = 1/2 Tr[((X^T X) ⊙ p^2) Λ_k]
        if self.reg_coeffs.lambda_k > 0.0:
            reg_loss = reg_loss + self.reg_coeffs.lambda_k * ExplicitDropoutRegularizer.key_regularizer(
                x=x_in,
                w_q=self.self_attn.q_proj.weight,
                w_k=self.self_attn.k_proj.weight,
                dropout_p=self.reg_coeffs.dropout_p,
                token_mask=attention_mask,
            )

        # J_v = 1/2 Tr[((X^T X) ⊙ p^2) (W_v^T W_v)]
        # J_av = 1/2 Tr[((X^T A^T A X) ⊙ p^2) (W_v^T W_v)]
        if self.reg_coeffs.lambda_v > 0.0:
            if self.reg_coeffs.value_mode == ValueRegularizerMode.TOKEN:
                v_reg = ExplicitDropoutRegularizer.value_regularizer_token(
                    x=x_in,
                    w_v=self.self_attn.v_proj.weight,
                    dropout_p=self.reg_coeffs.dropout_p,
                    token_mask=attention_mask,
                )
            else:
                v_reg = ExplicitDropoutRegularizer.value_regularizer_attention(
                    x=x_in,
                    attn_weights=attn_weights,
                    w_v=self.self_attn.v_proj.weight,
                    dropout_p=self.reg_coeffs.dropout_p,
                    token_mask=attention_mask,
                )
            reg_loss = reg_loss + self.reg_coeffs.lambda_v * v_reg

        # J_ff = sum_m 1/2 Tr[((X^T X) ⊙ p^2) (W_ff,m^T W_ff,m)]
        if self.reg_coeffs.lambda_ff > 0.0:
            reg_loss = reg_loss + self.reg_coeffs.lambda_ff * ExplicitDropoutRegularizer.ffn_regularizer(
                x=ffn_in,
                w_ff_1=self.ffn.fc1.weight,
                w_ff_2=self.ffn.fc2.weight,
                dropout_p=self.reg_coeffs.dropout_p,
                token_mask=attention_mask,
            )

        return x, reg_loss


class ExplicitTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        max_seq_len: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        reg_coeffs: Optional[Sequence[RegularizationCoefficients]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Identity() if input_dim == embed_dim else nn.Linear(input_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))

        if reg_coeffs is None:
            reg_coeffs = [RegularizationCoefficients() for _ in range(num_layers)]
        if len(reg_coeffs) != num_layers:
            raise ValueError("reg_coeffs length must match num_layers")

        self.layers = nn.ModuleList(
            [
                ExplicitTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    reg_coeffs=reg_coeffs[i],
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # x: [B, S, input_dim]
        bsz, seq_len, _ = x.shape
        x = self.input_proj(x)

        cls = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B, S+1, D]
        x = x + self.pos_embed[:, : seq_len + 1, :]

        if attention_mask is not None:
            # Prepend CLS token as always valid
            cls_mask = torch.ones((bsz, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        total_reg = x.new_tensor(0.0)
        for layer in self.layers:
            x, layer_reg = layer(x, attention_mask=attention_mask)
            total_reg = total_reg + layer_reg

        x = self.final_norm(x)
        logits = self.head(x[:, 0])  # CLS token
        return logits, total_reg


def compute_total_loss(
    logits: Tensor,
    targets: Tensor,
    reg_loss: Tensor,
    task_loss_fn: nn.Module,
) -> Tensor:
    # J_final = J_task + J_q + J_k + J_v + J_ff
    task_loss = task_loss_fn(logits, targets)
    return task_loss + reg_loss


def build_example_model() -> ExplicitTransformerClassifier:
    # Example: 2-layer encoder, explicit regularization coefficients can differ per layer.
    layer_cfgs = [
        RegularizationCoefficients(lambda_q=1e-4, lambda_k=1e-4, lambda_v=5e-4, lambda_ff=1e-4, dropout_p=0.2),
        RegularizationCoefficients(lambda_q=1e-4, lambda_k=1e-4, lambda_v=5e-4, lambda_ff=1e-4, dropout_p=0.2),
    ]
    model = ExplicitTransformerClassifier(
        input_dim=64,
        num_classes=10,
        max_seq_len=128,
        embed_dim=128,
        num_heads=4,
        ff_hidden_dim=256,
        num_layers=2,
        reg_coeffs=layer_cfgs,
    )
    return model


if __name__ == "__main__":
    torch.manual_seed(42)

    model = build_example_model()
    model.train()

    # Dummy batch: [B, S, input_dim]
    x = torch.randn(8, 128, 64)
    y = torch.randint(0, 10, (8,))

    # Optional token mask: 1 for valid tokens, 0 for padding
    attention_mask = torch.ones(8, 128, dtype=torch.bool)

    logits, reg_loss = model(x, attention_mask=attention_mask)
    criterion = nn.CrossEntropyLoss()
    loss = compute_total_loss(logits, y, reg_loss, criterion)

    loss.backward()

    print(f"logits shape: {logits.shape}")
    print(f"task+regularization loss: {loss.item():.4f}")
    print(f"regularization loss: {reg_loss.item():.4f}")