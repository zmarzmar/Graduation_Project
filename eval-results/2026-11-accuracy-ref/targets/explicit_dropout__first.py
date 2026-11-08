from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


LayerwiseCoeff = Union[float, Sequence[float]]
ValueRegularizerMode = Literal["token", "attn"]


def _resolve_layer_coeff(coeff: LayerwiseCoeff, layer_idx: int) -> float:
    # Layer-wise λ^(l) in the paper:
    # J_q = sum_l λ_q^(l) J_q^(l), similarly for J_k, J_v, J_ff
    if isinstance(coeff, (float, int)):
        return float(coeff)
    if layer_idx >= len(coeff):
        return float(coeff[-1])
    return float(coeff[layer_idx])


def _flatten_tokens(x: Tensor) -> Tensor:
    # x: [B, T, D] -> [B*T, D]
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, T, D], got shape {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-1])


def _empirical_gram(x: Tensor) -> Tensor:
    # Empirical second moment / Gram matrix:
    # X^T X (normalized by number of tokens for numerical stability)
    x_flat = _flatten_tokens(x)
    n_tokens = x_flat.shape[0]
    return (x_flat.transpose(0, 1) @ x_flat) / max(n_tokens, 1)


def _trace_product(a: Tensor, b: Tensor) -> Tensor:
    # Tr(A B) = sum_ij A_ij B_ji
    return torch.einsum("ij,ji->", a, b)


class ExplicitDropoutRegularizer(nn.Module):
    def __init__(
        self,
        drop_prob: float = 0.2,
        lambda_q: LayerwiseCoeff = 0.0,
        lambda_k: LayerwiseCoeff = 0.0,
        lambda_v: LayerwiseCoeff = 0.0,
        lambda_ff: LayerwiseCoeff = 0.0,
        value_mode: ValueRegularizerMode = "token",
    ) -> None:
        super().__init__()
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError("drop_prob must be in [0, 1).")
        self.drop_prob = float(drop_prob)
        self.lambda_q = lambda_q
        self.lambda_k = lambda_k
        self.lambda_v = lambda_v
        self.lambda_ff = lambda_ff
        self.value_mode = value_mode

    def _base_matrix(self, x: Tensor) -> Tensor:
        # B ≈ (X^T X) ⊙ p^2  -> for scalar p, B = p^2 * (X^T X)
        gram = _empirical_gram(x)
        return (self.drop_prob ** 2) * gram

    def query_regularizer(self, x: Tensor, w_q: Tensor, w_k: Tensor) -> Tensor:
        # J_q = 1/2 Tr[((X^T X) ⊙ p^2) Λ_q]
        # Λ_q = W_q^T W_k X^T X W_k^T W_q
        gram = _empirical_gram(x)
        base = self._base_matrix(x)
        lambda_q = w_q.transpose(0, 1) @ w_k @ gram @ w_k.transpose(0, 1) @ w_q
        return 0.5 * _trace_product(base, lambda_q)

    def key_regularizer(self, x: Tensor, w_q: Tensor, w_k: Tensor) -> Tensor:
        # J_k = 1/2 Tr[((X^T X) ⊙ p^2) Λ_k]
        # Λ_k = W_k^T W_q X^T X W_q^T W_k
        gram = _empirical_gram(x)
        base = self._base_matrix(x)
        lambda_k = w_k.transpose(0, 1) @ w_q @ gram @ w_q.transpose(0, 1) @ w_k
        return 0.5 * _trace_product(base, lambda_k)

    def value_regularizer(
        self,
        x: Tensor,
        w_v: Tensor,
        attn_weights: Optional[Tensor] = None,
    ) -> Tensor:
        # Token-level:
        # J_v = 1/2 Tr[((X^T X) ⊙ p^2) (W_v^T W_v)]
        # Attention-conditioned:
        # J_av = 1/2 Tr[((X^T A^T A X) ⊙ p^2) (W_v^T W_v)]
        if self.value_mode == "attn":
            if attn_weights is None:
                raise ValueError("attn_weights must be provided when value_mode='attn'.")

            # attn_weights: [B, H, T, T]
            # Y = A^T A, averaged over heads -> [B, T, T]
            y = attn_weights.transpose(-1, -2) @ attn_weights
            y = y.mean(dim=1)

            # psi = X^T Y X averaged over batch
            # x: [B, T, D] -> psi: [D, D]
            x_t = x.transpose(1, 2)  # [B, D, T]
            psi = torch.matmul(x_t, torch.matmul(y, x)).mean(dim=0) / max(x.shape[1], 1)
            base = (self.drop_prob ** 2) * psi
        else:
            base = self._base_matrix(x)

        lambda_v = w_v.transpose(0, 1) @ w_v
        return 0.5 * _trace_product(base, lambda_v)

    def feedforward_regularizer(self, x: Tensor, w_ff1: Tensor, w_ff2: Tensor) -> Tensor:
        # J_ff,m = 1/2 Tr[((X^T X) ⊙ p^2) (W_ff,m^T W_ff,m)]
        base = self._base_matrix(x)
        lambda_ff1 = w_ff1.transpose(0, 1) @ w_ff1
        lambda_ff2 = w_ff2.transpose(0, 1) @ w_ff2
        reg1 = 0.5 * _trace_product(base, lambda_ff1)
        reg2 = 0.5 * _trace_product(base, lambda_ff2)
        return reg1 + reg2

    def layer_regularizer(
        self,
        x: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        w_ff1: Tensor,
        w_ff2: Tensor,
        layer_idx: int = 0,
        attn_weights: Optional[Tensor] = None,
    ) -> Tensor:
        # J_final = J_task + J_q + J_k + J_v + J_ff
        coeff_q = _resolve_layer_coeff(self.lambda_q, layer_idx)
        coeff_k = _resolve_layer_coeff(self.lambda_k, layer_idx)
        coeff_v = _resolve_layer_coeff(self.lambda_v, layer_idx)
        coeff_ff = _resolve_layer_coeff(self.lambda_ff, layer_idx)

        reg_q = coeff_q * self.query_regularizer(x, w_q, w_k) if coeff_q != 0.0 else x.new_tensor(0.0)
        reg_k = coeff_k * self.key_regularizer(x, w_q, w_k) if coeff_k != 0.0 else x.new_tensor(0.0)
        reg_v = coeff_v * self.value_regularizer(x, w_v, attn_weights=attn_weights) if coeff_v != 0.0 else x.new_tensor(0.0)
        reg_ff = coeff_ff * self.feedforward_regularizer(x, w_ff1, w_ff2) if coeff_ff != 0.0 else x.new_tensor(0.0)

        return reg_q + reg_k + reg_v + reg_ff


class MultiHeadSelfAttentionExplicit(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Standard scaled dot-product attention:
        # S = QK^T / sqrt(d_h), A = softmax(S), Attention(Q,K,V) = AV
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # [B, H, T, head_dim]
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(context)
        return out, attn


class FeedForwardNetworkExplicit(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.ff1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ff2 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # FF(X) = σ(X W_ff,1) W_ff,2
        return self.ff2(F.gelu(self.ff1(x)))


class TransformerEncoderLayerExplicit(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttentionExplicit(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = FeedForwardNetworkExplicit(embed_dim=embed_dim, hidden_dim=ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Pre-norm transformer block
        x_in = x
        attn_out, attn_weights = self.self_attn(self.norm1(x))
        x = x + attn_out
        ff_out = self.ffn(self.norm2(x))
        x = x + ff_out
        return x, x_in, attn_weights


class TransformerEncoderExplicit(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerExplicit(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        regularizer: Optional[ExplicitDropoutRegularizer] = None,
    ) -> Tuple[Tensor, Tensor]:
        total_reg = x.new_tensor(0.0)
        for layer_idx, layer in enumerate(self.layers):
            x, x_in, attn_weights = layer(x)
            if regularizer is not None:
                total_reg = total_reg + regularizer.layer_regularizer(
                    x=x_in,
                    w_q=layer.self_attn.q_proj.weight,
                    w_k=layer.self_attn.k_proj.weight,
                    w_v=layer.self_attn.v_proj.weight,
                    w_ff1=layer.ffn.ff1.weight,
                    w_ff2=layer.ffn.ff2.weight,
                    layer_idx=layer_idx,
                    attn_weights=attn_weights,
                )
        return x, total_reg


class ExplicitDropoutTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_hidden_dim: int = 64,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, input_dim))
        self.encoder = TransformerEncoderExplicit(
            num_layers=num_layers,
            embed_dim=input_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: Tensor,
        regularizer: Optional[ExplicitDropoutRegularizer] = None,
    ) -> Tuple[Tensor, Tensor]:
        # x: [B, T, D]
        bsz, seq_len, dim = x.shape
        if dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {dim}")
        if seq_len + 1 > self.max_seq_len + 1:
            raise ValueError("Sequence length exceeds max_seq_len.")

        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : seq_len + 1, :]

        x, reg_loss = self.encoder(x, regularizer=regularizer)
        x = self.norm(x)
        logits = self.head(x[:, 0])  # CLS token
        return logits, reg_loss




if __name__ == "__main__":
    result = smoke_test()
    print({k: [tuple(t.shape) for t in v] if k == "outputs" else len(v) for k, v in result.items()})

def smoke_test() -> Dict[str, List[Tensor]]:
    torch.manual_seed(7)

    batch_size = 2
    seq_len = 8
    embed_dim = 16
    num_classes = 3

    model = ExplicitDropoutTransformerClassifier(
        input_dim=embed_dim,
        num_classes=num_classes,
        num_layers=2,
        num_heads=4,
        ff_hidden_dim=32,
        max_seq_len=16,
    )
    # Use all core paths: Q/K/V/FF regularizers and attention-conditioned value regularizer.
    regularizer = ExplicitDropoutRegularizer(
        drop_prob=0.2,
        lambda_q=[0.05, 0.05],
        lambda_k=[0.05, 0.05],
        lambda_v=[0.05, 0.05],
        lambda_ff=[0.05, 0.05],
        value_mode="attn",
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    y = torch.randint(0, num_classes, (batch_size,))

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    logits, reg_loss = model(x, regularizer=regularizer)
    task_loss = F.cross_entropy(logits, y)
    total_loss = task_loss + reg_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    grad_params: List[Tensor] = [
        model.encoder.layers[0].self_attn.q_proj.weight,
        model.encoder.layers[0].self_attn.k_proj.weight,
        model.encoder.layers[0].self_attn.v_proj.weight,
        model.encoder.layers[0].ffn.ff1.weight,
        model.encoder.layers[0].ffn.ff2.weight,
    ]

    return {
        "outputs": [logits.detach(), reg_loss.detach(), total_loss.detach()],
        "grad_params": grad_params,
    }
