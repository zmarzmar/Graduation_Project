from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PositionalEncodingFn = Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]
SoftmaxFn = Callable[[Tensor, Optional[Tensor]], Tensor]


def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def _to_pairwise_mask(attention_mask: Optional[Tensor], seq_len: int) -> Optional[Tensor]:
    if attention_mask is None:
        return None

    mask = attention_mask.to(dtype=torch.bool)
    if mask.dim() == 2:
        # [B, T] -> pairwise mask [B, 1, T, T]
        mask = mask[:, None, :, None] & mask[:, None, None, :]
    elif mask.dim() == 3:
        # [B, T, T] -> [B, 1, T, T]
        mask = mask[:, None, :, :]
    elif mask.dim() == 4:
        # Already pairwise
        pass
    else:
        raise ValueError(f"Unsupported attention_mask shape: {tuple(mask.shape)}")

    if mask.shape[-1] != seq_len or mask.shape[-2] != seq_len:
        raise ValueError(
            f"attention_mask must align with seq_len={seq_len}, got {tuple(mask.shape)}"
        )
    return mask


def masked_softmax(scores: Tensor, mask: Optional[Tensor], dim: int = -1) -> Tensor:
    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
    return torch.softmax(scores, dim=dim)


class IdentityPositionalEncoding(nn.Module):
    def forward(
        self, q: Tensor, k: Tensor, r: Tensor, norm_r: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return q, k, r, norm_r


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class Cubit(nn.Module):
    """
    Cubit token mixer with Kernel Ridge Regression (KRR).

    Core idea:
      # Eq. (9): Z^(h) = A^(h) O = A^(h) Σ^(h) S^(h) V^(h)
      # Eq. (10): \hat{s}_i^(h) = α * sigmoid(W_s x_i) + β
      # KRR-inspired normalization: Σ^(h) = (g(KK^T) + λI)^(-1)

    This implementation follows the paper's appendix-style PyTorch sketch.
    """

    def __init__(
        self,
        hidden_size: int,
        attention_head: int,
        eps: float = 1e-10,
        upper: float = 2.0,
        lower: float = 0.5,
        share: bool = False,
        causal_mask: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % attention_head != 0:
            raise ValueError("hidden_size must be divisible by attention_head")

        self.hidden_size = hidden_size
        self.attention_head = attention_head
        self.hidden_size_per_head = hidden_size // attention_head
        self.share = share
        self.causal_mask = causal_mask
        self.eps = eps

        # Linear projections for Q, K, V
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        # Optional separate R projection
        if not share:
            self.r = nn.Linear(hidden_size, hidden_size)

        # Limited-Range Rescale bounds
        self.lower = nn.Parameter(torch.full((1, attention_head, 1, 1), lower))
        self.upper_scale = nn.Parameter(torch.full((1, attention_head, 1, 1), upper - lower))

        # Per-head LRR factor
        self.lrr = nn.Linear(hidden_size, attention_head)

        # Learnable scale for normalized R
        self.scale = nn.Parameter(torch.ones(1, attention_head, 1, 1))

        # Regularization strength λ in log-space for stability
        self.log_lambda = nn.Parameter(
            torch.full((1, attention_head, 1, 1), math.log(eps)), requires_grad=True
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        pos_encoding_function: Optional[PositionalEncodingFn] = None,
        softmax_fn: Optional[SoftmaxFn] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b, t, d = x.shape
        device = x.device

        if pos_encoding_function is None:
            pos_encoding_function = lambda q, k, r, nr: (q, k, r, nr)
        if softmax_fn is None:
            softmax_fn = masked_softmax

        # Project to multi-head Q, K, V
        q = self.q(x).reshape(b, t, self.attention_head, self.hidden_size_per_head).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, t, self.attention_head, self.hidden_size_per_head).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, t, self.attention_head, self.hidden_size_per_head).permute(0, 2, 1, 3)

        # Determine R: separate projection or shared with K
        if self.share:
            r = k
        else:
            r = self.r(x).reshape(b, t, self.attention_head, self.hidden_size_per_head).permute(0, 2, 1, 3)

        # Eq. (10): Limited-Range Rescale factor in (lower, upper)
        lrr_logits = self.lrr(x).reshape(b, t, self.attention_head, 1).permute(0, 2, 1, 3)
        lrr = self.lower + self.upper_scale * torch.sigmoid(lrr_logits)

        # Normalize R with learnable scale
        # norm_r = r / ||r||_2 * scale
        norm_r = r / r.norm(dim=-1, p=2, keepdim=True).clamp_min(self.eps) * self.scale

        # Optional positional encoding hook
        q, k, r, norm_r = pos_encoding_function(q, k, r, norm_r)

        # KRR normalization matrix:
        # Σ^(h) = (g(R * norm(R)^T) + λ I)^(-1)
        sigma = torch.matmul(r, norm_r.transpose(-2, -1))
        mask = _to_pairwise_mask(attention_mask, t)

        if self.causal_mask:
            causal = _causal_mask(t, device=device).view(1, 1, t, t)
            mask = causal if mask is None else (mask & causal)

        sigma = softmax_fn(sigma, mask)

        # Add λI
        identity = torch.eye(t, device=device, dtype=x.dtype).view(1, 1, t, t)
        lambda_reg = torch.exp(self.log_lambda).to(dtype=x.dtype)
        sigma = sigma + lambda_reg * identity

        # Eq. (11)-(12): solve Σ o = S V
        rhs = lrr * v
        solution = torch.linalg.solve(sigma, rhs)

        # Standard similarity-based aggregation:
        # A = Softmax(Q K^T / sqrt(dh))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size_per_head)
        attn_weights = softmax_fn(attn_scores, mask)

        # Eq. (13): Z = A O
        output = torch.matmul(attn_weights, solution)
        output = output.permute(0, 2, 1, 3).reshape(b, t, self.hidden_size)
        output = self.dropout(output)
        return self.out_proj(output)


def local_linear_regression(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    attention_mask: Optional[Tensor],
    weight_func: SoftmaxFn,
    regularization_eps: float = 1.0,
) -> Tensor:
    """
    Local Linear Regression (LLR) token mixer.

    # Eq. (14): min_{β0, β1} Σ_j K_ij || v_j - β0 - β1^T (x_j - x_i) ||^2
    # Eq. (15): Z_LLR = S V,  S = e_1^T (X_i^T W_i X_i)^(-1) X_i^T W_i

    Args:
        Q: [B, T, D]
        K: [B, T, D]
        V: [B, T, D]
        attention_mask: optional attention mask
        weight_func: function that maps similarity -> weights, e.g. softmax
        regularization_eps: ridge penalty for slope terms

    Returns:
        Output tensor of shape [B, T, D]
    """
    batch_size, seq_len, dim = Q.shape
    scaling_factor = dim ** 0.5

    # Step 1: similarity scores
    similarity = torch.bmm(Q / scaling_factor, K.transpose(-2, -1))  # [B, T, T]
    similarity = similarity.unsqueeze(1)  # [B, 1, T, T]
    W = weight_func(similarity, _to_pairwise_mask(attention_mask, seq_len))  # [B, 1, T, T]
    W = W.squeeze(1)  # [B, T, T]

    # Step 2: augmented design matrix M = [1, K]
    ones = torch.ones(batch_size, seq_len, 1, device=Q.device, dtype=Q.dtype)
    M = torch.cat([ones, K], dim=-1)  # [B, T, D+1]

    # Step 3: weighted Gram matrix H[b, i, a, c] = Σ_j W[b,i,j] M[b,j,a] M[b,j,c]
    H = torch.einsum("bij,bja,bjc->biac", W, M, M)

    # Step 4: weighted moment matrix G[b, i, a, d] = Σ_j W[b,i,j] M[b,j,a] V[b,j,d]
    G = torch.einsum("bij,bja,bjd->biad", W, M, V)

    # Step 5: ridge regularization on slope terms only
    reg_matrix = torch.eye(1 + dim, device=H.device, dtype=H.dtype)
    reg_matrix[0, 0] = 0.0
    H = H + reg_matrix.view(1, 1, 1 + dim, 1 + dim) * regularization_eps

    # Step 6: solve theta = (M^T W M + λI)^(-1) M^T W V
    regression_coeffs = torch.linalg.solve(H, G)  # [B, T, D+1, D]

    # Step 7: evaluate local linear model at query positions
    intercept = regression_coeffs[:, :, 0, :]      # [B, T, D]
    slope = regression_coeffs[:, :, 1:, :]         # [B, T, D, D]
    output = intercept + torch.einsum("btdo,btd->bto", slope, Q)
    return output


class CubitBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        eps: float = 1e-10,
        share: bool = False,
        causal_mask: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.mixer = Cubit(
            hidden_size=hidden_size,
            attention_head=attention_head,
            eps=eps,
            share=share,
            causal_mask=causal_mask,
            dropout=dropout,
        )
        self.mlp = FeedForward(hidden_size=hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        pos_encoding_function: Optional[PositionalEncodingFn] = None,
        softmax_fn: Optional[SoftmaxFn] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.mixer(
            self.norm1(x),
            pos_encoding_function=pos_encoding_function,
            softmax_fn=softmax_fn,
            attention_mask=attention_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class CubitModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        attention_head: int,
        num_layers: int,
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        eps: float = 1e-10,
        share: bool = False,
        causal_mask: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                CubitBlock(
                    hidden_size=hidden_size,
                    attention_head=attention_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    eps=eps,
                    share=share,
                    causal_mask=causal_mask,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        pos_encoding_function: Optional[PositionalEncodingFn] = None,
        softmax_fn: Optional[SoftmaxFn] = None,
    ) -> Tensor:
        """
        Args:
            input_ids: [B, T]
            attention_mask: optional mask, either [B, T] or pairwise [B, T, T] / [B, 1, T, T]
        Returns:
            logits: [B, T, vocab_size]
        """
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.max_seq_len}")

        pos_ids = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.drop(x)

        for block in self.blocks:
            x = block(
                x,
                pos_encoding_function=pos_encoding_function,
                softmax_fn=softmax_fn,
                attention_mask=attention_mask,
            )

        x = self.norm_f(x)
        return self.lm_head(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Minimal smoke test
    batch_size, seq_len, vocab_size = 2, 16, 1000
    model = CubitModel(
        vocab_size=vocab_size,
        hidden_size=128,
        attention_head=8,
        num_layers=2,
        max_seq_len=64,
        dropout=0.0,
        share=False,
        causal_mask=True,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    print("logits shape:", logits.shape)  # [B, T, vocab_size]

    # LLR smoke test
    Q = torch.randn(batch_size, seq_len, 32)
    K = torch.randn(batch_size, seq_len, 32)
    V = torch.randn(batch_size, seq_len, 32)
    llr_out = local_linear_regression(Q, K, V, attention_mask=None, weight_func=masked_softmax)
    print("llr_out shape:", llr_out.shape)  # [B, T, D]