from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def exact_polynomial_kernel(
    q: Tensor,
    k: Tensor,
    degree: int,
    eps: float = 0.0,
) -> Tensor:
    # Eq. (polynomial kernel): κ(q, k) = (q^T k)^p
    # q: (..., d), k: (..., d) or (n, d) / (m, d)
    if q.dim() == 2 and k.dim() == 2:
        return (q @ k.transpose(-1, -2) + eps).pow(degree)
    if q.shape != k.shape:
        raise ValueError("For elementwise kernel evaluation, q and k must have the same shape.")
    return (torch.sum(q * k, dim=-1) + eps).pow(degree)


def rmae(
    target: Tensor,
    pred: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    # Eq. (15): RMAE = E |(target - pred) / target|
    denom = target.abs().clamp_min(eps)
    return ((target - pred).abs() / denom).mean()


class LearnableLowRankPolynomialSketch(nn.Module):
    """
    Learnable Low-Rank Polynomial Sketch (LLoPS).

    This module implements the core feature map:
        ϕ(x) = sum_{r=1}^t ∏_{j=1}^p (x Θ_{j,r})
    and the paper's practical setting uses t = 1:
        ϕ(x) = ∏_{j=1}^p (x Θ_j)

    Shapes:
        x: (..., d)
        Θ_j: (d, m)
        output: (..., m)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        degree: int,
        rank: int = 1,
        non_negative: bool = False,
    ) -> None:
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.rank = rank
        self.non_negative = non_negative

        # Learnable projection factors.
        # Eq. (11): ϕQ(Q) = ∏_{j=1}^p QΘ_j,  ϕK(K) = ∏_{j=1}^p KΘ_j
        # Here we keep a general CP-style rank t by summing over rank components.
        self.theta = nn.Parameter(torch.empty(rank, degree, in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # TODO: Explore initialization schemes that better approximate W* from the paper.
        for r in range(self.rank):
            for j in range(self.degree):
                nn.init.xavier_uniform_(self.theta[r, j])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., in_dim)

        Returns:
            Tensor of shape (..., out_dim)
        """
        if x.size(-1) != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.size(-1)}")

        rank_feats = []
        for r in range(self.rank):
            # Eq. (8): ψp(Q)W = sum_r (QΘ_{1,r}) ⊙ ... ⊙ (QΘ_{p,r})
            projections = []
            for j in range(self.degree):
                proj = torch.einsum("...d,dm->...m", x, self.theta[r, j])
                projections.append(proj)

            feat_r = torch.stack(projections, dim=0).prod(dim=0)
            rank_feats.append(feat_r)

        feat = torch.stack(rank_feats, dim=0).sum(dim=0)

        # Optional non-negativity trick mentioned in the paper:
        # ϕ(x) = (∏_j xΘ_j)^2
        if self.non_negative:
            feat = feat.square()

        return feat


class LLoPSAttention(nn.Module):
    """
    Linear attention using Learnable Low-Rank Polynomial Sketch (LLoPS).

    Core computation from the paper:
        Attn(Q, K, V) = ϕQ(Q) (ϕK(K)^T V)

    With optional normalization commonly used in linear attention:
        out_i = (ϕQ(q_i) · sum_j ϕK(k_j) v_j^T) / (ϕQ(q_i) · sum_j ϕK(k_j) + eps)

    This module provides both causal and non-causal forward passes.
    """

    def __init__(
        self,
        embed_dim: int,
        value_dim: int,
        sketch_dim: int,
        degree: int,
        rank: int = 1,
        non_negative: bool = False,
        normalize: bool = True,
        causal: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.sketch_dim = sketch_dim
        self.degree = degree
        self.rank = rank
        self.normalize = normalize
        self.causal = causal
        self.eps = eps

        # Separate learnable feature maps for queries and keys:
        # ϕQ(Q), ϕK(K): R^{n×d} -> R^{n×m}
        self.q_sketch = LearnableLowRankPolynomialSketch(
            in_dim=embed_dim,
            out_dim=sketch_dim,
            degree=degree,
            rank=rank,
            non_negative=non_negative,
        )
        self.k_sketch = LearnableLowRankPolynomialSketch(
            in_dim=embed_dim,
            out_dim=sketch_dim,
            degree=degree,
            rank=rank,
            non_negative=non_negative,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            q: (batch, seq_len, embed_dim)
            k: (batch, seq_len, embed_dim)
            v: (batch, seq_len, value_dim)
            attn_mask: optional mask of shape (batch, seq_len), where 1/True means keep

        Returns:
            Tensor of shape (batch, seq_len, value_dim)
        """
        if q.shape != k.shape:
            raise ValueError("q and k must have the same shape.")
        if q.dim() != 3:
            raise ValueError("Expected q, k to have shape (batch, seq_len, embed_dim).")
        if v.dim() != 3:
            raise ValueError("Expected v to have shape (batch, seq_len, value_dim).")
        if q.size(0) != v.size(0) or q.size(1) != v.size(1):
            raise ValueError("q/k and v must agree on batch and sequence dimensions.")

        q_feat = self.q_sketch(q)  # ϕQ(Q)
        k_feat = self.k_sketch(k)  # ϕK(K)

        if attn_mask is not None:
            # TODO: Support more general masking patterns (e.g., padding + block masks).
            mask = attn_mask.to(dtype=q_feat.dtype).unsqueeze(-1)  # (B, N, 1)
            q_feat = q_feat * mask
            k_feat = k_feat * mask
            v = v * mask

        if self.causal:
            return self._causal_linear_attention(q_feat=q_feat, k_feat=k_feat, v=v)

        return self._non_causal_linear_attention(q_feat=q_feat, k_feat=k_feat, v=v)

    def _non_causal_linear_attention(self, q_feat: Tensor, k_feat: Tensor, v: Tensor) -> Tensor:
        # Eq. (linear attention): ϕQ(Q)(ϕK(K)^T V)
        # kv: (B, M, Dv)
        kv = torch.einsum("bnm,bnd->bmd", k_feat, v)
        out = torch.einsum("bnm,bmd->bnd", q_feat, kv)

        if self.normalize:
            # Standard linear-attention normalization:
            # denom_n = ϕQ(q_n) · sum_i ϕK(k_i)
            k_sum = k_feat.sum(dim=1)  # (B, M)
            denom = torch.einsum("bnm,bm->bn", q_feat, k_sum).clamp_min(self.eps)
            out = out / denom.unsqueeze(-1)

        return out

    def _causal_linear_attention(self, q_feat: Tensor, k_feat: Tensor, v: Tensor) -> Tensor:
        # Causal linear attention uses prefix sums:
        #   S_n = sum_{i<=n} ϕK(k_i) v_i^T
        #   z_n = sum_{i<=n} ϕK(k_i)
        #   out_n = ϕQ(q_n) S_n / (ϕQ(q_n) z_n + eps)
        #
        # This is a practical implementation of the associative trick described in the paper.
        kv = k_feat.unsqueeze(-1) * v.unsqueeze(-2)  # (B, N, M, Dv)
        kv_prefix = torch.cumsum(kv, dim=1)  # (B, N, M, Dv)
        out = torch.einsum("bnm,bnmd->bnd", q_feat, kv_prefix)

        if self.normalize:
            k_prefix = torch.cumsum(k_feat, dim=1)  # (B, N, M)
            denom = torch.einsum("bnm,bnm->bn", q_feat, k_prefix).clamp_min(self.eps)
            out = out / denom.unsqueeze(-1)

        return out


class LLoPSMultiHeadAttention(nn.Module):
    """
    Optional multi-head wrapper for LLoPS.

    TODO:
      - Add per-head projections for Q/K/V
      - Add output projection
      - Add dropout and residual-friendly normalization
    """

    def __init__(
        self,
        embed_dim: int,
        value_dim: int,
        num_heads: int,
        sketch_dim: int,
        degree: int,
        rank: int = 1,
        non_negative: bool = False,
        normalize: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.head_value_dim = value_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)

        self.attn = LLoPSAttention(
            embed_dim=self.head_dim,
            value_dim=self.head_value_dim,
            sketch_dim=sketch_dim,
            degree=degree,
            rank=rank,
            non_negative=non_negative,
            normalize=normalize,
            causal=causal,
        )
        self.out_proj = nn.Linear(value_dim, value_dim, bias=False)

    def forward(
        self,
        x_q: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x_q: (B, N, E)
            x_k: (B, N, E)
            x_v: (B, N, E)

        Returns:
            (B, N, value_dim)
        """
        bsz, seq_len, _ = x_q.shape

        q = self.q_proj(x_q).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_k).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_v).view(bsz, seq_len, self.num_heads, self.head_value_dim).transpose(1, 2)

        # Merge heads into batch for simple reuse of the attention module.
        q = q.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, seq_len, self.head_value_dim)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        out = self.attn(q=q, k=k, v=v, attn_mask=attn_mask)
        out = out.view(bsz, self.num_heads, seq_len, self.head_value_dim).transpose(1, 2)
        out = out.reshape(bsz, seq_len, self.value_dim)

        return self.out_proj(out)


def compute_sketch_approximation_loss(
    q: Tensor,
    k: Tensor,
    sketch_q: nn.Module,
    sketch_k: nn.Module,
    degree: int,
) -> Tensor:
    """
    Utility for measuring the approximation error in Eq. (2).

    Args:
        q: (n, d)
        k: (n, d)
        sketch_q: feature map module for queries
        sketch_k: feature map module for keys
        degree: polynomial degree p

    Returns:
        Frobenius norm squared error between:
            ϕQ(Q)ϕK(K)^T and (QK^T)^{◦p}
    """
    q_feat = sketch_q(q)
    k_feat = sketch_k(k)
    approx = q_feat @ k_feat.transpose(-1, -2)
    target = (q @ k.transpose(-1, -2)).pow(degree)
    return torch.norm(approx - target, p="fro").pow(2)


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 8
    embed_dim = 16
    value_dim = 32
    sketch_dim = 12
    degree = 4

    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, value_dim)

    # Core LLoPS attention
    attn = LLoPSAttention(
        embed_dim=embed_dim,
        value_dim=value_dim,
        sketch_dim=sketch_dim,
        degree=degree,
        rank=1,
        non_negative=False,
        normalize=True,
        causal=False,
    )

    out = attn(q, k, v)
    print("LLoPSAttention output shape:", out.shape)

    # Multi-head wrapper
    mha = LLoPSMultiHeadAttention(
        embed_dim=embed_dim,
        value_dim=value_dim,
        num_heads=4,
        sketch_dim=sketch_dim,
        degree=degree,
        rank=1,
        non_negative=False,
        normalize=True,
        causal=False,
    )
    out_mha = mha(q, k, q)
    print("LLoPSMultiHeadAttention output shape:", out_mha.shape)

    # Approximation diagnostic
    sketch_loss = compute_sketch_approximation_loss(
        q=q[0],
        k=k[0],
        sketch_q=attn.q_sketch,
        sketch_k=attn.k_sketch,
        degree=degree,
    )
    print("Approximation loss:", float(sketch_loss))

    # Exact polynomial kernel example
    exact = exact_polynomial_kernel(q[0], k[0], degree=degree)
    print("Exact polynomial kernel shape:", exact.shape)

    # RMAE example against a random prediction
    pred = exact + 0.01 * torch.randn_like(exact)
    print("RMAE:", float(rmae(exact, pred)))


if __name__ == "__main__":
    main()