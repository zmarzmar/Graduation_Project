from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor, nn


@dataclass
class QuantizationStats:
    aggregated_values: Tensor
    aggregated_counts: Tensor
    parent_batch_sums: Tensor
    parent_batch_counts: Tensor
    child_batch_sums: Tensor
    child_batch_counts: Tensor


class HierarchicalCodebook(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_parents: int,
        num_children: int,
        dim: int,
        ema_decay: float = 0.99,
        init_scale: float = 0.1,
        empty_threshold: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_parents = num_parents
        self.num_children = num_children
        self.dim = dim
        self.total_codes = num_parents * (1 + num_children)
        self.ema_decay = ema_decay
        self.init_scale = init_scale
        self.empty_threshold = empty_threshold

        parent_codes = torch.randn(num_heads, num_parents, dim) * init_scale
        child_codes = parent_codes.unsqueeze(2).expand(-1, -1, num_children, -1).clone()
        child_codes = child_codes + init_scale * torch.randn_like(child_codes)

        # EMA statistics for constrained online k-means (Sec. A)
        parent_sums = parent_codes.clone()
        parent_counts = torch.ones(num_heads, num_parents)

        child_sums = child_codes.clone()
        child_counts = torch.ones(num_heads, num_parents, num_children)

        self.register_buffer("parent_codes", parent_codes)
        self.register_buffer("child_codes", child_codes)
        self.register_buffer("parent_sums", parent_sums)
        self.register_buffer("parent_counts", parent_counts)
        self.register_buffer("child_sums", child_sums)
        self.register_buffer("child_counts", child_counts)

        with torch.no_grad():
            self._project_children()

    def _project_children(self) -> None:
        # Eq. (12): C_{p,c} = M_c - delta / (N_c * sigma)
        # where delta = sum_c M_c - C * C_p and sigma = sum_c 1 / N_c
        child_counts = self.child_counts.clamp_min(self.empty_threshold)
        child_means = self.child_sums / child_counts.unsqueeze(-1)

        delta = child_means.sum(dim=2) - self.num_children * self.parent_codes
        sigma = (1.0 / child_counts).sum(dim=2, keepdim=True).clamp_min(self.empty_threshold)

        correction = delta.unsqueeze(2) / (child_counts.unsqueeze(-1) * sigma.unsqueeze(-1))
        projected = child_means - correction

        # Keep the constrained positions used by attention.
        self.child_codes.copy_(projected)

    def quantize_and_aggregate(self, keys: Tensor, values: Tensor) -> QuantizationStats:
        # keys/values: [B, H, N, D]
        bsz, num_heads, num_tokens, dim = keys.shape
        assert num_heads == self.num_heads
        assert dim == self.dim

        # Step 1: assign each key to nearest parent codeword (Eq. (2))
        # Parent distances: [B, H, N, M0]
        parent_dist = (keys.unsqueeze(3) - self.parent_codes.unsqueeze(0).unsqueeze(2)).pow(2).sum(dim=-1)
        parent_idx = parent_dist.argmin(dim=-1)  # [B, H, N]
        parent_best_dist = parent_dist.gather(dim=-1, index=parent_idx.unsqueeze(-1)).squeeze(-1)

        # Step 2: refine within the selected parent's children
        # Gather child candidates for each key: [B, H, N, C, D]
        head_idx = torch.arange(self.num_heads, device=keys.device).view(1, self.num_heads, 1).expand(
            bsz, self.num_heads, num_tokens
        )
        child_candidates = self.child_codes[head_idx, parent_idx]
        child_dist = (keys.unsqueeze(-2) - child_candidates).pow(2).sum(dim=-1)  # [B, H, N, C]
        child_best_dist, child_rel_idx = child_dist.min(dim=-1)  # [B, H, N]

        # Child replaces parent only if it is closer
        use_child = child_best_dist < parent_best_dist
        final_idx = parent_idx.clone()
        final_idx[use_child] = (
            self.num_parents
            + parent_idx[use_child] * self.num_children
            + child_rel_idx[use_child]
        )

        # Aggregate values/counts for all codewords (Eq. (3))
        aggregated_values = values.new_zeros((bsz, self.num_heads, self.total_codes, dim))
        aggregated_counts = values.new_zeros((bsz, self.num_heads, self.total_codes))

        scatter_idx_v = final_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
        aggregated_values.scatter_add_(dim=2, index=scatter_idx_v, src=values)
        aggregated_counts.scatter_add_(
            dim=2,
            index=final_idx,
            src=torch.ones((bsz, self.num_heads, num_tokens), device=values.device, dtype=values.dtype),
        )

        # Parent stats (keys first quantized to parent, then optionally to child)
        parent_batch_sums = values.new_zeros((bsz, self.num_heads, self.num_parents, dim))
        parent_batch_counts = values.new_zeros((bsz, self.num_heads, self.num_parents))
        parent_batch_sums.scatter_add_(
            dim=2,
            index=parent_idx.unsqueeze(-1).expand(-1, -1, -1, dim),
            src=values,
        )
        parent_batch_counts.scatter_add_(
            dim=2,
            index=parent_idx,
            src=torch.ones((bsz, self.num_heads, num_tokens), device=values.device, dtype=values.dtype),
        )

        # Child stats only for keys that actually move to a child
        child_batch_sums = values.new_zeros((bsz, self.num_heads, self.num_parents * self.num_children, dim))
        child_batch_counts = values.new_zeros((bsz, self.num_heads, self.num_parents * self.num_children))
        if bool(use_child.any()):
            masked_values = values * use_child.unsqueeze(-1).to(values.dtype)
            child_flat_idx = parent_idx * self.num_children + child_rel_idx
            child_batch_sums.scatter_add_(
                dim=2,
                index=child_flat_idx.unsqueeze(-1).expand(-1, -1, -1, dim),
                src=masked_values,
            )
            child_batch_counts.scatter_add_(
                dim=2,
                index=child_flat_idx,
                src=use_child.to(values.dtype),
            )
        child_batch_sums = child_batch_sums.view(bsz, self.num_heads, self.num_parents, self.num_children, dim)
        child_batch_counts = child_batch_counts.view(bsz, self.num_heads, self.num_parents, self.num_children)

        return QuantizationStats(
            aggregated_values=aggregated_values,
            aggregated_counts=aggregated_counts,
            parent_batch_sums=parent_batch_sums,
            parent_batch_counts=parent_batch_counts,
            child_batch_sums=child_batch_sums,
            child_batch_counts=child_batch_counts,
        )

    @torch.no_grad()
    def ema_update(self, stats: QuantizationStats) -> None:
        # Eq. (11): EMA update of codeword statistics
        decay = self.ema_decay
        one_minus = 1.0 - decay

        self.parent_sums.mul_(decay).add_(stats.parent_batch_sums, alpha=one_minus)
        self.parent_counts.mul_(decay).add_(stats.parent_batch_counts, alpha=one_minus)

        parent_counts = self.parent_counts.clamp_min(self.empty_threshold)
        parent_means = self.parent_sums / parent_counts.unsqueeze(-1)
        self.parent_codes.copy_(parent_means)

        self.child_sums.mul_(decay).add_(stats.child_batch_sums, alpha=one_minus)
        self.child_counts.mul_(decay).add_(stats.child_batch_counts, alpha=one_minus)

        self._project_children()


class AVQAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim: int,
        num_parents: int,
        num_children: int,
        top_p: int,
        query_tile_size: int = 32,
        ema_decay: float = 0.99,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.codebook = HierarchicalCodebook(
            num_heads=num_heads,
            num_parents=num_parents,
            num_children=num_children,
            dim=dim,
            ema_decay=ema_decay,
            init_scale=init_scale,
        )
        self.top_p = top_p
        self.query_tile_size = query_tile_size

    def forward(self, q: Tensor, k: Tensor, v: Tensor, update_codebook: bool = False) -> Tensor:
        # q, k, v: [B, H, N, D]
        stats = self.codebook.quantize_and_aggregate(k, v)

        if update_codebook:
            # Training-time EMA update of the hierarchical codebook.
            self.codebook.ema_update(stats)

        bsz, num_heads, num_tokens, dim = q.shape
        num_parents = self.codebook.num_parents
        num_children = self.codebook.num_children
        top_p = min(max(self.top_p, 0), num_parents)
        eps = 1e-6

        parent_codes = self.codebook.parent_codes  # [H, M0, D]
        parent_values = stats.aggregated_values[:, :, :num_parents, :]  # [B, H, M0, D]
        parent_counts = stats.aggregated_counts[:, :, :num_parents]  # [B, H, M0]

        y = q.new_zeros((bsz, num_heads, num_tokens, dim))
        head_idx = torch.arange(num_heads, device=q.device).view(1, num_heads).expand(bsz, num_heads)

        # FlashAttention-style tiled computation over query blocks.
        # TODO: Replace this Python-level tiling with fused Triton kernels in the paper.
        for start in range(0, num_tokens, self.query_tile_size):
            end = min(num_tokens, start + self.query_tile_size)
            q_tile = q[:, :, start:end, :]  # [B, H, T, D]
            tile_len = q_tile.shape[2]

            # Parent attention logits (Eq. (6))
            logits = torch.einsum("bhtd,hmd->bhtm", q_tile, parent_codes)  # [B, H, T, M0]

            # Mask empty codes (Sec. D); here counts are generally positive but we keep the logic.
            non_empty = parent_counts > self.codebook.empty_threshold
            logits = logits.masked_fill(~non_empty.unsqueeze(2), float("-inf"))

            # Online softmax normalizer (FlashAttention-style)
            m = logits.max(dim=-1).values
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            weights = torch.exp(logits - m.unsqueeze(-1))
            weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

            # Parent numerator/denominator (Eq. (6))
            x = torch.einsum("bhtm,bhmd->bhtd", weights, parent_values)
            z = (weights * parent_counts.unsqueeze(2)).sum(dim=-1).clamp_min(eps)

            # Importance score for top-P parent selection (Eq. (7))
            importance = (weights * parent_counts.unsqueeze(2) / z.unsqueeze(-1)).sum(dim=2)  # [B, H, M0]
            if top_p > 0:
                top_idx = importance.topk(k=top_p, dim=-1).indices  # [B, H, P]
            else:
                top_idx = importance.new_empty((bsz, num_heads, 0), dtype=torch.long)

            # Refinement pass over selected parents (Eq. (8) and Eq. (9))
            child_offsets = torch.arange(num_children, device=q.device).view(1, 1, num_children)
            for p in range(top_idx.shape[-1]):
                parent_sel = top_idx[..., p]  # [B, H]

                # Gather the corresponding child codewords and aggregates
                child_global_idx = num_parents + parent_sel.unsqueeze(-1) * num_children + child_offsets  # [B, H, C]
                child_codes = self.codebook.child_codes[head_idx, parent_sel]  # [B, H, C, D]
                child_values = stats.aggregated_values.gather(
                    dim=2,
                    index=child_global_idx.unsqueeze(-1).expand(-1, -1, -1, dim),
                )
                child_counts = stats.aggregated_counts.gather(dim=2, index=child_global_idx)

                # Child logits: S_ic = Q_i C_{p,c}^T
                child_logits = torch.einsum("bhtd,bhcd->bhtc", q_tile, child_codes)

                # Recover parent logits from children (Eq. (8))
                parent_logits = child_logits.mean(dim=-1, keepdim=True)

                # Online-softmax merge over the running maximum
                m_new = torch.maximum(m, child_logits.max(dim=-1).values)
                scale = torch.exp(m - m_new)
                x = x * scale.unsqueeze(-1)
                z = z * scale
                m = m_new

                # Correcting attention: Delta A_ic = exp(S_ic) - exp(S_ip) (Eq. (9))
                delta_a = torch.exp(child_logits - m.unsqueeze(-1)) - torch.exp(parent_logits - m.unsqueeze(-1))
                delta_a = torch.where(
                    child_counts.unsqueeze(2) > self.codebook.empty_threshold,
                    delta_a,
                    torch.zeros_like(delta_a),
                )

                x = x + torch.einsum("bhtc,bhcd->bhtd", delta_a, child_values)
                z = z + (delta_a * child_counts.unsqueeze(2)).sum(dim=-1)

            y[:, :, start:end, :] = x / z.unsqueeze(-1).clamp_min(eps)

        return y




if __name__ == "__main__":
    result = smoke_test()
    print(result)

def smoke_test() -> dict:
    torch.manual_seed(0)

    bsz, num_heads, num_tokens, dim = 2, 2, 8, 4
    model = AVQAttention(
        num_heads=num_heads,
        dim=dim,
        num_parents=4,
        num_children=2,
        top_p=2,
        query_tile_size=4,
        ema_decay=0.9,
        init_scale=0.1,
    )
    model.train()

    q = torch.randn(bsz, num_heads, num_tokens, dim, requires_grad=True)
    k = torch.randn(bsz, num_heads, num_tokens, dim, requires_grad=True)
    v = torch.randn(bsz, num_heads, num_tokens, dim, requires_grad=True)

    out = model(q, k, v, update_codebook=True)

    # Core path test: attention output depends on quantized key/value aggregation and adaptive refinement.
    loss = out.pow(2).mean() + 0.01 * out.sum()
    loss.backward()

    return {
        "outputs": [out.detach()],
        "grad_params": [],
    }
