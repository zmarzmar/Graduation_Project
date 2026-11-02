from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class AVQConfig:
    num_heads: int
    head_dim: int
    num_parents: int
    num_children: int
    top_p: int
    query_block_size: int = 32
    ema_decay: float = 0.99
    child_init_std: float = 0.1
    scale: Optional[float] = None
    eps: float = 1e-6


class HierarchicalVQCodebook(nn.Module):
    # Hierarchical codebook:
    # parents C_p and children C_{p,c}, with the constraint
    # C_p = (1/C) * sum_c C_{p,c}  (Eq. 5)

    def __init__(self, config: AVQConfig) -> None:
        super().__init__()
        self.config = config

        h, m0, c, d = (
            config.num_heads,
            config.num_parents,
            config.num_children,
            config.head_dim,
        )

        self.register_buffer("parent_codebook", torch.empty(h, m0, d))
        self.register_buffer("child_codebook", torch.empty(h, m0, c, d))

        # EMA statistics for online k-means updates
        self.register_buffer("parent_ema_count", torch.ones(h, m0))
        self.register_buffer("child_ema_count", torch.ones(h, m0, c))
        self.register_buffer("parent_ema_sum", torch.zeros(h, m0, d))
        self.register_buffer("child_ema_sum", torch.zeros(h, m0, c, d))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize parents from N(0, 1)
        nn.init.normal_(self.parent_codebook, mean=0.0, std=1.0)

        # Initialize children near their parent:
        # C_{p,c} = C_p + 0.1 * eps, eps ~ N(0, I)
        noise = torch.randn_like(self.child_codebook) * self.config.child_init_std
        self.child_codebook.copy_(self.parent_codebook.unsqueeze(2) + noise)

        self.parent_ema_count.fill_(1.0)
        self.child_ema_count.fill_(1.0)
        self.parent_ema_sum.copy_(self.parent_codebook * self.parent_ema_count.unsqueeze(-1))
        self.child_ema_sum.copy_(self.child_codebook * self.child_ema_count.unsqueeze(-1))

    def _assign_single_head(
        self, keys: Tensor, head: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # keys: [N, D]
        # Returns:
        # parent_idx: [N]
        # child_rel_idx: [N] relative child index in [0, C)
        # use_child: [N] whether child is closer than parent
        # code_idx: [N] global code index in [0, M0 + M0*C)
        parent_codes = self.parent_codebook[head]  # [M0, D]
        child_codes = self.child_codebook[head]  # [M0, C, D]

        # Eq. (2): a(K_j) = argmin_a ||C_a - K_j||^2
        parent_dist = torch.cdist(keys, parent_codes, p=2).pow(2)  # [N, M0]
        parent_idx = parent_dist.argmin(dim=-1)  # [N]
        parent_best = parent_dist.gather(1, parent_idx.unsqueeze(1)).squeeze(1)  # [N]

        child_rel_idx = torch.full(
            (keys.shape[0],), -1, device=keys.device, dtype=torch.long
        )
        use_child = torch.zeros(keys.shape[0], device=keys.device, dtype=torch.bool)

        # TODO: replace per-parent Python loop with a fused kernel.
        for p in range(self.config.num_parents):
            mask = parent_idx == p
            if not mask.any():
                continue

            k_p = keys[mask]  # [Np, D]
            c_p = child_codes[p]  # [C, D]
            child_dist = torch.cdist(k_p, c_p, p=2).pow(2)  # [Np, C]
            best_child_dist, best_child_rel = child_dist.min(dim=-1)

            is_child = best_child_dist < parent_best[mask]
            use_child[mask] = is_child
            child_rel_idx[mask] = best_child_rel

        code_idx = parent_idx.clone()
        child_global = self.config.num_parents + parent_idx * self.config.num_children + child_rel_idx
        code_idx = torch.where(use_child, child_global, code_idx)

        return parent_idx, child_rel_idx, use_child, code_idx

    @torch.no_grad()
    def precompute_aggregates(
        self, keys: Tensor, values: Tensor
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        # Compute aggregated values and counts over the full hierarchical codebook.
        #
        # Algorithm 1:
        # 1) Assign each key to nearest parent.
        # 2) For each parent, compare only its C children.
        # 3) Scatter values/counts to the final codeword.
        #
        # Inputs:
        #   keys:   [B, H, N, D]
        #   values: [B, H, N, D]
        # Outputs:
        #   agg_values: [B, H, M_total, D]
        #   counts:     [B, H, M_total]
        #   aux: dict with assignment tensors for debugging/analysis

        bsz, heads, n, d = keys.shape
        total_codes = self.config.num_parents * (1 + self.config.num_children)

        agg_values = torch.zeros(
            bsz, heads, total_codes, d, device=keys.device, dtype=values.dtype
        )
        counts = torch.zeros(bsz, heads, total_codes, device=keys.device, dtype=values.dtype)

        parent_idx_all = torch.empty(bsz, heads, n, device=keys.device, dtype=torch.long)
        child_rel_all = torch.empty(bsz, heads, n, device=keys.device, dtype=torch.long)
        use_child_all = torch.empty(bsz, heads, n, device=keys.device, dtype=torch.bool)
        code_idx_all = torch.empty(bsz, heads, n, device=keys.device, dtype=torch.long)

        for b in range(bsz):
            for h in range(heads):
                parent_idx, child_rel_idx, use_child, code_idx = self._assign_single_head(
                    keys[b, h], h
                )

                parent_idx_all[b, h] = parent_idx
                child_rel_all[b, h] = child_rel_idx
                use_child_all[b, h] = use_child
                code_idx_all[b, h] = code_idx

                ones = torch.ones(n, device=keys.device, dtype=values.dtype)

                # Scatter-accumulate value sums and counts.
                agg_values[b, h].index_add_(0, code_idx, values[b, h])
                counts[b, h].index_add_(0, code_idx, ones)

        aux: Dict[str, Tensor] = {
            "parent_idx": parent_idx_all,
            "child_rel_idx": child_rel_all,
            "use_child": use_child_all,
            "code_idx": code_idx_all,
        }
        return agg_values, counts, aux

    @torch.no_grad()
    def ema_update(self, keys: Tensor) -> None:
        # Online k-means with EMA, following Appendix A.
        #
        # Parent update:
        #   S_x^{t+1} = lambda S_x^t + (1-lambda) sum_{k: a(k)=x} k
        #   N_x^{t+1} = lambda N_x^t + (1-lambda) |{k: a(k)=x}|
        #
        # Child update: same idea, but only for keys reassigned to children.
        # Then project children to satisfy:
        #   C_p = (1/C) * sum_c C_{p,c}   (Eq. 5)
        #
        # keys: [B, H, N, D]

        bsz, heads, n, d = keys.shape
        decay = self.config.ema_decay
        one_minus = 1.0 - decay
        eps = self.config.eps

        for b in range(bsz):
            for h in range(heads):
                parent_idx, child_rel_idx, use_child, _ = self._assign_single_head(keys[b, h], h)

                # Parent accumulators.
                parent_sum = torch.zeros_like(self.parent_ema_sum[h])
                parent_count = torch.zeros_like(self.parent_ema_count[h])

                parent_sum.index_add_(0, parent_idx, keys[b, h])
                parent_count.index_add_(
                    0, parent_idx, torch.ones(n, device=keys.device, dtype=keys.dtype)
                )

                self.parent_ema_sum[h].mul_(decay).add_(parent_sum, alpha=one_minus)
                self.parent_ema_count[h].mul_(decay).add_(parent_count, alpha=one_minus)

                parent_centers = self.parent_ema_sum[h] / self.parent_ema_count[h].clamp_min(eps).unsqueeze(-1)
                self.parent_codebook[h].copy_(parent_centers)

                # Child accumulators.
                child_sum = torch.zeros_like(self.child_ema_sum[h])
                child_count = torch.zeros_like(self.child_ema_count[h])

                if use_child.any():
                    child_parent = parent_idx[use_child]
                    child_rel = child_rel_idx[use_child]
                    child_global = child_parent * self.config.num_children + child_rel
                    flat_child_sum = child_sum.view(-1, d)
                    flat_child_count = child_count.view(-1)

                    flat_child_sum.index_add_(0, child_global, keys[b, h][use_child])
                    flat_child_count.index_add_(
                        0,
                        child_global,
                        torch.ones(use_child.sum(), device=keys.device, dtype=keys.dtype),
                    )

                    child_sum.mul_(1.0)  # no-op, explicit for readability

                self.child_ema_sum[h].mul_(decay).add_(child_sum, alpha=one_minus)
                self.child_ema_count[h].mul_(decay).add_(child_count, alpha=one_minus)

                # Unconstrained child means.
                child_centers = self.child_ema_sum[h] / self.child_ema_count[h].clamp_min(eps).unsqueeze(-1)

                # Projection onto the parent-child constraint surface.
                # Eq. (12): C_{p,c} = M_c - delta / (N_c * sigma)
                # where delta = sum_c M_c - C * C_p and sigma = sum_c 1/N_c
                for p in range(self.config.num_parents):
                    c_counts = self.child_ema_count[h, p].clamp_min(eps)  # [C]
                    c_centers = child_centers[p]  # [C, D]
                    p_center = self.parent_codebook[h, p]  # [D]

                    sigma = (1.0 / c_counts).sum().clamp_min(eps)
                    delta = c_centers.sum(dim=0) - self.config.num_children * p_center

                    adjusted_children = c_centers - delta.unsqueeze(0) / (c_counts.unsqueeze(-1) * sigma)
                    self.child_codebook[h, p].copy_(adjusted_children)

                    # Restore parent as mean of children.
                    self.parent_codebook[h, p].copy_(adjusted_children.mean(dim=0))

    def extra_repr(self) -> str:
        return (
            f"heads={self.config.num_heads}, "
            f"head_dim={self.config.head_dim}, "
            f"parents={self.config.num_parents}, "
            f"children={self.config.num_children}"
        )


class AVQAttention(nn.Module):
    def __init__(self, config: AVQConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = HierarchicalVQCodebook(config)
        self.scale = config.scale if config.scale is not None else 1.0 / math.sqrt(config.head_dim)

    def _parent_pass(
        self,
        q: Tensor,
        parent_codes: Tensor,
        parent_values: Tensor,
        parent_counts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Parent attention stage:
        #   A_IJ = exp(Q_I C_J^T)
        #   X_I(J) = A_IJ \bar{V}_J
        #   Z_I(J) = A_IJ n_J
        # Eq. (6)
        #
        # q: [T, D]
        # parent_codes: [M0, D]
        # parent_values: [M0, D]
        # parent_counts: [M0]
        logits = (q @ parent_codes.t()) * self.scale  # [T, M0]

        # Eq. (18): m_i = max_{j: n_j > 0} S_ij
        nonempty = parent_counts > 0
        masked_logits = logits.masked_fill(~nonempty.unsqueeze(0), float("-inf"))
        m = masked_logits.max(dim=-1).values.unsqueeze(-1)  # [T, 1]

        # Stable softmax weights.
        attn = torch.exp(logits - m) * nonempty.unsqueeze(0).to(logits.dtype)
        x = attn @ parent_values  # [T, D]
        z = (attn * parent_counts.unsqueeze(0)).sum(dim=-1, keepdim=True)  # [T, 1]

        # Eq. (7): w_j(I) = sum_i A_ij * n_j / \bar{Z}_i
        denom = z.clamp_min(self.config.eps)
        importance = ((attn * parent_counts.unsqueeze(0)) / denom).sum(dim=0)  # [M0]

        return x, z, m, importance

    def _refine_selected_parents(
        self,
        q: Tensor,
        x: Tensor,
        z: Tensor,
        m: Tensor,
        selected_parents: Tensor,
        child_codes: Tensor,
        child_values: Tensor,
        child_counts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Child refinement stage for one query tile:
        #   S_ip = (1/C) sum_c S_ic    (Eq. 8)
        #   Delta A_ic = exp(S_ic) - exp(S_ip)  (Eq. 9)
        #
        # We update the accumulators in place:
        #   x += Delta A @ \bar{V}_c
        #   z += sum_c Delta A_ic * n_c
        #
        # q: [T, D]
        # x: [T, D]
        # z: [T, 1]
        # m: [T, 1]
        #
        # selected_parents: [P]
        # child_codes: [P, C, D]
        # child_values: [P, C, D]
        # child_counts: [P, C]

        for idx in range(selected_parents.numel()):
            p = int(selected_parents[idx].item())
            c_codes = child_codes[p]  # [C, D]
            c_values = child_values[p]  # [C, D]
            c_counts = child_counts[p]  # [C]

            nonempty = c_counts > 0
            if not nonempty.any():
                continue

            sc = (q @ c_codes.t()) * self.scale  # [T, C]

            # Online softmax merge:
            # m' = max(m, max_c S_ic)
            sc_max = torch.where(
                nonempty.unsqueeze(0),
                sc,
                torch.full_like(sc, float("-inf")),
            ).max(dim=-1).values.unsqueeze(-1)
            m_new = torch.maximum(m, sc_max)

            # Rescale previous accumulators.
            rescale = torch.exp(m - m_new)
            x = x * rescale
            z = z * rescale
            m = m_new

            # Recover parent logits from children:
            # S_ip = (1/C) sum_c S_ic
            sp = sc.mean(dim=-1, keepdim=True)  # [T, 1]

            ac = torch.exp(sc - m)  # [T, C]
            ap = torch.exp(sp - m)  # [T, 1]
            delta_a = ac - ap  # [T, C]

            # Empty children contribute nothing.
            delta_a = delta_a * nonempty.unsqueeze(0).to(delta_a.dtype)

            # Eq. (17): sum_c (A_ic - A_ip) * \bar{V}_c
            x = x + delta_a @ c_values
            z = z + (delta_a * c_counts.unsqueeze(0)).sum(dim=-1, keepdim=True)

        return x, z, m

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # AVQ-Attention forward pass.
        #
        # Inputs:
        #   q, k, v: [B, H, N, D]
        #
        # Output:
        #   y: [B, H, N, D]
        #
        # Notes:
        # - This is a PyTorch reference implementation.
        # - The paper uses fused Triton kernels; the loops below are the executable skeleton.
        # - All core equations are preserved:
        #   * Eq. (2) vector quantization assignment
        #   * Eq. (6) VQ attention
        #   * Eq. (7) importance
        #   * Eq. (8)-(9) child refinement
        bsz, heads, n, d = q.shape
        assert k.shape == q.shape and v.shape == q.shape, "q, k, v must have the same shape."
        assert heads == self.config.num_heads and d == self.config.head_dim

        # Kernel 1: VQ precompute.
        agg_values, counts, _ = self.codebook.precompute_aggregates(k, v)

        total_codes = self.config.num_parents * (1 + self.config.num_children)
        parent_codes = self.codebook.parent_codebook  # [H, M0, D]
        child_codes = self.codebook.child_codebook  # [H, M0, C, D]

        y = torch.zeros_like(q)
        block = self.config.query_block_size
        m0 = self.config.num_parents
        p_sel = min(self.config.top_p, m0)

        for b in range(bsz):
            for h in range(heads):
                q_bh = q[b, h]  # [N, D]
                parent_values = agg_values[b, h, :m0]  # [M0, D]
                parent_counts = counts[b, h, :m0]  # [M0]

                for start in range(0, n, block):
                    end = min(start + block, n)
                    q_tile = q_bh[start:end]  # [T, D]

                    # Parent pass: Eq. (6)
                    x, z, m, importance = self._parent_pass(
                        q_tile, parent_codes[h], parent_values, parent_counts
                    )

                    # Top-P selection.
                    _, selected = torch.topk(importance, k=p_sel, largest=True, sorted=False)

                    # Child refinement.
                    child_values = agg_values[b, h, m0:].view(m0, self.config.num_children, d)
                    child_counts = counts[b, h, m0:].view(m0, self.config.num_children)
                    x, z, m = self._refine_selected_parents(
                        q_tile,
                        x,
                        z,
                        m,
                        selected,
                        child_codes[h],
                        child_values,
                        child_counts,
                    )

                    y[b, h, start:end] = x / z.clamp_min(self.config.eps)

        return y

    @torch.no_grad()
    def update_codebook(self, k: Tensor) -> None:
        # Call during training to update the EMA codebook.
        self.codebook.ema_update(k)

    def extra_repr(self) -> str:
        return (
            f"heads={self.config.num_heads}, "
            f"head_dim={self.config.head_dim}, "
            f"parents={self.config.num_parents}, "
            f"children={self.config.num_children}, "
            f"top_p={self.config.top_p}"
        )


def _demo() -> None:
    torch.manual_seed(0)

    config = AVQConfig(
        num_heads=2,
        head_dim=16,
        num_parents=8,
        num_children=4,
        top_p=2,
        query_block_size=4,
    )

    layer = AVQAttention(config)

    bsz, heads, n, d = 2, config.num_heads, 12, config.head_dim
    q = torch.randn(bsz, heads, n, d)
    k = torch.randn(bsz, heads, n, d)
    v = torch.randn(bsz, heads, n, d)

    # Optional EMA update step, typically done during training.
    layer.update_codebook(k)

    y = layer(q, k, v)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    _demo()