from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _random_orthonormal_matrix(
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    # Random orthonormal columns via thin QR initialization.
    mat = torch.randn(rows, cols, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :cols]


def _diag_from_vector(v: Tensor) -> Tensor:
    return torch.diag(v)


def _normalize_rows_and_columns(
    c: Tensor,
    num_iters: int,
    eps: float,
) -> Tensor:
    # Eq. (16): approximate multi-norm normalization by alternating row/column normalization.
    h = c
    for _ in range(num_iters):
        # Row normalization: Ci,: <- Ci,: / ||Ci,:||_2
        row_norms = torch.linalg.norm(h, dim=1, keepdim=True).clamp_min(eps)
        h = h / row_norms
        # Column normalization: C:,j <- C:,j / ||C:,j||_2
        col_norms = torch.linalg.norm(h, dim=0, keepdim=True).clamp_min(eps)
        h = h / col_norms
    return h


@dataclass
class TangentComponents:
    # Tangent vector Z = U S V^T + Up V^T + U Vp^T
    S: Tensor
    Up: Tensor
    Vp: Tensor

    def scaled(self, scale: float) -> TangentComponents:
        return TangentComponents(
            S=self.S * scale,
            Up=self.Up * scale,
            Vp=self.Vp * scale,
        )


@dataclass
class MomentumState:
    frame_u: Tensor
    frame_v: Tensor
    components: TangentComponents


class FixedRankLoRALinear(nn.Module):
    """
    Fixed-rank LoRA adapter:
        W_eff = W0 + U diag(sigma) V^T
    with U^T U = I and V^T V = I.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if 2 * rank > min(in_features, out_features):
            raise ValueError(
                "RoM implementation assumes 2r <= min(in_features, out_features) "
                "for the thin 2r-dimensional conditioning frame."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        weight0 = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight0, a=5 ** 0.5)
        self.register_buffer("weight0", weight0)

        device = weight0.device
        dtype = weight0.dtype

        u0 = _random_orthonormal_matrix(out_features, rank, device=device, dtype=dtype)
        v0 = _random_orthonormal_matrix(in_features, rank, device=device, dtype=dtype)
        sigma0 = torch.full((rank,), init_scale, device=device, dtype=dtype)

        self.U = nn.Parameter(u0)
        self.sigma = nn.Parameter(sigma0)
        self.V = nn.Parameter(v0)

        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None

        self._last_input: Optional[Tensor] = None
        self._last_grad_output: Optional[Tensor] = None

    def effective_adapter(self) -> Tensor:
        # Eq. (2): low-rank adaptation matrix.
        return self.U @ _diag_from_vector(self.sigma) @ self.V.transpose(0, 1)

    def effective_weight(self) -> Tensor:
        return self.weight0 + self.effective_adapter()

    def forward(self, x: Tensor) -> Tensor:
        self._last_input = x.detach()
        self._last_grad_output = None

        y = x @ self.effective_weight().transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias

        def _save_grad_output(grad: Tensor) -> Tensor:
            self._last_grad_output = grad.detach()
            return grad

        y.register_hook(_save_grad_output)
        return y


def _euclidean_grad_to_tangent_components(
    x: Tensor,
    grad_output: Tensor,
    U: Tensor,
    V: Tensor,
) -> TangentComponents:
    """
    Compute Eq. (11) without materializing the full matrix gradient.

    Convention:
        y = x W^T,  W in R^{out x in}
        G = dL/dW = grad_output^T x
    Then:
        S  = U^T G V
        Up = G V - U S
        Vp = G^T U - V S^T
    """
    # Eq. (12) in transposed linear-layer convention:
    #   G V  = grad_output^T (x V)
    #   G^T U = x^T (grad_output U)
    g_v = grad_output.transpose(0, 1) @ (x @ V)
    g_t_u = x.transpose(0, 1) @ (grad_output @ U)

    s = U.transpose(0, 1) @ g_v
    up = g_v - U @ s
    vp = g_t_u - V @ s.transpose(0, 1)
    return TangentComponents(S=s, Up=up, Vp=vp)


def _transport_momentum(
    old_state: MomentumState,
    u_new: Tensor,
    v_new: Tensor,
) -> TangentComponents:
    """
    Algorithm 2: factorized momentum transport on the fixed-rank manifold.
    """
    u_old = old_state.frame_u
    v_old = old_state.frame_v
    old = old_state.components

    cu = u_old.transpose(0, 1) @ u_new
    cv = v_old.transpose(0, 1) @ v_new

    # Algorithm 2, lines 2-3:
    # Psi_V = U_k (S_k C_V + Vp_k^T V_{k+1}) + Up_k C_V
    # Psi_U = V_k (S_k^T C_U + Up_k^T U_{k+1}) + Vp_k C_U
    psi_v = u_old @ (old.S @ cv + old.Vp.transpose(0, 1) @ v_new) + old.Up @ cv
    psi_u = v_old @ (old.S.transpose(0, 1) @ cu + old.Up.transpose(0, 1) @ u_new) + old.Vp @ cu

    # Algorithm 2, lines 4-6:
    s_tr = u_new.transpose(0, 1) @ psi_v
    up_tr = psi_v - u_new @ s_tr
    vp_tr = psi_u - v_new @ s_tr.transpose(0, 1)

    return TangentComponents(S=s_tr, Up=up_tr, Vp=vp_tr)


def _conditioning_frame(
    U: Tensor,
    V: Tensor,
    S: Tensor,
    Up: Tensor,
    Vp: Tensor,
) -> Tensor:
    """
    Build the 2r-dimensional conditioning frame F.

    The paper describes using the left singular vectors of the tangent momentum.
    We compute them efficiently from thin QR factors and a 2r x 2r SVD.
    """
    # M = [U, Up] [S, I; I, 0] [V, Vp]^T
    left_factors = torch.cat([U @ S + Up, U], dim=1)   # out x 2r
    right_factors = torch.cat([V, Vp], dim=1)           # in  x 2r

    ql, rl = torch.linalg.qr(left_factors, mode="reduced")
    qr, rr = torch.linalg.qr(right_factors, mode="reduced")

    small = rl @ rr.transpose(0, 1)
    u_small, _, _ = torch.linalg.svd(small, full_matrices=False)
    return ql @ u_small


def _project_and_rotate(
    U: Tensor,
    sigma: Tensor,
    V: Tensor,
    M: TangentComponents,
    num_iters: int,
    eps: float,
) -> TangentComponents:
    """
    Algorithm 1:
      C = F^T M
      C <- h(C)
      Project F h(F^T M) back to T_W M_r
    """
    # Conditioning frame F uses the left singular vectors of the tangent momentum.
    frame = _conditioning_frame(U, V, M.S, M.Up, M.Vp)

    # Algorithm 1, line 3:
    # C = (F^T L) R^T with M = L R^T.
    L = torch.cat([U @ M.S + M.Up, U], dim=1)  # out x 2r
    R = torch.cat([V, M.Vp], dim=1)            # in  x 2r
    C = (frame.transpose(0, 1) @ L) @ R.transpose(0, 1)  # 2r x in_features

    # Eq. (16): alternating row/column normalization.
    C_hat = _normalize_rows_and_columns(C, num_iters=num_iters, eps=eps)

    # Algorithm 1, lines 5-9.
    P = frame.transpose(0, 1) @ U
    Q = C_hat @ V

    eS = P.transpose(0, 1) @ Q
    eUp = frame @ Q - U @ eS
    eVp = C_hat.transpose(0, 1) @ P - V @ eS.transpose(0, 1)
    return TangentComponents(S=eS, Up=eUp, Vp=eVp)


def _retract_fixed_rank(
    U: Tensor,
    sigma: Tensor,
    V: Tensor,
    delta: TangentComponents,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Eq. (13)-(14): retract by rank-r truncated SVD using thin QR factors.
    """
    rank = sigma.shape[0]
    q_u, r_u = torch.linalg.qr(delta.Up, mode="reduced")
    q_v, r_v = torch.linalg.qr(delta.Vp, mode="reduced")

    top_left = _diag_from_vector(sigma) + delta.S
    top_right = r_v.transpose(0, 1)
    bottom_left = r_u
    bottom_right = torch.zeros(
        rank,
        rank,
        device=U.device,
        dtype=U.dtype,
    )

    small = torch.cat(
        [
            torch.cat([top_left, top_right], dim=1),
            torch.cat([bottom_left, bottom_right], dim=1),
        ],
        dim=0,
    )

    u_small, s_small, vh_small = torch.linalg.svd(small, full_matrices=False)

    left_big = torch.cat([U, q_u], dim=1)
    right_big = torch.cat([V, q_v], dim=1)

    u_new = left_big @ u_small[:, :rank]
    v_new = right_big @ vh_small.transpose(0, 1)[:, :rank]
    sigma_new = s_small[:rank].clamp_min(eps)
    return u_new, sigma_new, v_new


class RoMOptimizer:
    """
    Rotated Manifold Optimization for FixedRankLoRALinear modules.

    The optimizer maintains one momentum state per module:
      M_k in T_{W_k} M_r, represented by (S, Up, Vp).
    """

    def __init__(
        self,
        modules: Sequence[FixedRankLoRALinear],
        lr: float = 1e-2,
        beta: float = 0.9,
        norm_iters: int = 5,
        eps: float = 1e-8,
    ) -> None:
        self.modules = list(modules)
        self.lr = lr
        self.beta = beta
        self.norm_iters = norm_iters
        self.eps = eps
        self._state: Dict[int, MomentumState] = {}

    @torch.no_grad()
    def step(self) -> None:
        for module in self.modules:
            if module._last_input is None or module._last_grad_output is None:
                continue

            key = id(module)
            u_cur = module.U.detach()
            v_cur = module.V.detach()

            # Fresh Riemannian gradient from the current mini-batch.
            fresh = _euclidean_grad_to_tangent_components(
                x=module._last_input,
                grad_output=module._last_grad_output,
                U=u_cur,
                V=v_cur,
            )

            old_state = self._state.get(key)
            if old_state is None:
                # First step: momentum equals the fresh gradient.
                momentum = fresh
            else:
                # Algorithm 2: transport old momentum to the new tangent space
                # before combining it with the fresh gradient.
                transported = _transport_momentum(old_state, u_cur, v_cur)
                momentum = TangentComponents(
                    S=self.beta * transported.S + (1.0 - self.beta) * fresh.S,
                    Up=self.beta * transported.Up + (1.0 - self.beta) * fresh.Up,
                    Vp=self.beta * transported.Vp + (1.0 - self.beta) * fresh.Vp,
                )

            # Store the momentum at the current point W_k for the next step.
            self._state[key] = MomentumState(
                frame_u=u_cur.clone(),
                frame_v=v_cur.clone(),
                components=TangentComponents(
                    S=momentum.S.clone(),
                    Up=momentum.Up.clone(),
                    Vp=momentum.Vp.clone(),
                ),
            )

            # Algorithm 1: rotated normalization on the tangent momentum.
            direction = _project_and_rotate(
                U=u_cur,
                sigma=module.sigma.detach(),
                V=v_cur,
                M=momentum,
                num_iters=self.norm_iters,
                eps=self.eps,
            )

            # Final tangent update ξ = -η Proj_W(F h(F^T M)).
            delta = TangentComponents(
                S=-self.lr * direction.S,
                Up=-self.lr * direction.Up,
                Vp=-self.lr * direction.Vp,
            )

            u_new, sigma_new, v_new = _retract_fixed_rank(
                U=u_cur,
                sigma=module.sigma.detach(),
                V=v_cur,
                delta=delta,
                eps=self.eps,
            )

            module.U.copy_(u_new)
            module.sigma.copy_(sigma_new)
            module.V.copy_(v_new)

            # TODO: if extending to multi-layer transformers, clear per-layer cached
            # activations after the outer training loop to reduce peak memory.

    def zero_state(self) -> None:
        self._state.clear()




def smoke_test() -> dict:
    """
    CPU smoke test:
      1) forward/backward through a fixed-rank LoRA layer,
      2) RoM update with rotation, normalization, momentum transport, and retraction,
      3) a second step to exercise the transport path.
    """
    torch.manual_seed(0)

    layer = FixedRankLoRALinear(in_features=8, out_features=6, rank=2, bias=False)
    optimizer = RoMOptimizer([layer], lr=0.1, beta=0.9, norm_iters=3, eps=1e-8)

    x1 = torch.randn(2, 8)
    target1 = torch.randn(2, 6)

    pred1 = layer(x1)
    loss1 = F.mse_loss(pred1, target1)
    loss1.backward()
    optimizer.step()

    # Reuse the same module for a second optimization step.
    # Clear only the accumulated .grad buffers, not the RoM momentum state.
    for p in layer.parameters():
        p.grad = None

    x2 = torch.randn(2, 8)
    target2 = torch.randn(2, 6)

    pred2 = layer(x2)
    loss2 = F.mse_loss(pred2, target2)
    loss2.backward()
    optimizer.step()

    outputs = [
        loss1.detach(),
        loss2.detach(),
        layer.effective_adapter().detach(),
    ]
    grad_params = [layer.U, layer.sigma, layer.V]
    return {"outputs": outputs, "grad_params": grad_params}