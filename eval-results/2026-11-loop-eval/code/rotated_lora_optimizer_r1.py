from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
import torch.nn.functional as Fnn


@dataclass
class LowRankState:
    # W = U diag(sigma) V^T
    U: Tensor
    sigma: Tensor
    V: Tensor
    # Tangent momentum components: M = U S V^T + Up V^T + U Vp^T
    S: Tensor
    Up: Tensor
    Vp: Tensor


def _random_orthonormal_columns(rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Generate a random matrix and orthonormalize its columns with QR.
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q[:, :cols]


def initialize_low_rank_state(
    weight: Tensor,
    rank: int,
    sigma_init: float = 1e-3,
) -> LowRankState:
    # Initialize W = U diag(sigma) V^T on the fixed-rank manifold.
    m, n = weight.shape
    r = min(rank, m, n)

    if weight.norm().item() < 1e-12:
        U = _random_orthonormal_columns(m, r, weight.device, weight.dtype)
        V = _random_orthonormal_columns(n, r, weight.device, weight.dtype)
        sigma = torch.full((r,), sigma_init, device=weight.device, dtype=weight.dtype)
    else:
        # SVD-based initialization: W ≈ U_r diag(s_r) V_r^T
        U_full, S_full, Vh_full = torch.linalg.svd(weight, full_matrices=False)
        U = U_full[:, :r]
        V = Vh_full[:r, :].transpose(-2, -1)
        sigma = S_full[:r].clone()

        if sigma.numel() < r:
            # TODO: Pad if needed for degenerate shapes.
            pad = r - sigma.numel()
            sigma = torch.cat(
                [sigma, torch.full((pad,), sigma_init, device=weight.device, dtype=weight.dtype)],
                dim=0,
            )

    S = torch.zeros((r, r), device=weight.device, dtype=weight.dtype)
    Up = torch.zeros((m, r), device=weight.device, dtype=weight.dtype)
    Vp = torch.zeros((n, r), device=weight.device, dtype=weight.dtype)
    return LowRankState(U=U, sigma=sigma, V=V, S=S, Up=Up, Vp=Vp)


def fixed_rank_matrix(U: Tensor, sigma: Tensor, V: Tensor) -> Tensor:
    # W = U diag(sigma) V^T
    return U @ torch.diag(sigma) @ V.transpose(-2, -1)


def tangent_components_to_matrix(U: Tensor, V: Tensor, S: Tensor, Up: Tensor, Vp: Tensor) -> Tensor:
    # Eq. (8): Z = U S V^T + Up V^T + U Vp^T
    return U @ S @ V.transpose(-2, -1) + Up @ V.transpose(-2, -1) + U @ Vp.transpose(-2, -1)


def project_gradient_to_tangent(grad: Tensor, U: Tensor, V: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    # Eq. (10)-(11):
    # Proj_W(G) = G - (I-UU^T) G (I-VV^T)
    # Tangent components:
    #   S  = U^T G V
    #   Up = G V - U S
    #   Vp = G^T U - V S^T
    S = U.transpose(-2, -1) @ grad @ V
    Up = grad @ V - U @ S
    Vp = grad.transpose(-2, -1) @ U - V @ S.transpose(-2, -1)
    return S, Up, Vp


def project_tangent_components_to_new_tangent(
    old_U: Tensor,
    old_V: Tensor,
    new_U: Tensor,
    new_V: Tensor,
    S: Tensor,
    Up: Tensor,
    Vp: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Algorithm 2, lines 1-6:
    # Project a tangent vector from T_{W_k}Mr to T_{W_{k+1}}Mr.
    CU = old_U.transpose(-2, -1) @ new_U
    CV = old_V.transpose(-2, -1) @ new_V

    # PsiV = M V_{k+1}
    PsiV = old_U @ (S @ CV + Vp.transpose(-2, -1) @ new_V) + Up @ CV
    # PsiU = M^T U_{k+1}
    PsiU = old_V @ (S.transpose(-2, -1) @ CU + Up.transpose(-2, -1) @ new_U) + Vp @ CU

    S_proj = new_U.transpose(-2, -1) @ PsiV
    Up_proj = PsiV - new_U @ S_proj
    Vp_proj = PsiU - new_V @ S_proj.transpose(-2, -1)
    return S_proj, Up_proj, Vp_proj


def multi_norm_normalize(C: Tensor, num_iters: int = 5, eps: float = 1e-8) -> Tensor:
    # Algorithm 3:
    # Alternating row/column normalization approximates
    # h(C) = argmax_H <C, H> s.t. max_i ||H_{i,:}||_2 = 1 and max_j ||H_{:,j}||_2 = 1
    H = C
    for _ in range(num_iters):
        row_norms = torch.linalg.norm(H, dim=-1, keepdim=True).clamp_min(eps)
        H = H / row_norms
        col_norms = torch.linalg.norm(H, dim=-2, keepdim=True).clamp_min(eps)
        H = H / col_norms
    return H


def build_tangent_factors(U: Tensor, S: Tensor, Up: Tensor, V: Tensor, Vp: Tensor) -> Tuple[Tensor, Tensor]:
    # Eq. (9): M = [U, Up] [[S, I], [I, 0]] [V, Vp]^T
    # Equivalent factorization used by Algorithm 1:
    #   L = [U S + Up, U], R = [V, Vp], so M = L R^T
    L = torch.cat([U @ S + Up, U], dim=-1)
    R = torch.cat([V, Vp], dim=-1)
    return L, R


def compute_conditioning_frame(
    U: Tensor,
    sigma: Tensor,
    V: Tensor,
    S: Tensor,
    Up: Tensor,
    Vp: Tensor,
    frame_dim: Optional[int] = None,
) -> Tensor:
    # Paper: conditioning frame F is formed by the left singular vectors of the tangent momentum.
    # TODO: Replace this with the QR + 2r x 2r SVD routine from the paper for memory efficiency.
    L, R = build_tangent_factors(U, S, Up, V, Vp)
    M = L @ R.transpose(-2, -1)
    U_m, _, _ = torch.linalg.svd(M, full_matrices=False)

    if frame_dim is None:
        frame_dim = min(2 * sigma.numel(), U.shape[0], U_m.shape[1])
    else:
        frame_dim = min(frame_dim, U_m.shape[1])

    F = U_m[:, :frame_dim]
    return F


def rotated_manifold_update(
    U: Tensor,
    V: Tensor,
    F: Tensor,
    S: Tensor,
    Up: Tensor,
    Vp: Tensor,
    num_iters: int = 5,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Algorithm 1:
    # 1) M = L R^T with L = [US+Up, U], R = [V, Vp]
    # 2) C = F^T M
    # 3) C_hat = h(C)
    # 4) Project F C_hat back to T_W Mr
    L, R = build_tangent_factors(U, S, Up, V, Vp)
    C = (F.transpose(-2, -1) @ L) @ R.transpose(-2, -1)
    C_hat = multi_norm_normalize(C, num_iters=num_iters, eps=eps)

    # Algorithm 1 lines 5-9
    P = F.transpose(-2, -1) @ U
    Q = C_hat @ V
    S_tilde = P.transpose(-2, -1) @ Q
    Up_tilde = F @ Q - U @ S_tilde
    Vp_tilde = C_hat.transpose(-2, -1) @ P - V @ S_tilde.transpose(-2, -1)
    return S_tilde, Up_tilde, Vp_tilde


def retract_fixed_rank(
    U: Tensor,
    sigma: Tensor,
    V: Tensor,
    S: Tensor,
    Up: Tensor,
    Vp: Tensor,
    rank: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Eq. (13)-(14):
    # W + Z = [U, Q_U] [[Sigma + S, R_V^T], [R_U, 0]] [V, Q_V]^T
    # Then take the rank-r truncated SVD of the 2r x 2r core matrix.
    r = min(rank, sigma.numel())

    Q_U, R_U = torch.linalg.qr(Up, mode="reduced")
    Q_V, R_V = torch.linalg.qr(Vp, mode="reduced")

    core_top = torch.cat([torch.diag(sigma) + S, R_V.transpose(-2, -1)], dim=-1)
    core_bottom = torch.cat(
        [R_U, torch.zeros((r, r), device=U.device, dtype=U.dtype)],
        dim=-1,
    )
    core = torch.cat([core_top, core_bottom], dim=-2)

    U_core, S_core, Vh_core = torch.linalg.svd(core, full_matrices=False)
    U_aug = torch.cat([U, Q_U], dim=-1)
    V_aug = torch.cat([V, Q_V], dim=-1)

    U_new = U_aug @ U_core[:, :r]
    V_new = V_aug @ Vh_core.transpose(-2, -1)[:, :r]
    sigma_new = S_core[:r].clone()

    W_new = fixed_rank_matrix(U_new, sigma_new, V_new)
    return U_new, sigma_new, V_new, W_new


class RoMOptimizer(Optimizer):
    # RoM: Rotated Manifold Optimization for fixed-rank LoRA adapters.
    def __init__(
        self,
        params: Iterable[Tensor],
        rank: int,
        lr: float = 1e-3,
        beta: float = 0.9,
        normalize_iters: int = 5,
        init_sigma: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta=beta,
            rank=rank,
            normalize_iters=normalize_iters,
            init_sigma=init_sigma,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
            loss = float(loss_val.item()) if isinstance(loss_val, Tensor) else float(loss_val)

        for group in self.param_groups:
            lr: float = float(group["lr"])
            beta: float = float(group["beta"])
            rank: int = int(group["rank"])
            normalize_iters: int = int(group["normalize_iters"])
            init_sigma: float = float(group["init_sigma"])
            eps: float = float(group["eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("RoMOptimizer only supports 2D matrix parameters.")

                grad = p.grad.detach()

                state = self.state[p]
                if len(state) == 0:
                    # Initialize the fixed-rank manifold state.
                    low_rank = initialize_low_rank_state(p.data, rank=rank, sigma_init=init_sigma)
                    state["U"] = low_rank.U
                    state["sigma"] = low_rank.sigma
                    state["V"] = low_rank.V
                    state["momentum_S"] = low_rank.S
                    state["momentum_Up"] = low_rank.Up
                    state["momentum_Vp"] = low_rank.Vp

                    # Sync the parameter to the initialized manifold point.
                    p.copy_(fixed_rank_matrix(state["U"], state["sigma"], state["V"]))

                U: Tensor = state["U"]
                sigma: Tensor = state["sigma"]
                V: Tensor = state["V"]
                momentum_S: Tensor = state["momentum_S"]
                momentum_Up: Tensor = state["momentum_Up"]
                momentum_Vp: Tensor = state["momentum_Vp"]

                # Eq. (11): project Euclidean gradient onto the tangent space.
                gS, gUp, gVp = project_gradient_to_tangent(grad, U, V)

                # Momentum update in the tangent space.
                # Eq. (15): M <- beta * M + (1 - beta) * grad
                M_S = beta * momentum_S + (1.0 - beta) * gS
                M_Up = beta * momentum_Up + (1.0 - beta) * gUp
                M_Vp = beta * momentum_Vp + (1.0 - beta) * gVp

                # Compute conditioning frame F and the rotated normalized direction.
                F = compute_conditioning_frame(U, sigma, V, M_S, M_Up, M_Vp)
                dS, dUp, dVp = rotated_manifold_update(
                    U=U,
                    V=V,
                    F=F,
                    S=M_S,
                    Up=M_Up,
                    Vp=M_Vp,
                    num_iters=normalize_iters,
                    eps=eps,
                )

                # Eq. (17): xi = -eta * Proj_W(F h(F^T M))
                step_S = -lr * dS
                step_Up = -lr * dUp
                step_Vp = -lr * dVp

                # Retraction: W_{k+1} = SVDr(W_k + xi)
                U_new, sigma_new, V_new, W_new = retract_fixed_rank(
                    U=U,
                    sigma=sigma,
                    V=V,
                    S=step_S,
                    Up=step_Up,
                    Vp=step_Vp,
                    rank=rank,
                )

                # Transport momentum to the new tangent space for the next iteration.
                M_S_new, M_Up_new, M_Vp_new = project_tangent_components_to_new_tangent(
                    old_U=U,
                    old_V=V,
                    new_U=U_new,
                    new_V=V_new,
                    S=M_S,
                    Up=M_Up,
                    Vp=M_Vp,
                )

                # Write back state and parameter.
                state["U"] = U_new
                state["sigma"] = sigma_new
                state["V"] = V_new
                state["momentum_S"] = M_S_new
                state["momentum_Up"] = M_Up_new
                state["momentum_Vp"] = M_Vp_new

                p.copy_(W_new)

        return loss


class RoMLoRALinear(nn.Module):
    # Simple demo module:
    # y = x (W0 + Wadp)^T + b, with Wadp optimized by RoM.
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.weight_base = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight_base, a=5 ** 0.5)

        self.weight_adapter = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # W_eff = W0 + W_adp
        weight_eff = self.weight_base + self.weight_adapter
        return Fnn.linear(x, weight_eff, self.bias)


def _demo_train_step() -> None:
    torch.manual_seed(0)

    batch_size = 8
    in_features = 16
    out_features = 12
    rank = 4

    model = RoMLoRALinear(in_features=in_features, out_features=out_features, rank=rank, bias=True)
    optimizer = RoMOptimizer([model.weight_adapter], rank=rank, lr=1e-2, beta=0.9)

    x = torch.randn(batch_size, in_features)
    target = torch.randn(batch_size, out_features)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    pred = model(x)
    loss = Fnn.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    _demo_train_step()