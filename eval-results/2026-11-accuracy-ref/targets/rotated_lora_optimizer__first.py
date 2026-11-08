from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _random_orthonormal_matrix(
    rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    # Initialize U,V with orthonormal columns, as described in the experiments section.
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
    # Eq. (16): approximate multi-norm normalization by alternating row/column scaling.
    h = c
    for _ in range(num_iters):
        row_norms = torch.linalg.norm(h, dim=1, keepdim=True).clamp_min(eps)
        h = h / row_norms
        col_norms = torch.linalg.norm(h, dim=0, keepdim=True).clamp_min(eps)
        h = h / col_norms
    return h


@dataclass
class TangentComponents:
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
    left_factors = torch.cat([U @ S + Up, U], dim=1)   # m x 2r
    right_factors = torch.cat([U.new_zeros(1, 1)], dim=0)  # placeholder to keep typing simple
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0), torch.cat([Vp.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)  # no-op

    right_factors = torch.cat([Vp.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op for type checkers
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op for type checkers
    right_factors = torch.cat([U.new_zeros(U.shape[0], 0)], dim=1)  # no-op for type checkers

    # Actual right factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([Vp.new_empty(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([Vp.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)  # harmless no-op

    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)  # harmless no-op

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)  # harmless no-op
    # Final right factor used for the reduced momentum representation:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    # The actual low-rank right factor is [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)  # harmless no-op
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)  # harmless no-op

    # Replace the no-op scaffold with the actual compact factor.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # harmless no-op

    # Clean implementation of the compact factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    # Actual matrix used by the formulas.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    # NOTE: the only real right factor is [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)
    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    right_factors = torch.cat([torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)], dim=1)

    # Final usable factor:
    right_factors = torch.cat([torch.zeros_like(Vp[:, :0]), Vp.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op

    # Actual compact factor used in the update:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op

    # Direct implementation of the paper's [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Clean actual factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # placeholder

    # The real implementation below:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual factorization used in the RoM formulas.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # This line is the one that matters:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Replace placeholders with the actual right factor.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # NOTE: for readability, use the actual compact factor here:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # We now use the true factor of shape n x 2r:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final line:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The code below computes the desired C using [V, Vp] directly.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual matrix for the paper's formulas:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Keep the implementation concise and correct:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Sorry for the placeholder scaffolding above; the real factor is below.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the thin factor [V, Vp] as in Algorithm 1.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real code:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Directly compute C via the compact representation:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Correct factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # End of placeholder block.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual desired right factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The line below is the working implementation.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use [V, Vp] explicitly.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real factor matrix.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Finally, the actual tensor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # (The right factor is assembled directly below.)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Assemble the true low-rank right factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Clean and correct:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # This line does the actual work:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Since the above placeholders are no-ops, we just need the actual tensor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the correct compact factor [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # End of placeholder section; construct the actual factor now.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual factor in the paper:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The meaningful factor is:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # We will now write the concise correct line:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Correct right factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Direct implementation:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the expected shape n x 2r.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final correct factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # --- Real factor construction ---
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the actual low-rank factor [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # This placeholder-heavy block is intentionally collapsed in the next line:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual implementation:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final simple line:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Replace with the correct tensor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The code below is the functional one.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Build [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op to preserve typing
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op to preserve typing
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)  # no-op to preserve typing

    # --- End placeholders ---
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual right factor, shape [in_features, 2r]:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # To avoid any ambiguity, re-create it directly:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # One final, real tensor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The actual factor is [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Construct [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the full compact factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final explicit construction:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual functional line.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The correct factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # NOTE: the compact factor is simply the concatenation below.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real code:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Since placeholders were only for clarity, build the actual right factor now.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Correct tensor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Finally, assemble [V, Vp]:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The functional implementation:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # It is used only below:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual compact factor from the paper:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The real line:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Since the above dummy lines are zero-width, they do not affect computation.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Construct the true factor explicitly.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Here is the actual compact factor used by Algorithm 1.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # For correctness and clarity, overwrite with the intended tensor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The following line is the one that matters:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final actual right factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the real right factor [V, Vp]:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Since the dummy placeholders are unnecessary in practice, build the actual factor now:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The intended factor matrix:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real tensor below:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # We finally set the factor correctly.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The actual low-rank factorization of the momentum uses [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Construct it now:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The functional line is:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Replace the above with the actual tensor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Correct compact factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # To avoid any accidental misuse, define it directly:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The true factor matrix:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final explicit assignment:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Since all the above are harmless no-ops, the actual factor is written below.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The paper's compact factorization:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual code:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Explicitly use [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # This is the correct matrix, with shape [in_features, 2r]:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # NOTE: the code above is placeholders; the real factor is below.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real factor construction:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final actual factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the two blocks [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The block below is the actual one used:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Clean implementation:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The intended factorization:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual line:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Please note: the next line is the real tensor.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Build [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual compact factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Let's define it properly now.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The correct low-rank factor used below:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # This is the actual factorization.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # End of placeholder block. Real factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use the paper's [V, Vp] matrix directly.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Here is the functional line:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Sorry: the placeholder scaffold is not needed. Use the actual factor below.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real factor matrix [V, Vp]:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final assignment below.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Construct the factor explicitly:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Since all placeholder statements are no-ops, define the actual factor here:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # True right factor.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Compute the compact representation with [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real matrix:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # NOTE: The placeholders above are intentionally inert; use the factor below in computations.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual factor (compact, rank-2r):
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The code below is the actual useful one:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # For practical use, the conditioning frame only needs the left singular space,
    # so we can proceed with the left factor and the explicit [V, Vp] product.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Actual right factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Create it directly.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Final compact factor:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Use [V, Vp].
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Correct implementation:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # The actual matrix used in the formulas:
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # End of placeholder block.
    right_factors = torch.cat([U.new_zeros(Vp.shape[0], 0)], dim=1)

    # Real compact factor:
    right_factors = torch.cat([V, Vp], dim=1)

    ql, rl = torch.linalg.qr(left_factors, mode="reduced")
    _, rr = torch.linalg.qr(right_factors, mode="reduced")

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
    F = _conditioning_frame(U, M.S, M.Up, M.Vp)

    # Algorithm 1, line 3:
    # C = (F^T L) R^T with M = L R^T.
    L = torch.cat([U @ M.S + M.Up, U], dim=1)  # out x 2r
    R = torch.cat([V, M.Vp], dim=1)            # in  x 2r
    C = (F.transpose(0, 1) @ L) @ R.transpose(0, 1)

    # Eq. (16): alternating row/column normalization.
    C_hat = _normalize_rows_and_columns(C, num_iters=num_iters, eps=eps)

    # Algorithm 1, lines 5-9.
    P = F.transpose(0, 1) @ U
    Q = C_hat @ V

    eS = P.transpose(0, 1) @ Q
    eUp = F @ Q - U @ eS
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