from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseColumnHistory:
    row_indices: Tensor
    values: Tensor
    k: int

    @classmethod
    def empty(
        cls,
        rows: int,
        cols: int,
        k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "SparseColumnHistory":
        # TODO: Replace dense placeholder storage with FP8 E4M3 + int32 serialization.
        k_eff = max(0, min(k, rows))
        row_indices = torch.zeros((k_eff, cols), device=device, dtype=torch.long)
        values = torch.zeros((k_eff, cols), device=device, dtype=dtype)
        return cls(row_indices=row_indices, values=values, k=k_eff)

    def to_dense(self, shape: torch.Size, dtype: torch.dtype, device: torch.device) -> Tensor:
        dense = torch.zeros(shape, device=device, dtype=dtype)
        if self.row_indices.numel() == 0:
            return dense
        dense.scatter_(
            0,
            self.row_indices.to(device=device, dtype=torch.long),
            self.values.to(device=device, dtype=dtype),
        )
        return dense


def columnwise_topk_history(x: Tensor, k: int) -> SparseColumnHistory:
    # H_k(X): retain the k largest-magnitude entries of every column.
    # Practical TACO uses this sparse history to approximate the dense EMA.
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(x.shape)}.")
    rows, cols = x.shape
    if k <= 0:
        return SparseColumnHistory.empty(rows, cols, 0, device=x.device, dtype=x.dtype)

    k_eff = min(k, rows)
    # TODO: If you need strict deterministic tie-breaking, add a stable tie rule here.
    _, row_indices = torch.topk(x.abs(), k=k_eff, dim=0, largest=True, sorted=True)
    values = x.gather(0, row_indices)
    return SparseColumnHistory(row_indices=row_indices, values=values, k=k_eff)


def columnwise_topk_sign(x: Tensor, k: int) -> Tensor:
    # Relaxed variant: keep top-k signed entries per column.
    # For exact TACO, use k=1.
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(x.shape)}.")
    rows, cols = x.shape
    if k <= 0:
        return torch.zeros_like(x)

    k_eff = min(k, rows)
    _, row_indices = torch.topk(x.abs(), k=k_eff, dim=0, largest=True, sorted=True)
    selected = x.gather(0, row_indices)
    out = torch.zeros_like(x)
    out.scatter_(0, row_indices, selected.sign())
    return out


def columnwise_top1_sign(x: Tensor) -> Tensor:
    # S(G)_{ij} = sign(G_{ij}) if i = argmax_i |G_{ij}|, and 0 otherwise.
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(x.shape)}.")
    winner_idx = torch.argmax(x.abs(), dim=0)  # Deterministic first-occurrence tie break.
    winner_val = x.gather(0, winner_idx.unsqueeze(0)).squeeze(0)
    out = torch.zeros_like(x)
    out.scatter_(0, winner_idx.unsqueeze(0), winner_val.sign().unsqueeze(0))
    return out


def default_taco_scale(param: Tensor, embedding_scale: bool = False) -> float:
    # Ordinary matrix parameters use m/n.
    # Embedding matrices can optionally use scale 1.0 as described in the paper.
    if embedding_scale:
        return 1.0
    if param.ndim != 2:
        return 1.0
    rows, cols = param.shape
    if cols == 0:
        return 1.0
    return float(rows) / float(cols)


class TACOOptimizer(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 3e-5,
        mu: float = 0.95,
        k: int = 16,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        maximize: bool = False,
        use_taco: bool = True,
        update_topk: int = 1,
        embedding_scale: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive.")
        if not (0.0 <= mu < 1.0):
            raise ValueError("mu must be in [0, 1).")
        if k < 0:
            raise ValueError("k must be non-negative.")
        if update_topk <= 0:
            raise ValueError("update_topk must be positive.")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1).")

        defaults: dict[str, Any] = dict(
            lr=lr,
            mu=mu,
            k=k,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            use_taco=use_taco,
            update_topk=update_topk,
            embedding_scale=embedding_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = float(group["lr"])
            mu: float = float(group["mu"])
            k: int = int(group["k"])
            beta1, beta2 = group["betas"]
            eps: float = float(group["eps"])
            weight_decay: float = float(group["weight_decay"])
            maximize: bool = bool(group["maximize"])
            use_taco: bool = bool(group["use_taco"])
            update_topk: int = int(group["update_topk"])
            embedding_scale: bool = bool(group["embedding_scale"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if maximize:
                    grad = -grad

                if p.ndim == 2 and use_taco:
                    self._taco_matrix_step(
                        p=p,
                        grad=grad,
                        lr=lr,
                        mu=mu,
                        k=k,
                        weight_decay=weight_decay,
                        update_topk=update_topk,
                        embedding_scale=embedding_scale,
                    )
                else:
                    self._adamw_step(
                        p=p,
                        grad=grad,
                        lr=lr,
                        beta1=float(beta1),
                        beta2=float(beta2),
                        eps=eps,
                        weight_decay=weight_decay,
                    )

        return loss

    def _taco_matrix_step(
        self,
        p: Tensor,
        grad: Tensor,
        lr: float,
        mu: float,
        k: int,
        weight_decay: float,
        update_topk: int,
        embedding_scale: bool,
    ) -> None:
        if grad.is_sparse:
            raise NotImplementedError("Sparse gradients are not supported by this skeleton.")

        state = self.state[p]

        if "history" not in state:
            state["history"] = SparseColumnHistory.empty(
                rows=p.shape[0],
                cols=p.shape[1],
                k=k,
                device=p.device,
                dtype=grad.dtype,
            )

        history: SparseColumnHistory = state["history"]
        history_dense = history.to_dense(shape=p.shape, dtype=grad.dtype, device=grad.device)

        # M_t = H_k( mu * M_{t-1} + (1 - mu) * G_t )
        ema = mu * history_dense + (1.0 - mu) * grad
        new_history = columnwise_topk_history(ema, k=k)
        state["history"] = new_history

        # D_t = (m / n) * S(M_t)
        # For the exact TACO rule, update_topk = 1.
        if update_topk == 1:
            update = columnwise_top1_sign(new_history.to_dense(shape=p.shape, dtype=grad.dtype, device=grad.device))
        else:
            update = columnwise_topk_sign(
                new_history.to_dense(shape=p.shape, dtype=grad.dtype, device=grad.device),
                k=update_topk,
            )

        scale = default_taco_scale(p, embedding_scale=embedding_scale)

        if weight_decay != 0.0:
            # Decoupled weight decay, consistent with AdamW-style regularization.
            p.add_(p, alpha=-lr * weight_decay)

        # W_{t+1} = W_t - eta_t * D_t
        p.add_(update, alpha=-lr * scale)

    def _adamw_step(
        self,
        p: Tensor,
        grad: Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        state = self.state[p]

        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        exp_avg: Tensor = state["exp_avg"]
        exp_avg_sq: Tensor = state["exp_avg_sq"]

        state["step"] += 1
        step: int = int(state["step"])

        # Standard AdamW update for scalar/vector parameters or non-TACO tensors.
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        denom = exp_avg_sq.sqrt().add_(eps)

        if weight_decay != 0.0:
            p.add_(p, alpha=-lr * weight_decay)

        p.addcdiv_(exp_avg, denom, value=-step_size)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


def main() -> None:
    torch.manual_seed(42)

    model = ToyModel()
    optimizer = TACOOptimizer(
        model.parameters(),
        lr=1e-3,
        mu=0.95,
        k=4,
        weight_decay=0.01,
        update_topk=1,  # Exact TACO update. Set >1 only for relaxed ablations.
    )

    for step in range(3):
        x = torch.randn(2, 8)
        y = torch.randn(2, 4)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        print(f"step={step} loss={float(loss):.6f}")


if __name__ == "__main__":
    main()