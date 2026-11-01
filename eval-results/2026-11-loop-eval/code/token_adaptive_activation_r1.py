from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ActivationName = Literal["id", "relu", "relu2", "leaky_relu", "gelu", "silu", "tanh"]
GateKind = Literal["sigmoid", "tanh", "softmax"]
TypeIIVariant = Literal["one_sided", "bi_sided", "quadratic"]


def _pairwise_indices(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i, n):
            pairs.append((i, j))
    return pairs


def _reshape_coeffs(coeffs: Tensor, target: Tensor) -> Tensor:
    # target shape: [..., H, K] or [..., H, P]
    while coeffs.dim() < target.dim() - 1:
        coeffs = coeffs.unsqueeze(1)
    return coeffs.unsqueeze(-2)


def apply_activation(x: Tensor, name: ActivationName, negative_slope: float = 0.01) -> Tensor:
    if name == "id":
        return x
    if name == "relu":
        return F.relu(x)
    if name == "relu2":
        return F.relu(x).pow(2)
    if name == "leaky_relu":
        return F.leaky_relu(x, negative_slope=negative_slope)
    if name == "gelu":
        return F.gelu(x)
    if name == "silu":
        return F.silu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise ValueError(f"Unsupported activation: {name}")


class ActivationDictionary(nn.Module):
    def __init__(
        self,
        activation_names: Sequence[ActivationName],
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.activation_names: tuple[ActivationName, ...] = tuple(activation_names)
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        # Returns: [..., H, K]
        outputs: list[Tensor] = []
        for name in self.activation_names:
            outputs.append(apply_activation(x, name, negative_slope=self.negative_slope))
        return torch.stack(outputs, dim=-1)


class LightweightGate(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_choices: int,
        gate_kind: GateKind = "sigmoid",
        gate_init_std: float = 0.022,
        normalize_non_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, num_choices, bias=True)
        self.gate_kind = gate_kind
        self.normalize_non_softmax = normalize_non_softmax
        self.gate_init_std = gate_init_std
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.gate_init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.proj(x)  # [*, K]
        if self.gate_kind == "softmax":
            weights = torch.softmax(logits, dim=-1)
        elif self.gate_kind == "sigmoid":
            weights = torch.sigmoid(logits)
        elif self.gate_kind == "tanh":
            weights = torch.tanh(logits)
        else:
            raise ValueError(f"Unsupported gate kind: {self.gate_kind}")

        # TODO: optionally normalize non-softmax gates for stable scale matching.
        if self.normalize_non_softmax and self.gate_kind != "softmax":
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
        return weights


def _default_type1_activations() -> tuple[ActivationName, ...]:
    # Paper: {ReLU2, GELU, SiLU, LeakyReLU, ReLU, tanh}
    return ("relu2", "gelu", "silu", "leaky_relu", "relu", "tanh")


def _default_type2_activations() -> tuple[ActivationName, ...]:
    # Paper: {Id, GELU, SiLU, ReLU, LeakyReLU, ReLU2, tanh}
    return ("id", "gelu", "silu", "relu", "leaky_relu", "relu2", "tanh")


class TypeIFFNBaseline(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        activation: ActivationName = "relu2",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_hidden, d_model, bias=bias)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # f(x) = W2 sigma(W1 x)
        h = self.w1(x)
        h = apply_activation(h, self.activation)
        return self.w2(h)


class TypeIFFNLA(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        activation_names: Sequence[ActivationName] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_hidden, d_model, bias=bias)
        self.activation_names = tuple(activation_names or _default_type1_activations())
        self.activations = ActivationDictionary(self.activation_names)
        self.alpha = nn.Parameter(torch.ones(len(self.activation_names)))

    def forward(self, x: Tensor) -> Tensor:
        # f_LA(x) = W2 sum_k alpha_k sigma_k(W1 x)
        h = self.w1(x)  # [*, H]
        bank = self.activations(h)  # [*, H, K]
        coeffs = self.alpha.view(*([1] * (bank.dim() - 2)), -1)  # [*, 1, K]
        mixed = (bank * coeffs).sum(dim=-1)  # [*, H]
        return self.w2(mixed)


class TypeIFFNMoA(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        activation_names: Sequence[ActivationName] | None = None,
        gate_kind: GateKind = "sigmoid",
        bias: bool = True,
        gate_init_std: float = 0.022,
        normalize_non_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_hidden, d_model, bias=bias)
        self.activation_names = tuple(activation_names or _default_type1_activations())
        self.activations = ActivationDictionary(self.activation_names)
        self.gate = LightweightGate(
            in_features=d_model,
            num_choices=len(self.activation_names),
            gate_kind=gate_kind,
            gate_init_std=gate_init_std,
            normalize_non_softmax=normalize_non_softmax,
        )

    def forward(self, x: Tensor) -> Tensor:
        # f_MoA(x) = W2 sum_k pi_k(x) sigma_k(W1 x)
        h = self.w1(x)  # [*, H]
        bank = self.activations(h)  # [*, H, K]
        gates = self.gate(x)  # [*, K]
        mixed = (bank * gates.unsqueeze(-2)).sum(dim=-1)  # [*, H]
        return self.w2(mixed)


class TypeIIFeedForwardBaseline(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        activation: ActivationName = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w3 = nn.Linear(d_hidden, d_model, bias=bias)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        # g(x) = W3( sigma(W1 x) ⊙ (W2 x) )
        y = apply_activation(self.w1(x), self.activation)
        z = self.w2(x)
        return self.w3(y * z)


class TypeIIFeedForwardLA(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        variant: TypeIIVariant = "bi_sided",
        activation_names: Sequence[ActivationName] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w3 = nn.Linear(d_hidden, d_model, bias=bias)
        self.variant = variant
        self.activation_names = tuple(activation_names or _default_type2_activations())
        self.activations = ActivationDictionary(self.activation_names)
        self.pair_indices = _pairwise_indices(len(self.activation_names))

        if self.variant == "one_sided":
            self.alpha_z = nn.Parameter(torch.ones(len(self.activation_names)))
        elif self.variant == "bi_sided":
            self.alpha_y = nn.Parameter(torch.ones(len(self.activation_names)))
            self.alpha_z = nn.Parameter(torch.ones(len(self.activation_names)))
        elif self.variant == "quadratic":
            self.alpha_pairs = nn.Parameter(torch.ones(len(self.pair_indices)))
        else:
            raise ValueError(f"Unsupported Type-II LA variant: {self.variant}")

    def forward(self, x: Tensor) -> Tensor:
        # one-sided LA:
        # g_one-LA(x) = W3( SiLU(W1x) ⊙ sum_k alpha_k sigma_k(W2x) )
        # bi-sided LA:
        # g_bi-LA(x) = W3( sum_k beta_k sigma_k(W1x) ⊙ sum_l alpha_l sigma_l(W2x) )
        # quadratic LA:
        # g_qd-LA(x) = W3( sum_{p<=q} alpha_{p,q} sigma_p(W1x) ⊙ sigma_q(W2x) )
        y_pre = self.w1(x)
        z_pre = self.w2(x)

        if self.variant == "one_sided":
            y = F.silu(y_pre)
            z_bank = self.activations(z_pre)  # [*, H, K]
            z_coeffs = self.alpha_z.view(*([1] * (z_bank.dim() - 2)), -1)
            z = (z_bank * z_coeffs).sum(dim=-1)
            return self.w3(y * z)

        if self.variant == "bi_sided":
            y_bank = self.activations(y_pre)
            z_bank = self.activations(z_pre)
            y_coeffs = self.alpha_y.view(*([1] * (y_bank.dim() - 2)), -1)
            z_coeffs = self.alpha_z.view(*([1] * (z_bank.dim() - 2)), -1)
            y = (y_bank * y_coeffs).sum(dim=-1)
            z = (z_bank * z_coeffs).sum(dim=-1)
            return self.w3(y * z)

        # quadratic
        pair_terms: list[Tensor] = []
        for p, q in self.pair_indices:
            term = apply_activation(y_pre, self.activation_names[p]) * apply_activation(
                z_pre, self.activation_names[q]
            )
            pair_terms.append(term)
        pair_bank = torch.stack(pair_terms, dim=-1)  # [*, H, P]
        pair_coeffs = self.alpha_pairs.view(*([1] * (pair_bank.dim() - 2)), -1)
        mixed = (pair_bank * pair_coeffs).sum(dim=-1)
        return self.w3(mixed)


class TypeIIFeedForwardMoA(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        variant: TypeIIVariant = "bi_sided",
        activation_names: Sequence[ActivationName] | None = None,
        gate_kind: GateKind = "sigmoid",
        bias: bool = True,
        gate_init_std: float = 0.022,
        normalize_non_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_model, d_hidden, bias=bias)
        self.w3 = nn.Linear(d_hidden, d_model, bias=bias)
        self.variant = variant
        self.activation_names = tuple(activation_names or _default_type2_activations())
        self.activations = ActivationDictionary(self.activation_names)
        self.pair_indices = _pairwise_indices(len(self.activation_names))
        self.gate_kind = gate_kind

        if self.variant == "one_sided":
            self.gate_z = LightweightGate(
                in_features=d_model,
                num_choices=len(self.activation_names),
                gate_kind=gate_kind,
                gate_init_std=gate_init_std,
                normalize_non_softmax=normalize_non_softmax,
            )
        elif self.variant == "bi_sided":
            self.gate_y = LightweightGate(
                in_features=d_model,
                num_choices=len(self.activation_names),
                gate_kind=gate_kind,
                gate_init_std=gate_init_std,
                normalize_non_softmax=normalize_non_softmax,
            )
            self.gate_z = LightweightGate(
                in_features=d_model,
                num_choices=len(self.activation_names),
                gate_kind=gate_kind,
                gate_init_std=gate_init_std,
                normalize_non_softmax=normalize_non_softmax,
            )
        elif self.variant == "quadratic":
            self.gate_pairs = LightweightGate(
                in_features=d_model,
                num_choices=len(self.pair_indices),
                gate_kind=gate_kind,
                gate_init_std=gate_init_std,
                normalize_non_softmax=normalize_non_softmax,
            )
        else:
            raise ValueError(f"Unsupported Type-II MoA variant: {self.variant}")

    def forward(self, x: Tensor) -> Tensor:
        # one-sided MoA:
        # g_one-MoA(x) = W3( SiLU(W1x) ⊙ sum_k pi_k(x) sigma_k(W2x) )
        # bi-sided MoA:
        # g_bi-MoA(x) = W3( sum_k rho_k(x) sigma_k(W1x) ⊙ sum_l pi_l(x) sigma_l(W2x) )
        # quadratic MoA:
        # g_qd-MoA(x) = W3( sum_{p<=q} pi_{p,q}(x) sigma_p(W1x) ⊙ sigma_q(W2x) )
        y_pre = self.w1(x)
        z_pre = self.w2(x)

        if self.variant == "one_sided":
            y = F.silu(y_pre)
            z_bank = self.activations(z_pre)  # [*, H, K]
            z_gates = self.gate_z(x)  # [*, K]
            z = (z_bank * z_gates.unsqueeze(-2)).sum(dim=-1)
            return self.w3(y * z)

        if self.variant == "bi_sided":
            y_bank = self.activations(y_pre)
            z_bank = self.activations(z_pre)
            y_gates = self.gate_y(x)
            z_gates = self.gate_z(x)
            y = (y_bank * y_gates.unsqueeze(-2)).sum(dim=-1)
            z = (z_bank * z_gates.unsqueeze(-2)).sum(dim=-1)
            return self.w3(y * z)

        pair_terms: list[Tensor] = []
        for p, q in self.pair_indices:
            term = apply_activation(y_pre, self.activation_names[p]) * apply_activation(
                z_pre, self.activation_names[q]
            )
            pair_terms.append(term)
        pair_bank = torch.stack(pair_terms, dim=-1)  # [*, H, P]
        pair_gates = self.gate_pairs(x)  # [*, P]
        mixed = (pair_bank * pair_gates.unsqueeze(-2)).sum(dim=-1)
        return self.w3(mixed)


@dataclass
class MoAConfig:
    d_model: int
    d_hidden: int
    ffn_kind: Literal["type1", "type2"] = "type2"
    variant: Literal["baseline", "la", "moa"] = "moa"
    type2_variant: TypeIIVariant = "bi_sided"
    gate_kind: GateKind = "sigmoid"
    activation_names: tuple[ActivationName, ...] | None = None
    activation: ActivationName = "silu"
    bias: bool = True
    gate_init_std: float = 0.022
    normalize_non_softmax: bool = False


def build_ffn(config: MoAConfig) -> nn.Module:
    # TODO: integrate this into a full Transformer block.
    if config.ffn_kind == "type1":
        if config.variant == "baseline":
            return TypeIFFNBaseline(
                d_model=config.d_model,
                d_hidden=config.d_hidden,
                activation=config.activation,
                bias=config.bias,
            )
        if config.variant == "la":
            return TypeIFFNLA(
                d_model=config.d_model,
                d_hidden=config.d_hidden,
                activation_names=config.activation_names,
                bias=config.bias,
            )
        if config.variant == "moa":
            return TypeIFFNMoA(
                d_model=config.d_model,
                d_hidden=config.d_hidden,
                activation_names=config.activation_names,
                gate_kind=config.gate_kind,
                bias=config.bias,
                gate_init_std=config.gate_init_std,
                normalize_non_softmax=config.normalize_non_softmax,
            )
        raise ValueError(f"Unsupported variant: {config.variant}")

    if config.ffn_kind == "type2":
        if config.variant == "baseline":
            return TypeIIFeedForwardBaseline(
                d_model=config.d_model,
                d_hidden=config.d_hidden,
                activation=config.activation,
                bias=config.bias,
            )
        if config.variant == "la":
            return TypeIIFeedForwardLA(
                d_model=config.d_model,
                d_hidden=config.d_hidden,
                variant=config.type2_variant,
                activation_names=config.activation_names,
                bias=config.bias,
            )
        if config.variant == "moa":
            return TypeIIFeedForwardMoA(
                d_model=config.d_model,
                d_hidden=config.d_hidden,
                variant=config.type2_variant,
                activation_names=config.activation_names,
                gate_kind=config.gate_kind,
                bias=config.bias,
                gate_init_std=config.gate_init_std,
                normalize_non_softmax=config.normalize_non_softmax,
            )
        raise ValueError(f"Unsupported variant: {config.variant}")

    raise ValueError(f"Unsupported ffn_kind: {config.ffn_kind}")


class ResidualFFNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn: nn.Module,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm = norm if norm is not None else nn.RMSNorm(d_model)
        self.ffn = ffn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.norm(x))


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 4, 768)

    type1_baseline = TypeIFFNBaseline(d_model=768, d_hidden=3072, activation="relu2")
    type1_la = TypeIFFNLA(d_model=768, d_hidden=3072)
    type1_moa = TypeIFFNMoA(d_model=768, d_hidden=3072, gate_kind="sigmoid")

    type2_baseline = TypeIIFeedForwardBaseline(d_model=768, d_hidden=2048, activation="silu")
    type2_la = TypeIIFeedForwardLA(d_model=768, d_hidden=2048, variant="bi_sided")
    type2_moa = TypeIIFeedForwardMoA(d_model=768, d_hidden=2048, variant="bi_sided", gate_kind="sigmoid")

    y1 = type1_baseline(x)
    y2 = type1_la(x)
    y3 = type1_moa(x)

    z1 = type2_baseline(x)
    z2 = type2_la(x)
    z3 = type2_moa(x)

    print("Type-I baseline:", y1.shape)
    print("Type-I LA:", y2.shape)
    print("Type-I MoA:", y3.shape)
    print("Type-II baseline:", z1.shape)
    print("Type-II LA:", z2.shape)
    print("Type-II MoA:", z3.shape)