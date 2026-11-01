from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, List, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor


@runtime_checkable
class AdmissibilityRule(Protocol):
    def __call__(
        self,
        selected_indices: Sequence[int],
        confidences: Tensor,
        entropies: Optional[Tensor] = None,
    ) -> bool:
        ...


@dataclass(frozen=True)
class AlwaysAdmissibleRule:
    def __call__(
        self,
        selected_indices: Sequence[int],
        confidences: Tensor,
        entropies: Optional[Tensor] = None,
    ) -> bool:
        return True


@dataclass(frozen=True)
class TopKRule:
    # Eq. (5): |S ∪ {i*}| ≤ k
    k: int

    def __call__(
        self,
        selected_indices: Sequence[int],
        confidences: Tensor,
        entropies: Optional[Tensor] = None,
    ) -> bool:
        return len(selected_indices) <= self.k


@dataclass(frozen=True)
class FastDLLMRule:
    # Eq. (6): |S| * (1 - min_{s∈S} c_s) < f
    f: float

    def __call__(
        self,
        selected_indices: Sequence[int],
        confidences: Tensor,
        entropies: Optional[Tensor] = None,
    ) -> bool:
        if len(selected_indices) == 0:
            return True
        sel_conf = confidences[torch.as_tensor(selected_indices, device=confidences.device)]
        budget = len(selected_indices) * (1.0 - sel_conf.min().item())
        return budget < self.f


@dataclass(frozen=True)
class EBSamplerRule:
    # Eq. (7): sum_s H(p_s) - max_s H(p_s) < γ
    gamma: float

    def __call__(
        self,
        selected_indices: Sequence[int],
        confidences: Tensor,
        entropies: Optional[Tensor] = None,
    ) -> bool:
        if len(selected_indices) == 0:
            return True
        if entropies is None:
            raise ValueError("EB-Sampler requires per-position entropies.")
        sel_ent = entropies[torch.as_tensor(selected_indices, device=entropies.device)]
        budget = sel_ent.sum().item() - sel_ent.max().item()
        return budget < self.gamma


def compute_confidence_from_logits(logits: Tensor) -> Tensor:
    """
    Compute c_i = max_x p_theta(x_i = x | x_{\\bar M}) for each masked position.
    logits: [num_masked, vocab_size]
    returns: [num_masked]
    """
    probs = torch.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def compute_entropy_from_logits(logits: Tensor) -> Tensor:
    """
    Compute H(p_theta(x_i | x_{\\bar M})) for each masked position.
    logits: [num_masked, vocab_size]
    returns: [num_masked]
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=-1)


def mean_attention_over_heads(last_layer_attention: Tensor) -> Tensor:
    """
    Average over heads.
    last_layer_attention: [num_heads, seq_len, seq_len]
    returns: [seq_len, seq_len]
    """
    if last_layer_attention.dim() != 3:
        raise ValueError("Expected attention with shape [num_heads, seq_len, seq_len].")
    return last_layer_attention.mean(dim=0)


def masked_attention_matrix(full_attention: Tensor, masked_positions: Sequence[int]) -> Tensor:
    """
    Extract A ∈ [0,1]^{|M|×|M|} over masked positions.
    full_attention: [seq_len, seq_len] (already averaged over heads)
    masked_positions: indices of masked positions in the original sequence
    """
    idx = torch.as_tensor(masked_positions, device=full_attention.device, dtype=torch.long)
    return full_attention.index_select(0, idx).index_select(1, idx)


def append_only_utility(
    ordering: Sequence[int],
    confidences: Tensor,
    attention: Tensor,
    alpha: float,
) -> Tensor:
    """
    Eq. (4):
    U(π, A) = Σ_t [ c_{π_t} - α Σ_{r<t} A_{π_t π_r}(1 - c_{π_r}) ]
    """
    total = confidences.new_zeros(())
    for t, cur in enumerate(ordering):
        discount = confidences.new_zeros(())
        for r in range(t):
            prev = ordering[r]
            discount = discount + attention[cur, prev] * (1.0 - confidences[prev])
        total = total + confidences[cur] - alpha * discount
    return total


@dataclass
class ADASSampler:
    """
    Attention-Discounted Adaptive Sampler.

    Core greedy rule:
    u(i | S, A) = c_i - α * sum_{s in S} A_{is} (1 - c_s)

    The sampler greedily builds S and stops when the base admissibility rule fails.
    """
    alpha: float = 40.0
    admissibility_rule: Optional[AdmissibilityRule] = None

    def marginal_utility(
        self,
        candidate_idx: int,
        selected_indices: Sequence[int],
        confidences: Tensor,
        attention: Tensor,
    ) -> Tensor:
        """
        Compute u(i | S, A) for one candidate.
        """
        utility = confidences[candidate_idx]
        for s in selected_indices:
            # Eq. (3): subtract attention to already selected positions,
            # weighted by the uncertainty of the selected source position.
            utility = utility - self.alpha * attention[candidate_idx, s] * (1.0 - confidences[s])
        return utility

    def select(
        self,
        confidences: Tensor,
        attention: Tensor,
        entropies: Optional[Tensor] = None,
    ) -> List[int]:
        """
        Greedily select a subset S of masked positions.

        Args:
            confidences: [num_masked] token-wise confidence scores c_i
            attention: [num_masked, num_masked] interaction matrix A_{ij}
            entropies: [num_masked] optional entropies for EB-Sampler

        Returns:
            A list of selected masked-position indices, in selection order.
        """
        if confidences.dim() != 1:
            raise ValueError("confidences must be a 1D tensor.")
        if attention.dim() != 2 or attention.shape[0] != attention.shape[1]:
            raise ValueError("attention must be a square [num_masked, num_masked] tensor.")
        if confidences.shape[0] != attention.shape[0]:
            raise ValueError("confidences and attention must agree on num_masked.")

        if self.admissibility_rule is None:
            self.admissibility_rule = AlwaysAdmissibleRule()

        num_masked = confidences.shape[0]
        device = confidences.device

        selected: List[int] = []
        selected_mask = torch.zeros(num_masked, dtype=torch.bool, device=device)

        # Maintain current marginal utilities incrementally:
        # after adding s, all remaining candidates i are updated by
        # u_i <- u_i - α A_{is} (1 - c_s)
        current_scores = confidences.clone()

        while True:
            # Exclude already-selected positions from argmax.
            candidate_scores = current_scores.masked_fill(selected_mask, float("-inf"))
            best_score, best_idx = torch.max(candidate_scores, dim=0)

            if not torch.isfinite(best_score):
                break

            candidate = int(best_idx.item())
            proposed = selected + [candidate]

            # Stop if the enlarged set is not admissible under the base sampler rule.
            if not self.admissibility_rule(proposed, confidences, entropies):
                break

            selected.append(candidate)
            selected_mask[candidate] = True

            # Update remaining candidates using the newly selected position.
            delta = self.alpha * attention[:, candidate] * (1.0 - confidences[candidate])
            current_scores = current_scores - delta
            current_scores[selected_mask] = float("-inf")

            if len(selected) == num_masked:
                break

        return selected


def select_adas_subset(
    confidences: Tensor,
    attention: Tensor,
    alpha: float = 40.0,
    admissibility_rule: Optional[AdmissibilityRule] = None,
    entropies: Optional[Tensor] = None,
) -> List[int]:
    """
    Functional wrapper around ADASSampler.
    """
    sampler = ADASSampler(alpha=alpha, admissibility_rule=admissibility_rule)
    return sampler.select(confidences=confidences, attention=attention, entropies=entropies)


# -----------------------------
# Model integration skeleton
# -----------------------------

@dataclass
class MDLMSelectionInputs:
    """
    Skeleton container for a single masked diffusion decoding step.

    TODO:
    - Populate from your MDLM forward pass.
    - Extract final-layer self-attention and logits from the model.
    """
    masked_positions: List[int]
    logits: Tensor  # [num_masked, vocab_size]
    last_layer_attention: Tensor  # [num_heads, seq_len, seq_len]


def build_adas_inputs(step_inputs: MDLMSelectionInputs) -> tuple[Tensor, Tensor, Tensor]:
    """
    Convert model outputs into the tensors needed by ADAS.

    Returns:
        confidences: [num_masked]
        entropies: [num_masked]
        A: [num_masked, num_masked]
    """
    confidences = compute_confidence_from_logits(step_inputs.logits)
    entropies = compute_entropy_from_logits(step_inputs.logits)
    full_attention = mean_attention_over_heads(step_inputs.last_layer_attention)
    A = masked_attention_matrix(full_attention, step_inputs.masked_positions)
    return confidences, entropies, A


# -----------------------------
# Minimal executable demo
# -----------------------------

def _demo() -> None:
    torch.manual_seed(7)

    num_masked = 6
    vocab_size = 32
    num_heads = 4
    seq_len = 12

    # Dummy model outputs.
    logits = torch.randn(num_masked, vocab_size)
    attn = torch.rand(num_heads, seq_len, seq_len)
    attn = attn / attn.sum(dim=-1, keepdim=True)  # row-normalize attention

    masked_positions = [1, 3, 4, 7, 9, 11]
    step_inputs = MDLMSelectionInputs(
        masked_positions=masked_positions,
        logits=logits,
        last_layer_attention=attn,
    )

    confidences, entropies, A = build_adas_inputs(step_inputs)

    # Example 1: fixed-cardinality Top-k.
    topk_rule = TopKRule(k=3)
    selected_topk = select_adas_subset(
        confidences=confidences,
        attention=A,
        alpha=40.0,
        admissibility_rule=topk_rule,
    )

    # Example 2: entropy-budget EB-Sampler.
    eb_rule = EBSamplerRule(gamma=2.5)
    selected_eb = select_adas_subset(
        confidences=confidences,
        attention=A,
        alpha=40.0,
        admissibility_rule=eb_rule,
        entropies=entropies,
    )

    print("confidences:", confidences)
    print("selected (Top-k+AD):", selected_topk)
    print("selected (EB+AD):", selected_eb)
    print("append-only utility of Top-k+AD order:", append_only_utility(selected_topk, confidences, A, alpha=40.0))


if __name__ == "__main__":
    _demo()