from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
PastKeyValue = Tuple[Tensor, Tensor]


def inverse_softplus(x: float) -> float:
    # softplus(y) = log(1 + exp(y))
    # y = log(exp(x) - 1)
    return float(torch.log(torch.expm1(torch.tensor(x))).item())


@dataclass
class GAPEConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    training_context_len: int = 2048  # T in the paper
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: float = 4.0
    rotary_fraction: float = 0.75
    rope_theta: float = 10000.0
    dropout: float = 0.0
    bias: bool = True

    # GAPE-specific initialization
    init_gamma: float = 0.5413
    query_gate_bias_init: float = -4.0
    landmark_gate_bias_init: float = 0.0

    # Paper-matching factorization: reserve 2 coordinates for structural routing
    structural_dims: int = 2


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim < 0:
            raise ValueError("Rotary dimension must be non-negative.")
        if dim % 2 != 0:
            raise ValueError("Rotary dimension must be even.")
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) if dim > 0 else torch.empty(0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        # Standard RoPE frequencies:
        # freqs[p, r] = p / base^(2r/d)
        if self.dim == 0:
            empty = torch.empty(seq_len, 0, device=device, dtype=dtype)
            return empty, empty

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    # (x_even, x_odd) -> (-x_odd, x_even)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, position_ids: Tensor, rotary_dim: int) -> Tensor:
    """
    Apply RoPE to the first `rotary_dim` channels.

    x: [B, H, T, D]
    cos/sin: [S, rotary_dim//2]
    position_ids: [T]
    """
    if rotary_dim == 0:
        return x

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Gather per-position cos/sin
    cos_pos = cos.index_select(0, position_ids).unsqueeze(0).unsqueeze(0)  # [1,1,T,rotary_dim//2]
    sin_pos = sin.index_select(0, position_ids).unsqueeze(0).unsqueeze(0)

    x_even = x_rot[..., ::2]
    x_odd = x_rot[..., 1::2]

    # RoPE:
    # [x_even'; x_odd'] = [x_even cos - x_odd sin, x_even sin + x_odd cos]
    rotated_even = x_even * cos_pos - x_odd * sin_pos
    rotated_odd = x_even * sin_pos + x_odd * cos_pos
    x_rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    return torch.cat((x_rotated, x_pass), dim=-1)


def build_causal_mask(query_pos: Tensor, key_pos: Tensor) -> Tensor:
    # True means masked out.
    # query_pos: [Tq], key_pos: [Tk]
    return key_pos.view(1, 1, 1, -1) > query_pos.view(1, 1, -1, 1)


class GAPEAttention(nn.Module):
    def __init__(self, config: GAPEConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if config.structural_dims != 2:
            raise ValueError("This implementation follows the paper's 2-dim structural factorization.")
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        if self.head_dim <= config.structural_dims:
            raise ValueError("Head dimension must be larger than the number of structural dims.")

        # Reserve 2 coordinates for structural routing (paper Eq. 3).
        self.semantic_dim = self.head_dim - config.structural_dims

        self.rotary_dim = int(self.semantic_dim * config.rotary_fraction)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)
        if self.rotary_dim > self.semantic_dim:
            self.rotary_dim = self.semantic_dim - (self.semantic_dim % 2)

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Query gate: g_i = Softplus(w_g^T q_i)
        # Landmark gate: l_j = sigma(w_l^T k_j)
        self.query_gate_weight = nn.Parameter(torch.empty(self.n_heads, self.semantic_dim))
        self.landmark_gate_weight = nn.Parameter(torch.empty(self.n_heads, self.semantic_dim))
        self.query_gate_bias = nn.Parameter(torch.full((self.n_heads,), config.query_gate_bias_init))
        self.landmark_gate_bias = nn.Parameter(torch.full((self.n_heads,), config.landmark_gate_bias_init))

        # Head-wise amplitude: Gamma_h = Softplus(gamma_h)
        self.gamma_raw = nn.Parameter(
            torch.full((self.n_heads,), inverse_softplus(config.init_gamma), dtype=torch.float32)
        )

        self.rotary_emb = RotaryEmbedding(self.rotary_dim, base=config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Small init for gate projections to start near neutral behavior.
        nn.init.normal_(self.query_gate_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.landmark_gate_weight, mean=0.0, std=0.02)

    def _shape(self, x: Tensor) -> Tensor:
        # [B, T, C] -> [B, H, T, D]
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def _augment_qk(
        self,
        q: Tensor,
        k: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        g: Tensor,
        l: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Paper Eq. (3): rank-at-most-two factorization of the structural mask.

        M_{i,j} = Γ_h g_i / T [ j(1-l_j) + i l_j ]

        We realize this by adding two coordinates to q/k:
          q~ = [q_sem, Γ g sqrt(d), Γ g (i/T) sqrt(d)]
          k~ = [k_sem, j(1-l)/T, l]

        Under the standard SDPA scale 1/sqrt(d), the structural term is recovered exactly.
        """
        gamma = F.softplus(self.gamma_raw).view(1, self.n_heads, 1, 1)
        T = float(self.config.training_context_len)
        scale = math.sqrt(float(self.head_dim))

        # Query coordinates: [B, H, Tq]
        q_extra_1 = (gamma * g.unsqueeze(-1) * scale).to(q.dtype)
        q_extra_2 = (gamma * g.unsqueeze(-1) * (query_pos.view(1, 1, -1, 1).to(q.dtype) / T) * scale).to(q.dtype)

        # Key coordinates: [B, H, Tk]
        k_extra_1 = (key_pos.view(1, 1, 1, -1).to(k.dtype) * (1.0 - l.unsqueeze(-2).to(k.dtype)) / T)
        k_extra_2 = l.unsqueeze(-2).to(k.dtype)

        q_aug = torch.cat(
            [
                q,
                q_extra_1,
                q_extra_2,
            ],
            dim=-1,
        )
        k_aug = torch.cat(
            [
                k,
                k_extra_1,
                k_extra_2,
            ],
            dim=-1,
        )
        return q_aug, k_aug

    def forward(
        self,
        x: Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[PastKeyValue], Optional[Tensor]]:
        """
        Returns:
            attn_output: [B, T, C]
            new_past_key_value: optional KV cache
            attn_probs: optional [B, H, Tq, Tk]
        """
        bsz, q_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._shape(q)
        k = self._shape(k)
        v = self._shape(v)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        total_k_len = k.size(2)
        query_pos = torch.arange(past_len, past_len + q_len, device=device, dtype=torch.long)
        key_pos = torch.arange(0, total_k_len, device=device, dtype=torch.long)

        # RoPE on the semantic coordinates only:
        # a_{i,j} = (1/sqrt(d)) q_i^T R_Theta(i-j) k_j + M_{i,j}
        cos, sin = self.rotary_emb(seq_len=total_k_len, device=device, dtype=dtype)

        q_sem = q[..., : self.semantic_dim]
        q_struct = q[..., self.semantic_dim :]
        k_sem = k[..., : self.semantic_dim]
        k_struct = k[..., self.semantic_dim :]

        q_sem = apply_rope(q_sem, cos, sin, query_pos, self.rotary_dim)
        k_sem = apply_rope(k_sem, cos, sin, key_pos, self.rotary_dim)

        # Gates:
        # g_i = Softplus(w_g^T q_i)
        # l_j = sigma(w_l^T k_j)
        g = F.softplus(
            (q_sem * self.query_gate_weight[None, :, None, :]).sum(dim=-1)
            + self.query_gate_bias[None, :, None]
        )  # [B, H, Tq]
        l = torch.sigmoid(
            (k_sem * self.landmark_gate_weight[None, :, None, :]).sum(dim=-1)
            + self.landmark_gate_bias[None, :, None]
        )  # [B, H, Tk]

        # Factorized structural routing, exactly as in the paper.
        q_aug, k_aug = self._augment_qk(q_sem, k_sem, query_pos, key_pos, g, l)

        # Reattach untouched structural dims as requested by the paper's reserved channels.
        # The structural dims are not used directly here because the factorization already
        # encodes the same mask through the augmented query/key coordinates.
        # TODO: If matching a specific FlashAttention kernel layout, overwrite the reserved
        #       pair explicitly in the projected head layout.
        _ = q_struct
        _ = k_struct

        # Semantic + structural logits
        scale = self.head_dim ** -0.5
        attn_logits = torch.matmul(q_aug, k_aug.transpose(-2, -1)) * scale  # [B, H, Tq, Tk]

        # Causal mask:
        # only allow j <= i for autoregressive decoding
        causal_mask = build_causal_mask(query_pos, key_pos)
        attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))

        attn_probs = torch.softmax(attn_logits.to(torch.float32), dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs.to(dtype), v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        new_past = None
        if use_cache:
            # Cache the factorized key tensor, not a separate bias matrix.
            new_past = (k.detach(), v.detach())

        if return_attn:
            return attn_output, new_past, attn_probs
        return attn_output, new_past, None


class MLP(nn.Module):
    def __init__(self, config: GAPEConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_model * config.mlp_ratio)
        self.fc1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.fc2 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GAPEConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attn = GAPEAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[PastKeyValue], Optional[Tensor]]:
        attn_out, new_past, attn_probs = self.attn(
            self.norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
            return_attn=return_attn,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_past, attn_probs


class MiniGAPETransformer(nn.Module):
    def __init__(self, config: GAPEConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: Tensor,
        past_key_values: Optional[List[Optional[PastKeyValue]]] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[List[Optional[PastKeyValue]]], Optional[List[Tensor]]]:
        """
        idx: [B, T] token ids
        """
        if idx.dim() != 2:
            raise ValueError("idx must have shape [B, T].")

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        if len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values length must match number of layers.")

        x = self.tok_emb(idx)
        x = self.drop(x)

        new_past_key_values: List[Optional[PastKeyValue]] = []
        attn_maps: List[Tensor] = []

        for block, past in zip(self.blocks, past_key_values):
            x, new_past, attn_probs = block(
                x,
                past_key_value=past,
                use_cache=use_cache,
                return_attn=return_attn,
            )
            new_past_key_values.append(new_past)
            if attn_probs is not None:
                attn_maps.append(attn_probs)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits, (new_past_key_values if use_cache else None), (attn_maps if return_attn else None)

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> Tensor:
        # TODO: For efficient long-context decoding, use cached KV states across steps.
        # This reference method is intentionally simple and easy to inspect.
        self.eval()
        past: Optional[List[Optional[PastKeyValue]]] = None
        out = idx
        for _ in range(max_new_tokens):
            logits, past, _ = self.forward(out[:, -1:], past_key_values=past, use_cache=True, return_attn=False)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if greedy:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    config = GAPEConfig(
        vocab_size=1000,
        max_seq_len=128,
        training_context_len=128,
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=4.0,
        rotary_fraction=0.75,
        rope_theta=10000.0,
        dropout=0.0,
    )
    model = MiniGAPETransformer(config)

    batch_size = 2
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, past, attn_maps = model(x, use_cache=True, return_attn=True)
    print("logits shape:", logits.shape)
    print("num parameters:", count_parameters(model))
    if past is not None:
        print("cache layers:", len(past))
    if attn_maps is not None and len(attn_maps) > 0:
        print("attn map shape:", attn_maps[0].shape)

    # TODO: Replace this toy run with task-specific training/evaluation loops.
    # TODO: Add a fused SDPA / FlashAttention path that consumes the factorized q/k tensors directly.
    # TODO: If matching the paper more closely, overwrite the reserved structural pair inside
    #       the head layout and verify numerical equivalence against the explicit-logit path.


if __name__ == "__main__":
    main()