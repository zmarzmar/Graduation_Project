from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


def l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def pairwise_squared_distances(x: Tensor) -> Tensor:
    # ||a_i - a_j||_2^2 = ||a_i||^2 + ||a_j||^2 - 2 a_i^T a_j
    x2 = (x * x).sum(dim=-1, keepdim=True)
    dist2 = x2 + x2.transpose(-1, -2) - 2.0 * (x @ x.transpose(-1, -2))
    return dist2.clamp_min_(0.0)


def gaussian_quantile_grid(
    n: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    # Eq. (6): gamma_k = Phi^{-1}((k + 1/2) / N)
    probs = (torch.arange(n, device=device, dtype=dtype) + 0.5) / float(n)
    normal = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device, dtype=dtype),
        scale=torch.tensor(1.0, device=device, dtype=dtype),
    )
    return normal.icdf(probs.clamp(1e-6, 1.0 - 1e-6))


def compute_alignment_loss(z1: Tensor, z2: Tensor) -> Tensor:
    # Eq. (4): maximize cosine similarity between two augmented views
    return (1.0 - F.cosine_similarity(z1, z2, dim=-1)).mean()


def representation_gradient(loss: Tensor, representations: Tensor) -> Tensor:
    # Eq. (11)–(13): representation-space gradient g_i = ∇_{z_i} R(...)
    grad = torch.autograd.grad(
        loss,
        representations,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    return grad


def extract_inputs_and_indices(
    batch: Union[Tuple[Any, ...], List[Any], Dict[str, Any]],
) -> Tuple[Tensor, Tensor]:
    # TODO: Adapt this parser to your dataset's batch format.
    if isinstance(batch, dict):
        x = batch["inputs"]
        indices = batch["indices"]
        return x, indices

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, indices = batch
            return x, indices
        if len(batch) == 3:
            x, _, indices = batch
            return x, indices
        if len(batch) > 3:
            x = batch[0]
            indices = batch[-1]
            return x, indices

    raise ValueError("Unsupported batch format. Expected dict or tuple/list.")


def scatter_by_index(values: Tensor, indices: Tensor, size: int) -> Tensor:
    out_shape = (size,) + tuple(values.shape[1:])
    out = torch.empty(out_shape, device=values.device, dtype=values.dtype)
    out[indices.long()] = values
    return out


@torch.no_grad()
def collect_dataset_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    # Refresh pass: run one deterministic view over the full dataset and collect z_i
    model.eval()
    embeddings: List[Tensor] = []
    indices_list: List[Tensor] = []

    for batch in dataloader:
        x, indices = extract_inputs_and_indices(batch)
        x = x.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True).long()
        z = model(x)
        embeddings.append(z.detach().cpu())
        indices_list.append(indices.detach().cpu())

    return torch.cat(embeddings, dim=0), torch.cat(indices_list, dim=0)


class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.net = self._build_mlp()

    def _build_mlp(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for layer_idx in range(self.num_layers - 1):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(start_dim=1))


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.net = self._build_mlp()

    def _build_mlp(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for layer_idx in range(self.num_layers - 1):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class JointEmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def encode(self, x: Tensor) -> Tensor:
        h = self.backbone(x)
        z = self.projector(h)
        return z

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)


@dataclass
class OOBCache:
    targets: Tensor
    directions: Optional[Tensor] = None
    refresh_step: int = 0


class OOBRegularizer(nn.Module):
    def __init__(self, num_samples: int) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.cache: Optional[OOBCache] = None

    @property
    def is_ready(self) -> bool:
        return self.cache is not None

    def refresh(self, dataset_embeddings: Tensor, dataset_indices: Tensor) -> None:
        raise NotImplementedError

    def online_loss(self, batch_embeddings: Tensor, batch_indices: Tensor) -> Tensor:
        raise NotImplementedError


class SigRegOOBRegularizer(OOBRegularizer):
    def __init__(
        self,
        num_samples: int,
        num_projections: int = 256,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.num_projections = num_projections

    @torch.no_grad()
    def refresh(self, dataset_embeddings: Tensor, dataset_indices: Tensor) -> None:
        # Eq. (6): construct projected Gaussian anchors by global rank transport
        z_ordered = scatter_by_index(dataset_embeddings, dataset_indices, self.num_samples)
        device = z_ordered.device
        dtype = z_ordered.dtype
        embedding_dim = z_ordered.shape[-1]

        directions = l2_normalize(
            torch.randn(self.num_projections, embedding_dim, device=device, dtype=dtype),
            dim=-1,
        )
        projections = z_ordered @ directions.transpose(0, 1)  # [N, M]
        gamma = gaussian_quantile_grid(self.num_samples, device=device, dtype=dtype)

        targets = torch.empty_like(projections)
        for m in range(self.num_projections):
            order = torch.argsort(projections[:, m], dim=0)
            ranks = torch.empty(self.num_samples, device=device, dtype=torch.long)
            ranks[order] = torch.arange(self.num_samples, device=device)
            targets[:, m] = gamma[ranks]

        self.cache = OOBCache(
            targets=targets.detach().cpu(),
            directions=directions.detach().cpu(),
            refresh_step=0 if self.cache is None else self.cache.refresh_step + 1,
        )

    def online_loss(self, batch_embeddings: Tensor, batch_indices: Tensor) -> Tensor:
        if self.cache is None or self.cache.directions is None:
            raise RuntimeError("SIGReg-OOB cache is empty. Call refresh() first.")

        # Eq. (7): L_ac = (1 / BVM) sum_{i,v,m} (v_m^T z_i^(v) - q_im)^2
        if batch_embeddings.ndim == 2:
            batch_embeddings = batch_embeddings.unsqueeze(1)  # [B, 1, D]

        directions = self.cache.directions.to(batch_embeddings.device, non_blocking=True)
        targets = self.cache.targets[batch_indices.long().cpu()].to(
            batch_embeddings.device, non_blocking=True
        )  # [B, M]

        projections = torch.einsum("bvd,md->bvm", batch_embeddings, directions)
        expanded_targets = targets.unsqueeze(1).expand_as(projections)
        return F.mse_loss(projections, expanded_targets)


class ContrastiveOOBRegularizer(OOBRegularizer):
    def __init__(
        self,
        num_samples: int,
        anchor_dim: int,
        num_anchor_updates: int = 20,
        anchor_subset_size: int = 256,
        tau_u: float = 2.0,
        lambda_f: float = 0.1,
        anchor_lr: float = 0.05,
        previous_cache_mixture: float = 0.5,
        init_noise_std: float = 1e-3,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.anchor_dim = anchor_dim
        self.num_anchor_updates = num_anchor_updates
        self.anchor_subset_size = anchor_subset_size
        self.tau_u = tau_u
        self.lambda_f = lambda_f
        self.anchor_lr = anchor_lr
        self.previous_cache_mixture = previous_cache_mixture
        self.init_noise_std = init_noise_std

    @torch.no_grad()
    def refresh(self, dataset_embeddings: Tensor, dataset_indices: Tensor) -> None:
        # Eq. (8)–(9): optimize spherical anchors with a population-level spread term plus correspondence term
        z_ordered = scatter_by_index(dataset_embeddings, dataset_indices, self.num_samples)
        z_ordered = l2_normalize(z_ordered, dim=-1)
        device = z_ordered.device
        dtype = z_ordered.dtype

        if self.cache is not None and self.cache.targets.shape == z_ordered.shape:
            prev = self.cache.targets.to(device, non_blocking=True)
            init_anchors = self.previous_cache_mixture * prev + (1.0 - self.previous_cache_mixture) * z_ordered
        else:
            init_anchors = z_ordered + self.init_noise_std * torch.randn_like(z_ordered)

        anchor_param = nn.Parameter(l2_normalize(init_anchors, dim=-1))
        optimizer = torch.optim.Adam([anchor_param], lr=self.anchor_lr)

        for _ in range(self.num_anchor_updates):
            subset_size = min(self.anchor_subset_size, self.num_samples)
            subset = torch.randperm(self.num_samples, device=device)[:subset_size]
            anchors_sub = l2_normalize(anchor_param[subset], dim=-1)
            data_sub = z_ordered[subset]

            # Eq. (9): preserve anchor-sample correspondence
            fit_loss = self.lambda_f * (1.0 - (anchors_sub * data_sub).sum(dim=-1)).mean()

            if subset_size > 1:
                # Eq. (8): U(A) = log[mean_{i != j} exp(-tau_u ||a_i - a_j||^2)]
                dist2 = pairwise_squared_distances(anchors_sub)
                mask = ~torch.eye(subset_size, device=device, dtype=torch.bool)
                pairwise_term = torch.exp(-self.tau_u * dist2[mask]).mean().clamp_min(1e-12)
                spread_loss = torch.log(pairwise_term)
            else:
                spread_loss = torch.tensor(0.0, device=device, dtype=dtype)

            loss = spread_loss + fit_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                anchor_param.copy_(l2_normalize(anchor_param, dim=-1))

        self.cache = OOBCache(
            targets=l2_normalize(anchor_param.detach(), dim=-1).cpu(),
            directions=None,
            refresh_step=0 if self.cache is None else self.cache.refresh_step + 1,
        )

    def online_loss(self, batch_embeddings: Tensor, batch_indices: Tensor) -> Tensor:
        if self.cache is None:
            raise RuntimeError("CONTRASTIVE-OOB cache is empty. Call refresh() first.")

        # Eq. (10): L_ac = (1 / BV) sum_{i,v} (1 - u_i^(v)^T a_i)
        if batch_embeddings.ndim == 2:
            batch_embeddings = batch_embeddings.unsqueeze(1)  # [B, 1, D]

        anchors = self.cache.targets[batch_indices.long().cpu()].to(
            batch_embeddings.device, non_blocking=True
        )  # [B, D]
        anchors = anchors.unsqueeze(1)  # [B, 1, D]
        u = l2_normalize(batch_embeddings, dim=-1)
        return (1.0 - (u * anchors).sum(dim=-1)).mean()


class OOBSSLTrainer:
    def __init__(
        self,
        model: JointEmbeddingModel,
        regularizer: OOBRegularizer,
        optimizer: torch.optim.Optimizer,
        lambda_align: float = 1.0,
        lambda_ac: float = 1.0,
    ) -> None:
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.lambda_align = lambda_align
        self.lambda_ac = lambda_ac

    def training_step(
        self,
        views: Sequence[Tensor],
        indices: Tensor,
    ) -> Dict[str, Tensor]:
        if len(views) < 2:
            raise ValueError("Expected at least two augmented views for alignment loss.")

        z_views = [self.model(view) for view in views]

        # Eq. (4): L_OOB = lambda_align * L_align + lambda_ac * L_ac
        align_loss = compute_alignment_loss(z_views[0], z_views[1])
        stacked = torch.stack(z_views, dim=1)  # [B, V, D]
        ac_loss = self.regularizer.online_loss(stacked, indices)

        total_loss = self.lambda_align * align_loss + self.lambda_ac * ac_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.detach(),
            "align_loss": align_loss.detach(),
            "ac_loss": ac_loss.detach(),
        }

    @torch.no_grad()
    def refresh_regularizer(
        self,
        refresh_loader: DataLoader,
        device: torch.device,
    ) -> None:
        # Refresh phase: encode the whole dataset once, build global anchors, keep them detached in CPU memory
        embeddings, indices = collect_dataset_embeddings(self.model, refresh_loader, device=device)
        self.regularizer.refresh(embeddings, indices)


def make_dummy_dataloader(
    num_samples: int = 128,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    batch_size: int = 16,
) -> DataLoader:
    x = torch.randn(num_samples, channels, height, width)
    indices = torch.arange(num_samples, dtype=torch.long)
    dataset = TensorDataset(x, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = 128
    channels, height, width = 3, 32, 32
    input_dim = channels * height * width

    backbone = MLPBackbone(input_dim=input_dim, hidden_dim=512, output_dim=256, num_layers=2)
    projector = ProjectionHead(input_dim=256, hidden_dim=256, output_dim=128, num_layers=2, use_layer_norm=False)
    model = JointEmbeddingModel(backbone=backbone, projector=projector).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Demo with SIGReg-OOB
    sigreg_regularizer = SigRegOOBRegularizer(num_samples=num_samples, num_projections=32)
    sigreg_trainer = OOBSSLTrainer(
        model=model,
        regularizer=sigreg_regularizer,
        optimizer=optimizer,
        lambda_align=5.0,
        lambda_ac=1.0,
    )

    refresh_loader = make_dummy_dataloader(
        num_samples=num_samples,
        channels=channels,
        height=height,
        width=width,
        batch_size=16,
    )

    sigreg_trainer.refresh_regularizer(refresh_loader, device=device)

    # Simulate one online update with two augmented views
    batch = next(iter(refresh_loader))
    x, indices = extract_inputs_and_indices(batch)
    x = x.to(device)
    indices = indices.to(device)

    x1 = x + 0.01 * torch.randn_like(x)
    x2 = x + 0.01 * torch.randn_like(x)

    sigreg_losses = sigreg_trainer.training_step([x1, x2], indices)
    print("SIGReg-OOB:", {k: float(v) for k, v in sigreg_losses.items()})

    # Demo with CONTRASTIVE-OOB
    contrastive_regularizer = ContrastiveOOBRegularizer(
        num_samples=num_samples,
        anchor_dim=128,
        num_anchor_updates=5,
        anchor_subset_size=32,
        tau_u=2.0,
        lambda_f=0.1,
        anchor_lr=0.05,
        previous_cache_mixture=0.5,
    )
    contrastive_trainer = OOBSSLTrainer(
        model=model,
        regularizer=contrastive_regularizer,
        optimizer=optimizer,
        lambda_align=1.0,
        lambda_ac=1.0,
    )

    contrastive_trainer.refresh_regularizer(refresh_loader, device=device)
    contrastive_losses = contrastive_trainer.training_step([x1, x2], indices)
    print("CONTRASTIVE-OOB:", {k: float(v) for k, v in contrastive_losses.items()})


if __name__ == "__main__":
    main()