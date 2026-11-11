from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde

from models.base import BaseClassifier
from models.common import FocalLoss, init_weights, make_padding_mask


@dataclass(frozen=True, slots=True)
class NeuralCDEClassifierConfig:
    d_input: int = 34
    d_hidden: int = 96
    d_cde_hidden: int = 96
    n_cde_layers: int = 2
    d_classifier_ffn: int = 96
    d_out: int = 2
    dropout: float = 0.2
    max_freq_scale: float = 10.0
    use_initial_network: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    class_weights: list[float] | None = None
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    interpolation: Literal["linear", "cubic"] = "linear"
    solver: str = "rk4"
    step_size: float = 1.0
    adjoint: bool = False


class CDEVectorField(nn.Module):
    def __init__(
        self, d_hidden: int, d_path: int, d_inner: int = 96, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_path = d_path

        layers: list[nn.Module] = [nn.Linear(d_hidden, d_inner), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_inner, d_inner), nn.Tanh()]
        layers.append(nn.Linear(d_inner, d_hidden * d_path))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_shape = z.shape[:-1]
        return self.net(z).view(*batch_shape, self.d_hidden, self.d_path)


class TorchCDEEncoder(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        d_path: int,
        d_inner: int = 96,
        n_layers: int = 2,
        interpolation: Literal["linear", "cubic"] = "linear",
        solver: str = "rk4",
        step_size: float = 1.0,
        adjoint: bool = False,
    ) -> None:
        super().__init__()
        self.interpolation = interpolation
        self.solver = solver
        self.step_size = step_size
        self.adjoint = adjoint
        self.vector_field = CDEVectorField(d_hidden, d_path, d_inner, n_layers)

    def forward(
        self,
        z0: torch.Tensor,
        path: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = path.size(1)
        if seq_len == 1:
            return z0.unsqueeze(1)

        if self.interpolation == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(path)
            control = torchcde.LinearInterpolation(coeffs)
        elif self.interpolation == "cubic":
            coeffs = torchcde.natural_cubic_coeffs(path)
            control = torchcde.CubicSpline(coeffs)
        else:
            raise ValueError(f"Unsupported interpolation: {self.interpolation}")

        t_eval = torch.arange(seq_len, dtype=path.dtype, device=path.device)
        kwargs = {}
        if self.solver in {"euler", "midpoint", "rk4"}:
            kwargs["options"] = {"step_size": self.step_size}

        return torchcde.cdeint(
            X=control,
            func=self.vector_field,
            z0=z0,
            t=t_eval,
            adjoint=self.adjoint,
            method=self.solver,
            **kwargs,
        )


class InitialStateNetwork(nn.Module):
    def __init__(self, d_path: int, d_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_path, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.net(x0)


class NeuralCDEClassifier(BaseClassifier):
    def __init__(
        self, config: NeuralCDEClassifierConfig, learning_rate: float | None = 1e-3
    ) -> None:
        super().__init__(config, learning_rate)

        self.d_path = config.d_input * 2 + 1

        if config.use_initial_network:
            self.initial_net = InitialStateNetwork(
                self.d_path, config.d_hidden, config.dropout
            )
        else:
            self.initial_net = None
            self.z0 = nn.Parameter(torch.randn(config.d_hidden) * 0.01)

        self.cde_solver = TorchCDEEncoder(
            d_hidden=config.d_hidden,
            d_path=self.d_path,
            d_inner=config.d_cde_hidden,
            n_layers=config.n_cde_layers,
            interpolation=config.interpolation,
            solver=config.solver,
            step_size=config.step_size,
            adjoint=config.adjoint,
        )
        self.post_norm = nn.LayerNorm(config.d_hidden)

        self.pool_attn = nn.Sequential(
            nn.Linear(config.d_hidden, config.d_hidden // 4),
            nn.Tanh(),
            nn.Linear(config.d_hidden // 4, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(config.d_hidden, config.d_classifier_ffn),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_classifier_ffn, config.d_classifier_ffn),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_classifier_ffn, config.d_out),
        )
        self.loss_fn = FocalLoss(
            alpha=config.class_weights,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        self.apply(init_weights)
        nn.init.normal_(self.cde_solver.vector_field.net[-1].weight, std=0.01)
        nn.init.zeros_(self.cde_solver.vector_field.net[-1].bias)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        padding_mask = make_padding_mask(x, self.config.d_input)

        path = torch.cat([x, time_steps.unsqueeze(-1)], dim=-1)
        path = self._fill_padded_path(path, padding_mask)

        if self.initial_net is not None:
            z0 = self.initial_net(path[:, 0])
        else:
            z0 = self.z0.unsqueeze(0).expand(B, -1)

        hidden = self.cde_solver(z0, path)
        hidden = self.post_norm(hidden)

        attn = self.pool_attn(hidden).squeeze(-1).masked_fill(padding_mask, -1e9)
        attn = F.softmax(attn, dim=1)
        pooled = (hidden * attn.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled)

    @staticmethod
    def _fill_padded_path(
        path: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = ~padding_mask
        batch, seq_len, channels = path.shape
        positions = torch.arange(seq_len, device=path.device).expand(batch, seq_len)
        last_valid = torch.where(valid, positions, torch.zeros_like(positions))
        last_valid = torch.cummax(last_valid, dim=1).values
        gather_idx = last_valid.unsqueeze(-1).expand(batch, seq_len, channels)
        return torch.gather(path, dim=1, index=gather_idx)
