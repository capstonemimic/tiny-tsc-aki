from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseClassifier
from models.common import (
    ContinuousTimeEncoding,
    FocalLoss,
    init_weights,
    make_padding_mask,
)


@dataclass(frozen=True, slots=True)
class CTMambaClassifierConfig:
    d_input: int = 34
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    n_layers: int = 4
    d_classifier_ffn: int = 128
    d_out: int = 2
    dropout: float = 0.2
    max_freq_scale: float = 10.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    class_weights: list[float] | None = None
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    dt_min: float = 1e-3
    dt_max: float = 1e-2
    dt_init_std: float = 0.1

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand_factor

    @property
    def dt_rank(self) -> int:
        return math.ceil(self.d_model / 16)


class ContinuousTimeSelectiveSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int = 8,
        dt_min: float = 1e-3,
        dt_max: float = 1e-2,
        dt_init_std: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.time_proj = nn.Linear(1, d_inner, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).contiguous()
        )
        self.D = nn.Parameter(torch.ones(d_inner))

        self._init_dt(dt_min, dt_max, dt_init_std)
        self.dropout = nn.Dropout(dropout)

    def _init_dt(self, dt_min: float, dt_max: float, dt_init_std: float) -> None:
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(
        self,
        x: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_proj = self.x_proj(x_conv)
        delta_raw = x_proj[:, :, : self.dt_proj.in_features]
        b_input = x_proj[
            :, :, self.dt_proj.in_features : self.dt_proj.in_features + self.d_state
        ]
        c_input = x_proj[:, :, self.dt_proj.in_features + self.d_state :]

        tg = time_gaps.unsqueeze(-1)
        delta = F.softplus(self.dt_proj(delta_raw) + self.time_proj(tg))

        A = -torch.exp(self.A_log)
        states = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            # Computing the discretized A/B terms for the entire sequence would
            # allocate two [B, T, D, N] tensors per layer (hundreds of MB for
            # the production config) and retain them for backpropagation. The
            # recurrence only needs the current timestep, so form those terms
            # lazily. This is algebraically identical and cuts peak memory and
            # memory traffic substantially.
            delta_t = delta[:, t].unsqueeze(-1)
            a_bar_t = torch.exp(delta_t * A.unsqueeze(0))
            b_bar_t = delta_t * b_input[:, t].unsqueeze(1)
            states = a_bar_t * states + b_bar_t * x_conv[:, t].unsqueeze(-1)
            y_t = (states * c_input[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)

        y = y + self.D * x_conv
        return self.dropout(y)


class CTMambaBlock(nn.Module):
    def __init__(self, config: CTMambaClassifierConfig) -> None:
        super().__init__()
        d_inner = config.d_inner
        self.norm = nn.LayerNorm(config.d_model)
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)
        self.ssm = ContinuousTimeSelectiveSSM(
            d_model=config.d_model,
            d_inner=d_inner,
            d_state=config.d_state,
            d_conv=config.d_conv,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_std=config.dt_init_std,
            dropout=config.dropout,
        )
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_gaps: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_ssm, z = self.in_proj(x).chunk(2, dim=-1)
        x_ssm = self.ssm(x_ssm, time_gaps, padding_mask)
        out = x_ssm * F.silu(z)
        return residual + self.dropout(self.out_proj(out))


class CTMambaClassifier(BaseClassifier):
    def __init__(
        self, config: CTMambaClassifierConfig, learning_rate: float | None = 1e-3
    ) -> None:
        super().__init__(config, learning_rate)

        self.input_proj = nn.Sequential(
            nn.Linear(config.d_input * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.time_encoding = ContinuousTimeEncoding(
            config.d_model, config.max_freq_scale
        )
        self.mamba_layers = nn.ModuleList(
            CTMambaBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self.pool_attn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.Tanh(),
            nn.Linear(config.d_model // 4, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_classifier_ffn),
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

    def _compute_time_gaps(
        self, timestamps: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        gaps = torch.zeros_like(timestamps)
        gaps[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
        gaps = gaps.clamp(min=0.0).masked_fill(padding_mask, 0.0)
        return gaps

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        padding_mask = make_padding_mask(x, self.config.d_input)
        time_gaps = self._compute_time_gaps(time_steps, padding_mask)

        h = self.input_proj(x) + self.time_encoding(time_steps)
        for block in self.mamba_layers:
            h = block(h, time_gaps, padding_mask)
        h = self.final_norm(h)

        attn = self.pool_attn(h).squeeze(-1).masked_fill(padding_mask, -1e9)
        attn = F.softmax(attn, dim=1)
        pooled = (h * attn.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled)
