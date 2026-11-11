import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def make_padding_mask(x: torch.Tensor, d_input: int) -> torch.Tensor:
    mask_portion = x[..., d_input:]
    has_observation = mask_portion.sum(dim=-1) > 0
    return ~has_observation


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        p_t = torch.exp(-ce)
        weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            weight = self.alpha[targets] * weight
        return (weight * ce).mean()


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_freq_scale: float = 10.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_freq = (d_model - 1) // 2
        self.has_extra = (d_model - 1) - 2 * self.n_freq > 0

        init_freq = (
            torch.exp(torch.linspace(0, -math.log(max_freq_scale), self.n_freq))
            * 2
            * math.pi
        )
        self.freqs = nn.Parameter(init_freq)
        self.phases = nn.Parameter(torch.zeros(self.n_freq))
        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        t = timestamps.unsqueeze(-1)
        linear_out = self.linear(t)
        angles = t * self.freqs + self.phases
        parts = [linear_out, torch.sin(angles), torch.cos(angles)]
        if self.has_extra:
            parts.append(torch.zeros_like(linear_out))
        return torch.cat(parts, dim=-1)
