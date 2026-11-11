from dataclasses import dataclass

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
class PerceiverClassifierConfig:
    d_input: int = 34
    d_model: int = 128
    max_freq_scale: float = 10.0
    n_latents: int = 32
    n_self_attn_layers: int = 6
    n_heads: int = 8
    d_ffn: int = 512
    d_out: int = 2
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    class_weights: list[float] | None = None
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, context_dim: int | None = None) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        if self.norm_context is not None:
            context = self.norm_context(context)
        return (
            self.fn(x, context=context, mask=mask)
            if context is not None
            else self.fn(x)
        )


class CrossAttention(nn.Module):
    def __init__(
        self, dim: int, context_dim: int, n_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim,
            n_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        out, _ = self.attn(x, context, context, key_padding_mask=mask)
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


def ensure_nonempty_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    all_masked = mask.all(dim=1)
    if not all_masked.any():
        return mask

    mask = mask.clone()
    mask[all_masked, 0] = False
    return mask


class FeedForward(nn.Module):
    def __init__(self, dim: int, d_ffn: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, d_ffn * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        )
        return x / keep_prob * mask


class PerceiverBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        fn: nn.Module,
        context_dim: int | None = None,
        d_ffn: int | None = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.prenorm = PreNorm(dim, fn, context_dim)
        self.ff = PreNorm(dim, FeedForward(dim, d_ffn or dim * 4, dropout))
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.prenorm(x, context=context, mask=mask))
        x = x + self.ff(x)
        return x


class PerceiverIO(nn.Module):
    def __init__(self, config: PerceiverClassifierConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.input_proj = nn.Linear(config.d_input * 2, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model, config.max_freq_scale)

        self.latents = nn.Parameter(torch.randn(1, config.n_latents, d_model) * 0.02)

        self.cross_attn = PerceiverBlock(
            d_model,
            CrossAttention(d_model, d_model, config.n_heads, config.dropout),
            context_dim=d_model,
            d_ffn=config.d_ffn,
            dropout=config.dropout,
        )

        dp_rates = [
            config.drop_path_rate * i / max(config.n_self_attn_layers - 1, 1)
            for i in range(config.n_self_attn_layers)
        ]
        self.self_attn_layers = nn.ModuleList(
            [
                PerceiverBlock(
                    d_model,
                    SelfAttention(d_model, config.n_heads, config.dropout),
                    d_ffn=config.d_ffn,
                    dropout=config.dropout,
                    drop_path=dp_rates[i],
                )
                for i in range(config.n_self_attn_layers)
            ]
        )

        self.class_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.decoder_cross_attn = PerceiverBlock(
            d_model,
            CrossAttention(d_model, d_model, config.n_heads, config.dropout),
            context_dim=d_model,
            d_ffn=config.d_ffn,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        padding_mask = ensure_nonempty_attention_mask(
            make_padding_mask(x, self.config.d_input)
        )

        h = self.input_proj(x) + self.time_encoding(timestamps)

        latents = self.latents.expand(x.size(0), -1, -1)
        latents = self.cross_attn(latents, context=h, mask=padding_mask)
        for layer in self.self_attn_layers:
            latents = layer(latents)

        query = self.class_query.expand(x.size(0), -1, -1)
        out = self.decoder_cross_attn(query, context=latents)
        return out.squeeze(1)


class PerceiverClassifier(BaseClassifier):
    def __init__(
        self, config: PerceiverClassifierConfig, learning_rate: float | None = 1e-3
    ) -> None:
        super().__init__(config, learning_rate)
        self.encoder = PerceiverIO(config)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_out),
        )
        self.loss_fn = FocalLoss(
            alpha=config.class_weights,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, time_steps))
