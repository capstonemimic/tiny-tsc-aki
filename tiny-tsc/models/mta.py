import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEmbedding(nn.Module):
    def __init__(
        self, embed_time: int, learn_embeddings: bool = True, freq: float = 10.0
    ):
        super().__init__()
        self.embed_time = embed_time
        self.learn_embeddings = learn_embeddings
        self.freq = freq

        div_term = torch.exp(
            torch.arange(0, embed_time, 2).float() * -(math.log(freq) / embed_time)
        )
        self.register_buffer("div_term", div_term)

        if learn_embeddings:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        tt = tt.unsqueeze(-1)
        out1 = self.linear(tt)
        out2 = torch.sin(self.periodic(tt))
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tt.shape
        device = tt.device
        pe = torch.zeros(batch_size, seq_len, self.embed_time, device=device)
        position = 48.0 * tt.unsqueeze(2)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        if self.learn_embeddings:
            return self.learn_time_embedding(time_steps)
        return self.fixed_time_embedding(time_steps)


@dataclass(frozen=True, slots=True)
class MultiTimeAttentionConfig:
    d_input: int
    n_hidden: int = 16
    embed_time: int = 16
    n_heads: int = 1

    def __post_init__(self) -> None:
        if self.embed_time % self.n_heads != 0:
            raise ValueError(
                f"embed_time ({self.embed_time}) must be divisible by n_heads ({self.n_heads})"
            )


class MultiTimeAttention(nn.Module):
    def __init__(self, config: MultiTimeAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.linears: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(config.embed_time, config.embed_time),
                nn.Linear(config.embed_time, config.embed_time),
                nn.Linear(config.d_input * config.n_heads, config.n_hidden),
            ]
        )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> torch.Tensor:
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            layer(x)
            .view(
                x.size(0),
                -1,
                self.config.n_heads,
                self.config.embed_time // self.config.n_heads,
            )
            .transpose(1, 2)
            for layer, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.config.n_heads * dim)
        return self.linears[-1](x)
