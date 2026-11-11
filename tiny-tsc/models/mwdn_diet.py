# Copyright (c) 2026. All rights reserved.

"""Low-rank mWDN model variant with factored wavelet weight matrices.

Replaces the full-rank nn.Linear(seq_len, seq_len) in each WaveBlock
with a factored representation. This reduces wavelet parameters from
O(seq_len^2) to O(seq_len * rank).

Checkpoints saved by ``compress_mwdn.py`` can be loaded via
``mWDN_diet.from_compressed_checkpoint()`` for evaluation or resumed training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.utils import build_model

from ._wavelet import FactoredWaveBlock as DietWaveBlock

if TYPE_CHECKING:
    from pathlib import Path


class mWDN_diet(nn.Module):  # noqa: N801  # follows tsai model naming convention
    """mWDN with factored low-rank wavelet weight matrices."""

    def __init__(  # noqa: PLR0913,PLR0917  # model constructors mirror tsai APIs
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        rank: int = 16,
        levels: int = 3,
        wavelet: str | None = None,
    ) -> None:
        """Initialize the low-rank wavelet front-end and classifier backbone."""
        super().__init__()
        self.levels = levels
        self.rank = rank
        self.blocks = nn.ModuleList(
            DietWaveBlock(seq_len // 2**i, rank, wavelet=wavelet, concat_dim=-1)
            for i in range(levels)
        )
        self._model = build_model(InceptionTimePlus, c_in, c_out, seq_len=seq_len)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Run wavelet preprocessing and the classifier backbone.

        Returns:
            Model logits.
        """
        features: list[Tensor] = []
        for block in self.blocks:
            x, out_ = block(x)
            features.append(out_)
        return self._model(torch.cat(features, dim=-1))

    @classmethod
    def from_compressed_checkpoint(
        cls,
        checkpoint_path: str | Path,
        c_in: int,
        c_out: int,
        seq_len: int,
    ) -> mWDN_diet:
        """Load an mWDN_diet model from a compressed checkpoint.

        Returns:
            Model initialized from compressed checkpoint weights.
        """
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        rank = int(state.get("rank", 16))
        levels = sum(
            1
            for k in model_state
            if k.startswith("blocks.") and k.endswith(".mWDN_H.left")
        )

        model = cls(c_in, c_out, seq_len, rank=rank, levels=levels)

        mapping = {}
        for k, v in model_state.items():
            if k.endswith(".lowrank_left"):
                base = k.removesuffix(".lowrank_left")
                mapping[f"{base}.left"] = v
            elif k.endswith(".lowrank_right"):
                base = k.removesuffix(".lowrank_right")
                mapping[f"{base}.right"] = v
            elif k.endswith(".lowrank_bias"):
                base = k.removesuffix(".lowrank_bias")
                mapping[f"{base}.bias"] = v
            else:
                mapping[k] = v

        model.load_state_dict(mapping, strict=False)
        return model
