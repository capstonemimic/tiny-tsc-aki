# Copyright (c) 2026. All rights reserved.

"""Ultra-compact mWDN variant with SVD-compressed wavelet blocks."""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.utils import build_model

from ._wavelet import FactoredWaveBlock as DietWaveBlock


class DietWaveBlocks(nn.Module):
    """Multi-resolution low-rank wavelet preprocessing blocks."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        levels: int = 3,
        rank: int = 16,
        wavelet: str | None = None,
    ) -> None:
        """Initialize the stacked low-rank wavelet blocks."""
        super().__init__()
        self.c_in = c_in
        self.blocks = nn.ModuleList(
            DietWaveBlock(seq_len // 2**i, rank, wavelet=wavelet, concat_dim=1)
            for i in range(levels)
        )
        self.pool = nn.AdaptiveAvgPool1d(seq_len)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Return concatenated features from every wavelet level."""
        features = []
        for block in self.blocks:
            x, out_ = block(x)
            features.append(self.pool(out_))
        return torch.cat(features, dim=1)


class mWDN_micro_diet(nn.Module):  # noqa: N801  # follows tsai model naming convention
    """mWDN_micro_diet: compact backbone with low-rank wavelet blocks."""

    def __init__(  # noqa: PLR0913,PLR0917  # model constructors mirror tsai APIs
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        levels: int = 3,
        rank: int = 16,
        wavelet: str | None = None,
    ) -> None:
        """Initialize mWDN_micro_diet."""
        super().__init__()
        self.preprocess = DietWaveBlocks(
            c_in,
            seq_len,
            levels=levels,
            rank=rank,
            wavelet=wavelet,
        )
        gmlp_c_in = c_in * 2 * levels
        self._model = build_model(
            InceptionTimePlus,
            gmlp_c_in,
            c_out,
            seq_len=seq_len,
            nf=12,
            depth=2,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Run wavelet preprocessing and the classifier backbone.

        Returns:
            Model logits.
        """
        x = self.preprocess(x)
        return self._model(x)
