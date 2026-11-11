# Copyright (c) 2026. All rights reserved.

"""Ultra-compact wavelet + InceptionTimePlus model."""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.utils import build_model

from ._wavelet import LinearWaveBlock as WaveBlock


class FixedWaveBlocks(nn.Module):
    """Multi-resolution full-rank wavelet preprocessing blocks."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        levels: int = 3,
        wavelet: str | None = None,
    ) -> None:
        """Initialize the stacked wavelet blocks."""
        super().__init__()
        self.c_in = c_in
        self.blocks = nn.ModuleList(
            WaveBlock(seq_len // 2**i, wavelet=wavelet, concat_dim=1)
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


class mWDN_micro(nn.Module):  # noqa: N801  # follows tsai model naming convention
    """mWDN_micro: compact wavelet preprocessing plus a small backbone."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        levels: int = 3,
        wavelet: str | None = None,
    ) -> None:
        """Initialize mWDN_micro."""
        super().__init__()
        self.preprocess = FixedWaveBlocks(
            c_in,
            seq_len,
            levels=levels,
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
