# Copyright (c) 2026. All rights reserved.

"""Shared wavelet blocks for custom mWDN variants."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence

H_FILTER: list[float] = [
    -0.2304,
    0.7148,
    -0.6309,
    -0.028,
    0.187,
    0.0308,
    -0.0329,
    -0.0106,
]
L_FILTER: list[float] = [
    -0.0106,
    0.0329,
    0.0308,
    -0.187,
    -0.028,
    0.6309,
    0.7148,
    0.2304,
]


def resolve_wavelet_filters(
    wavelet: str | None,
) -> tuple[Sequence[float], Sequence[float]]:
    """Return high-pass and low-pass filters for a named wavelet.

    Returns:
        High-pass and low-pass filter coefficient sequences.

    Raises:
        ImportError: If a custom wavelet is requested and pywavelets is missing.
    """
    if wavelet is None:
        return H_FILTER, L_FILTER

    try:
        import pywt  # noqa: PLC0415  # ty: ignore[unresolved-import]  # optional dependency
    except ImportError as exc:
        msg = "Install pywt to use a custom wavelet, or set wavelet=None"
        raise ImportError(msg) from exc

    wavelet_spec = pywt.Wavelet(wavelet)
    return wavelet_spec.dec_hi, wavelet_spec.dec_lo


def create_banded_weight(
    seq_len: int,
    filter_values: Sequence[float],
) -> np.ndarray:
    """Create a noisy banded wavelet initialization matrix.

    Returns:
        A ``seq_len`` by ``seq_len`` numpy weight matrix.
    """
    max_epsilon = np.min(np.abs(filter_values))
    rng = np.random.default_rng()
    weight = rng.standard_normal((seq_len, seq_len)) * 0.1 * max_epsilon
    for i in range(seq_len):
        for j_idx, j in enumerate(range(i, min(i + len(filter_values), seq_len))):
            weight[i, j] = filter_values[j_idx]
    return weight


def init_banded_linear(
    layer: nn.Linear,
    seq_len: int,
    filter_values: Sequence[float],
) -> None:
    """Initialize a linear layer with the shared banded wavelet matrix."""
    weight = create_banded_weight(seq_len, filter_values)
    layer.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))


class FactoredLinear(nn.Module):
    """Linear layer stored as two low-rank factors."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        """Initialize the low-rank factors and bias."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.left = nn.Parameter(torch.empty(out_features, rank))
        self.right = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.left, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.right, a=np.sqrt(5))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the low-rank linear transform.

        Returns:
            Transformed tensor with the configured output feature dimension.
        """
        # x: (batch, in_features)
        # right.T: (in_features, rank)
        # left.T: (rank, out_features)
        # Result: (batch, out_features)
        return x @ self.right.T @ self.left.T + self.bias

    @staticmethod
    def flop_count(in_features: int, out_features: int, rank: int) -> int:
        """Return FLOPs for a single forward pass through FactoredLinear.

        FLOPs = 2 * in_features * rank (x @ right.T) + 2 * rank * out_features (@ left.T) + out_features (bias)
        """
        return 2 * in_features * rank + 2 * rank * out_features + out_features

    @override
    def extra_repr(self) -> str:
        """Return layer shape information for ``repr``."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}"
        )


def svd_init_factored(
    seq_len: int,
    rank: int,
    filter_values: Sequence[float],
) -> tuple[Tensor, Tensor]:
    """Initialize low-rank factors from a banded wavelet matrix SVD.

    Returns:
        ``(left, right)`` factor tensors.
    """
    weight_full = create_banded_weight(seq_len, filter_values)
    u, s, vt = np.linalg.svd(weight_full, full_matrices=False)
    r = min(rank, len(s))
    left = torch.tensor(u[:, :r] * np.sqrt(s[:r]), dtype=torch.float32)
    right = torch.tensor(np.sqrt(s[:r])[:, None] * vt[:r, :], dtype=torch.float32)
    return left, right


class LinearWaveBlock(nn.Module):
    """Full-rank wavelet block matching tsai mWDN preprocessing."""

    def __init__(
        self,
        seq_len: int,
        wavelet: str | None = None,
        concat_dim: int = 1,
    ) -> None:
        """Initialize full-rank high-pass and low-pass wavelet layers."""
        super().__init__()
        h_filter, l_filter = resolve_wavelet_filters(wavelet)
        self.concat_dim = concat_dim
        self.mWDN_H = nn.Linear(seq_len, seq_len)
        self.mWDN_L = nn.Linear(seq_len, seq_len)
        init_banded_linear(self.mWDN_H, seq_len, h_filter)
        init_banded_linear(self.mWDN_L, seq_len, l_filter)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool1d(2)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the low-pass stream and concatenated wavelet features."""
        hp_out = self.pool(self.sigmoid(self.mWDN_H(x)))
        lp_out = self.pool(self.sigmoid(self.mWDN_L(x)))
        return lp_out, torch.cat((hp_out, lp_out), dim=self.concat_dim)


class FactoredWaveBlock(nn.Module):
    """Low-rank wavelet block backed by ``FactoredLinear`` layers."""

    def __init__(
        self,
        seq_len: int,
        rank: int,
        wavelet: str | None = None,
        concat_dim: int = -1,
    ) -> None:
        """Initialize factored high-pass and low-pass wavelet layers."""
        super().__init__()
        h_filter, l_filter = resolve_wavelet_filters(wavelet)
        self.concat_dim = concat_dim
        self.mWDN_H = FactoredLinear(seq_len, seq_len, rank)
        self.mWDN_L = FactoredLinear(seq_len, seq_len, rank)

        h_left, h_right = svd_init_factored(seq_len, rank, h_filter)
        l_left, l_right = svd_init_factored(seq_len, rank, l_filter)
        with torch.no_grad():
            self.mWDN_H.left.copy_(h_left)
            self.mWDN_H.right.copy_(h_right)
            self.mWDN_L.left.copy_(l_left)
            self.mWDN_L.right.copy_(l_right)

        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool1d(2)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the low-pass stream and concatenated wavelet features."""
        hp_out = self.pool(self.sigmoid(self.mWDN_H(x)))
        lp_out = self.pool(self.sigmoid(self.mWDN_L(x)))
        return lp_out, torch.cat((hp_out, lp_out), dim=self.concat_dim)
