# Copyright (c) 2026. All rights reserved.

"""Mixture-of-experts variant of TSAI's gMLP with true sparse dispatch.

Architecture
------------
Each block applies:

1. LayerNorm
2. ``channel_proj1`` (d_model → d_ffn*2) + GELU  — shared linear expansion
3. ``SGU`` (Spatial Gating Unit)                  — shared cross-token mixing
4. **Sparse MoE projection** (d_ffn → d_model)    — per-token sparse dispatch
5. Residual add

The SGU (the only cross-position operator in gMLP) is kept as a *shared*
module because it requires a fixed ``seq_len`` that cannot be preserved
when flattening unrelated tokens for sparse dispatch.  The final channel
contraction is replaced by a mixture of linear experts routed per-token.

Routing
-------
* ``NoisyTopKGating`` (Shazeer et al., 2017 "Outrageously Large Neural
  Networks") adds per-expert learned noise during training for better
  load-balance exploration.
* ``SparseDispatcher`` sends each token only to its ``top_k`` selected
  experts — truly sparse FLOPs, unlike the previous "run all, gather
  top-k" approach.
* Load-balancing loss is the sum of CV² of importance and CV² of load,
  both scaled by ``loss_coef`` (default 1e-2). Exposed via
  ``model.moe_aux_loss`` but **not** auto-added to the training loss.

Based on:
  - Shazeer et al. (2017), https://arxiv.org/abs/1701.06538
  - https://github.com/YeonwooSung/Pytorch_mixture-of-experts

Usage::

    from models.gmlp_moe import gMLP_MoE
    model = gMLP_MoE(c_in=34, c_out=2, seq_len=128,
                     d_model=256, d_ffn=512, depth=6,
                     num_experts=8, top_k=2)
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F  # noqa: N812  # conventional naming for this domain
from torch import nn
from torch.distributions.normal import Normal


class SparseDispatcher:
    """Sparse token dispatcher (Shazeer et al., 2017).

    Routes a flat batch of tokens to per-expert sub-batches based on a
    sparse ``gates`` matrix, then recombines weighted expert outputs.
    Only tokens with non-zero gate values are sent to each expert.

    Adapted for GPU: replaces ``.numpy()`` calls with ``.tolist()``.

    Parameters
    ----------
    num_experts : int
    gates : Tensor  shape [N, num_experts], mostly zero (from top-k gating)
    """

    def __init__(self, num_experts: int, gates: torch.Tensor) -> None:
        """Initialize the dispatcher with gate matrix.

        Computes dispatch indices and part sizes from the sparse gates.

        Args:
            num_experts: Number of experts.
            gates: Sparse gate matrix of shape [N, num_experts].
        """
        self._gates = gates
        self._num_experts = num_experts
        nonzero = torch.nonzero(gates)
        order = nonzero[:, 1].argsort(stable=True)
        sorted_nz = nonzero[order]
        self._expert_index = sorted_nz[:, 1:2]
        self._batch_index = sorted_nz[:, 0]
        self._part_sizes: list[int] = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp: torch.Tensor) -> list[torch.Tensor]:
        """Slice ``inp`` into per-expert sub-batches.

        Parameters
        ----------
        inp : Tensor  shape [N, d]

        Returns:
            list of ``num_experts`` tensors, shape [n_i, d].
        """
        return list(torch.split(inp[self._batch_index], self._part_sizes, dim=0))

    def combine(self, expert_out: list[torch.Tensor]) -> torch.Tensor:
        """Weighted recombination of expert outputs.

        Parameters
        ----------
        expert_out : list of ``num_experts`` tensors, shape [n_i, d_out]

        Returns:
            Tensor  shape [N, d_out] with gate-weighted expert contributions.
        """
        stitched = torch.cat(expert_out, dim=0)
        stitched *= self._nonzero_gates
        out = torch.zeros(
            self._gates.size(0),
            stitched.size(1),
            dtype=stitched.dtype,
            device=stitched.device,
        )
        out.index_add_(0, self._batch_index, stitched)
        return out

    def expert_to_gates(self) -> list[torch.Tensor]:
        """Return per-expert gate weight lists."""
        return list(torch.split(self._nonzero_gates, self._part_sizes, dim=0))


class NoisyTopKGating(nn.Module):
    """Noisy top-k gating with CV-squared load-balancing loss.

    During training, adds per-expert learned noise to the router logits.
    This improves exploration and discourages winner-take-all routing.
    During inference, noise is disabled.

    Parameters
    ----------
    d_in : int        Input feature dimension.
    num_experts : int
    top_k : int
    loss_coef : float  Multiplier on the returned aux loss.
    """

    def __init__(
        self,
        d_in: int,
        num_experts: int,
        top_k: int,
        loss_coef: float = 1e-2,
    ) -> None:
        """Initialize gating network.

        Args:
            d_in: Input feature dimension.
            num_experts: Total number of experts.
            top_k: Number of experts per token.
            loss_coef: Multiplier on the load-balancing aux loss.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.w_gate = nn.Parameter(torch.zeros(d_in, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(d_in, num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _cv_squared(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.zeros((), device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        noise_std: torch.Tensor,
        top_vals: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth estimate of P(logit_i is in top-k) for load computation.

        Returns:
            Probability that each logit is in the top-k set.
        """
        dist = Normal(
            torch.zeros(1, device=clean.device),
            torch.ones(1, device=clean.device),
        )
        n = clean.size(0)
        m = top_vals.size(1)
        flat = top_vals.flatten()
        arange = torch.arange(n, device=clean.device)
        thresh_in = flat[arange * m + self.top_k].unsqueeze(1)
        thresh_out = flat[arange * m + self.top_k - 1].unsqueeze(1)
        is_in = noisy > thresh_in
        prob_in = dist.cdf((clean - thresh_in) / noise_std)
        prob_out = dist.cdf((clean - thresh_out) / noise_std)
        return torch.where(is_in, prob_in, prob_out)

    @override
    def forward(  # noqa: PLR0914  # complex but necessary logic
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse gate matrix and load-balancing loss.

        Parameters
        ----------
        x : Tensor  shape [N, d_in]

        Returns:
            gates : Tensor  shape [N, num_experts] — sparse, non-negative,
                same dtype as x
            aux_loss : scalar
        """
        orig_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            wg = self.w_gate.float()
            wn = self.w_noise.float()
            clean = xf @ wg
            if self.training:
                noise_std = self.softplus(xf @ wn) + 1e-2
                noisy = clean + torch.randn_like(clean).detach() * noise_std.detach()
            else:
                noisy = clean
                noise_std = torch.ones_like(clean)

            with torch.no_grad():
                _, top_idx = noisy.topk(
                    min(self.top_k + 1, self.num_experts),
                    dim=-1,
                )
            clean_at_selected = clean.gather(1, top_idx)
            top_k_gates = self.softmax(clean_at_selected[:, : self.top_k])
            zeros = torch.zeros_like(clean)
            gates = zeros.scatter(1, top_idx[:, : self.top_k], top_k_gates)

            importance = gates.sum(0)
            if self.training and self.top_k < self.num_experts:
                noisy_top_logits = noisy.gather(1, top_idx)
                load = self._prob_in_top_k(
                    clean,
                    noisy,
                    noise_std,
                    noisy_top_logits,
                ).sum(0)
            else:
                load = (gates > 0).sum(0).float()

        aux = (self._cv_squared(importance) + self._cv_squared(load)) * self.loss_coef
        return gates.to(orig_dtype), aux


class _SpatialGatingUnit(nn.Module):
    """Same SGU as upstream tsai.models.gMLP."""

    def __init__(self, d_ffn: int, seq_len: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        if self.spatial_proj.bias is None:
            msg = "spatial_proj bias is None"
            raise ValueError(msg)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.normal_(self.spatial_proj.weight, std=1e-6)

    @override
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        return u * v


class _gMLPSparseBlock(nn.Module):  # noqa: N801  # matches underlying model/class name
    """gMLP block with the channel-projection MoE-ified via sparse dispatch.

    Factored into a shared prefix (LayerNorm + channel_proj1 + GELU + SGU)
    and per-token sparse experts (linear channel_proj2 equivalents).  The
    SGU stays shared because it requires a fixed ``seq_len`` that cannot be
    maintained when tokens from different positions are packed into
    per-expert sub-batches.

    Parameters
    ----------
    d_model     : hidden width of the residual stream.
    d_ffn       : inner FFN dimension (after SGU output = d_ffn).
    seq_len     : number of tokens; needed by the SGU Conv1d.
    num_experts : total experts.
    top_k       : experts per token (1 <= top_k <= num_experts).
    loss_coef   : multiplier on the load-balancing aux loss.
    """

    def __init__(  # noqa: PLR0913, PLR0917  # many params required by function signature
        self,
        d_model: int,
        d_ffn: int,
        seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        loss_coef: float = 1e-2,
    ) -> None:
        """Initialize the sparse gMLP block.

        Args:
            d_model: Hidden width of the residual stream.
            d_ffn: Inner FFN dimension.
            seq_len: Number of tokens.
            num_experts: Total number of experts.
            top_k: Experts per token.
            loss_coef: Multiplier on load-balancing aux loss.

        Raises:
            ValueError: If top_k is not in [1, num_experts].
        """
        super().__init__()
        if not 1 <= top_k <= num_experts:
            msg = (
                f"top_k must be in [1, num_experts]; "
                f"got top_k={top_k}, num_experts={num_experts}"
            )
            raise ValueError(msg)
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.sgu = _SpatialGatingUnit(d_ffn, seq_len)
        self.gating = NoisyTopKGating(d_ffn, num_experts, top_k, loss_coef)
        self.experts = nn.ModuleList(
            [nn.Linear(d_ffn, d_model) for _ in range(num_experts)],
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.last_aux_loss: torch.Tensor | None = None

    @override
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, t, _ = x.shape

        h = self.norm(x)
        h = F.gelu(self.channel_proj1(h))
        h = self.sgu(h)

        flat = h.reshape(b * t, -1)
        gates, aux = self.gating(flat)

        self.last_aux_loss = aux

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(flat)
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        out_flat = dispatcher.combine(expert_outputs)
        out = out_flat.reshape(b, t, -1)

        return residual + out


class gMLP_MoE(nn.Module):  # noqa: N801  # matches underlying model/class name
    """Drop-in replacement for tsai.models.gMLP with sparse MoE dispatch.

    Signature matches the TSAI ``(c_in, c_out, seq_len, ...)`` convention so
    it plugs straight into ``main.create_model``.

    Routing uses noisy top-k gating (Shazeer et al., 2017) with true sparse
    dispatch: each token is only processed by its ``top_k`` selected experts,
    making actual compute proportional to ``top_k / num_experts`` per block.
    """

    def __init__(  # noqa: PLR0913, PLR0917  # many params required by function signature
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        patch_size: int = 1,
        d_model: int = 256,
        d_ffn: int = 512,
        depth: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        loss_coef: float = 1e-2,
    ) -> None:
        """Initialize the gMLP-MoE model.

        Args:
            c_in: Number of input channels.
            c_out: Number of output classes.
            seq_len: Sequence length.
            patch_size: Patch size for patcher conv.
            d_model: Hidden dimension.
            d_ffn: Inner FFN dimension.
            depth: Number of gMLP blocks.
            num_experts: Total experts per block.
            top_k: Top-k experts per token.
            loss_coef: Load-balancing loss coefficient.

        Raises:
            ValueError: If seq_len is not divisible by patch_size.
        """
        super().__init__()
        if seq_len % patch_size != 0:
            msg = "`seq_len` must be divisible by `patch_size`"
            raise ValueError(msg)
        self.patcher = nn.Conv1d(
            c_in,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        effective_len = seq_len // patch_size
        self.blocks = nn.ModuleList(
            [
                _gMLPSparseBlock(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    seq_len=effective_len,
                    num_experts=num_experts,
                    top_k=top_k,
                    loss_coef=loss_coef,
                )
                for _ in range(depth)
            ],
        )
        self.head = nn.Linear(d_model, c_out)

    @property
    def moe_aux_loss(self) -> torch.Tensor:
        """Mean load-balancing loss across all blocks from the last forward pass.

        Returns zero if no forward pass has been run yet.
        Add ``alpha * model.moe_aux_loss`` to your training loss to encourage
        uniform expert utilisation (typical alpha: 1.0 since loss_coef
        already scales the raw CV² values).
        """
        terms = [b.last_aux_loss for b in self.blocks if b.last_aux_loss is not None]
        if not terms:
            return torch.zeros(())
        return torch.stack(terms).mean()

    @override
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: patchify, apply blocks, pool, classify.

        Args:
            x: Input tensor of shape [B, C_in, seq_len].

        Returns:
            Logits of shape [B, C_out].
        """
        patches = self.patcher(x)
        b, c, _ = patches.shape
        tokens = patches.permute(0, 2, 1).contiguous().view(b, -1, c)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens.mean(dim=1))
