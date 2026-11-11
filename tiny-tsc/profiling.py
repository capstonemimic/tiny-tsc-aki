"""Checkpoint-based inference FLOPs at explicit window and timestamp counts."""

from collections import Counter
import inspect
from pathlib import Path
import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode
from models import IRREGULAR
from utils import sha


class OperatorAudit(TorchDispatchMode):
    """Retain dispatched operator names so opaque kernels remain reviewable."""

    def __init__(self):
        super().__init__()
        self.calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func)] += 1
        return func(*args, **(kwargs or {}))


def measure(name, checkpoint, timestamps=128, windows=1, expected_hash=None):
    from evaluation import load_checkpoint

    model = load_checkpoint(name, checkpoint, expected_hash)
    result = profile_model(name, model, timestamps, windows)
    result["checkpoint_sha256"] = sha(checkpoint)
    return result


def profile_model(name, model, timestamps=128, windows=1):
    """Profile an instantiated model in place on CPU; timestamps are per window."""
    if timestamps < 2 or windows < 1:
        raise ValueError("Reference input needs at least two timestamps and one window")
    if windows != 1 and name not in ("perceiver", "micro_diet_gru"):
        raise ValueError("Only the two history models accept multiple windows")
    torch.manual_seed(42)
    timestamps_per_window = timestamps
    if name == "perceiver":
        timestamps *= windows
    if name == "minirocket":
        assert bool(model.backbone.prefit) and not model.backbone.fitting, (
            "MiniRocket must load training-fitted kernels/biases before profiling"
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.cpu().eval().requires_grad_(False)
    if name in IRREGULAR:
        values = torch.randn(32, timestamps, 33)
        x = torch.cat((values, torch.ones_like(values)), -1)
        times = torch.linspace(0, 1, timestamps).expand(32, -1)
        args = (x, times)
        if name == "perceiver":
            args += (torch.full((32,), timestamps, dtype=torch.long),)
    elif name == "micro_diet_gru":
        args = (
            torch.randn(32 * windows, 33, 128),
            torch.arange(1 - windows, 1).float().expand(32, -1),
            torch.full((32,), windows, dtype=torch.long),
        )
    else:
        args = (torch.randn(32, 33, 128),)

    original_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        with torch.no_grad():
            reference = model(*args)
        torch.backends.mha.set_fastpath_enabled(False)
        extra = Counter()
        handles = []

        def attention_contraction(module, inputs, output):
            # MultiTimeAttention implements AV as broadcast multiply + reduction.
            query, key, value = inputs[:3]
            extra["mtan_attention_value_contraction"] += (
                2
                * value.shape[0]
                * module.config.n_heads
                * query.shape[1]
                * key.shape[1]
                * value.shape[2]
            )

        def selective_scan(module, inputs, output):
            # Executed parallel_ssm: discretization, doubling scan, state
            # readout (two FLOPs/MAC), and D skip. Projections/conv are already
            # counted by FlopCounterMode. Transcendentals remain excluded.
            batch, length, width = inputs[0].shape
            state = module.d_state
            offset, scan_positions = 1, 0
            while offset < length:
                scan_positions += length - offset
                offset *= 2
            extra["mamba_discretization"] += 3 * batch * length * width * state
            extra["mamba_parallel_scan"] += 3 * batch * width * state * scan_positions
            extra["mamba_state_readout"] += 2 * batch * length * width * state
            extra["mamba_skip"] += 2 * batch * length * width

        def rocket_channels(module, inputs, output):
            batch, channels, length = inputs[0].shape
            if channels > 1:
                extra["minirocket_channel_contraction"] += (
                    2
                    * batch
                    * channels
                    * length
                    * module.num_kernels
                    * module.num_dilations
                )

        def cde_attention_pool(module, inputs, output):
            # The CDE readout applies attention weights by multiply+sum, not bmm.
            hidden = inputs[0]
            extra["cde_attention_value_contraction"] += 2 * hidden.numel()

        def moe_combination(module, inputs, output, width):
            # SparseDispatcher combines only nonzero gate/expert contributions
            # through multiply + index_add, which the matrix counter omits.
            gates = output[0]
            extra["moe_weighted_expert_contraction"] += (
                2 * int(gates.count_nonzero()) * width
            )

        if name == "mtan_vae":
            from models.mta import MultiTimeAttention

            handles = [
                m.register_forward_hook(attention_contraction)
                for m in model.modules()
                if isinstance(m, MultiTimeAttention)
            ]
        elif name == "mamba":
            handles = [
                layer.ssm.register_forward_hook(selective_scan)
                for layer in model.mamba_layers
            ]
        elif name == "minirocket":
            handles = [model.backbone.register_forward_hook(rocket_channels)]
        elif name == "neural_cde":
            handles = [model.pool_attn.register_forward_hook(cde_attention_pool)]
        elif name == "gmlp_moe":
            from functools import partial

            handles = [
                block.gating.register_forward_hook(
                    partial(moe_combination, width=block.experts[0].out_features)
                )
                for block in model.blocks
            ]
        try:
            audit = OperatorAudit()
            with torch.no_grad(), audit, FlopCounterMode(display=False) as counter:
                actual = model(*args)
        finally:
            for handle in handles:
                handle.remove()
        assert actual.shape == (32, 2) and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)
        opaque = [
            op
            for op in audit.calls
            if any(
                term in op
                for term in (
                    "_native_multi_head_attention",
                    "mkldnn_rnn_layer",
                    "_cudnn_rnn",
                    "_scaled_dot_product_flash_attention_for_cpu",
                )
            )
        ]
        assert not opaque, f"Opaque operations require review: {opaque}"
        operations = {str(k): v for k, v in counter.flop_counts["Global"].items()}
        by_module = {
            k: sum(v.values()) for k, v in counter.flop_counts.items() if k != "Global"
        }
        assert counter.get_total_flops() > 0
        if name == "mtan_vae":
            for recurrent in ("encoder.gru_rnn", "classifier.gru_rnn"):
                assert any(
                    k.endswith(recurrent) and v > 0 for k, v in by_module.items()
                )
            assert not any("decoder" in k for k in by_module)
            assert extra["mtan_attention_value_contraction"] > 0
        if name.startswith("mwdn") or name == "micro_diet_gru":
            for band in ("mWDN_H", "mWDN_L"):
                assert (
                    sum(k.endswith(band) and v > 0 for k, v in by_module.items()) == 3
                )
        if name == "micro_diet_gru":
            assert any(k.endswith("ordered") and v > 0 for k, v in by_module.items())
        if name == "gmlp_moe":
            assert extra["moe_weighted_expert_contraction"] > 0
            assert any(
                "experts" in key and value > 0 for key, value in by_module.items()
            )
        if name == "neural_cde":
            assert (
                extra["cde_attention_value_contraction"]
                == 2 * 32 * timestamps * model.config.d_hidden
            )
            assert any("bmm" in key and value > 0 for key, value in operations.items())
        if name == "ode_rnn":
            assert any(
                "gru_unit" in key and value > 0 for key, value in by_module.items()
            )
        module_sources = {inspect.getfile(type(m)) for m in model.modules()}
        return {
            "model": name,
            "parameters_trainable": trainable,
            "parameters_total": total,
            "input_shapes": [list(a.shape) for a in args],
            "reference_timestamps": timestamps if name in IRREGULAR else 128,
            "reference_windows": windows,
            "timestamps_per_window": timestamps_per_window,
            "matrix_conv_flops_batch32": counter.get_total_flops(),
            "supplemental_flops": dict(extra),
            "flops_batch32": counter.get_total_flops() + sum(extra.values()),
            "operations": operations,
            "by_module": by_module,
            "module_source_sha256": {p: sha(Path(p)) for p in sorted(module_sources)},
            "dispatched_operator_calls": dict(audit.calls),
            "fastpath_output_agreement": True,
        }
    finally:
        torch.backends.mha.set_fastpath_enabled(original_fastpath)
