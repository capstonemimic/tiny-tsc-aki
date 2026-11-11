"""Exploratory SVD compression of a trained Perceiver checkpoint."""

import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dataset import make_dataset
from evaluation import load_checkpoint, predict, metrics
from utils import sha, write_json
from profiling import profile_model

ENERGIES = (90, 95, 99)


def spectra(model):
    """SVD of learned 2D matrices; only independent Linear modules may change."""
    modules = dict(model.named_modules())
    attention = [
        name
        for name, module in modules.items()
        if isinstance(module, nn.MultiheadAttention)
    ]
    rows, decompositions = [], {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue
        owner, _, leaf = name.rpartition(".")
        module = modules[owner]
        is_attention = any(owner == p or owner.startswith(p + ".") for p in attention)
        eligible = (
            isinstance(module, nn.Linear) and leaf == "weight" and not is_attention
        )
        u, singular, vh = torch.linalg.svd(
            parameter.detach().cpu().double(), full_matrices=False
        )
        energy = singular.square()
        total = float(energy.sum())
        cumulative = (
            (energy.cumsum(0) / total).numpy() if total else np.ones(len(singular))
        )
        ranks = {
            str(e): int(min(len(singular), np.searchsorted(cumulative, e / 100) + 1))
            if total
            else 0
            for e in ENERGIES
        }
        probabilities = (
            singular / singular.sum() if total else torch.zeros_like(singular)
        )
        positive = probabilities[probabilities > 0]
        rows.append(
            {
                "name": name,
                "module": owner,
                "shape": list(parameter.shape),
                "singular_values": singular.tolist(),
                "energy_ranks": ranks,
                "stable_rank": total / float(energy[0]) if total else 0.0,
                "effective_rank": float(torch.exp(-(positive * positive.log()).sum()))
                if total
                else 0.0,
                "eligible_linear": eligible,
                "category": "attention; unchanged"
                if is_attention
                else "linear"
                if eligible
                else "embedding; unchanged"
                if isinstance(module, nn.Embedding)
                else "other; unchanged",
            }
        )
        if eligible:
            decompositions[owner] = (u, singular, vh)
    return rows, decompositions


def factor_linear(layer, decomposition, rank):
    u, singular, vh = decomposition
    assert 1 <= rank <= min(layer.in_features, layer.out_features)
    first = nn.Linear(
        layer.in_features,
        rank,
        bias=False,
        dtype=layer.weight.dtype,
        device=layer.weight.device,
    )
    second = nn.Linear(
        rank,
        layer.out_features,
        bias=layer.bias is not None,
        dtype=layer.weight.dtype,
        device=layer.weight.device,
    )
    with torch.no_grad():
        first.weight.copy_(vh[:rank].to(first.weight))
        second.weight.copy_((u[:, :rank] * singular[:rank]).to(second.weight))
        if layer.bias is not None:
            second.bias.copy_(layer.bias)
    return nn.Sequential(first, second)


def factorized_copy(original, rows, decompositions, energy=None):
    model, changes = copy.deepcopy(original), []
    for row in rows:
        if not row["eligible_linear"]:
            continue
        layer = model.get_submodule(row["module"])
        rank = (
            min(row["shape"])
            if energy is None
            else max(1, row["energy_ranks"][str(energy)])
        )
        old_count = sum(p.numel() for p in layer.parameters())
        new_count = rank * (layer.in_features + layer.out_features) + (
            layer.out_features if layer.bias is not None else 0
        )
        if energy is not None and new_count >= old_count:
            continue
        replacement = factor_linear(layer, decompositions[row["module"]], rank)
        if energy is None:
            torch.testing.assert_close(
                replacement[1].weight @ replacement[0].weight,
                layer.weight,
                rtol=2e-5,
                atol=2e-6,
            )
        parent, _, leaf = row["module"].rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
        changes.append(
            {
                "module": row["module"],
                "rank": rank,
                "original_parameters": old_count,
                "factor_parameters": new_count,
            }
        )
    return model.eval(), changes


def plot_spectra(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for row in rows:
        singular = np.asarray(row["singular_values"])
        if not np.any(singular):
            continue
        cumulative = np.cumsum(singular**2) / np.sum(singular**2)
        ax.plot(
            np.arange(1, len(singular) + 1) / len(singular),
            cumulative,
            color="#2374ab" if row["eligible_linear"] else "#969696",
            alpha=0.32,
            linewidth=1,
        )
    for energy in ENERGIES:
        ax.axhline(energy / 100, color="#a23b3b", linestyle=":", linewidth=0.8)
    ax.set(
        xlabel="Retained rank / full rank",
        ylabel="Cumulative squared singular-value energy",
        title="Perceiver IO: trained weight spectra\nBlue: eligible linear matrices; gray: unchanged matrices",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def validate_replay(replay, previous, replay_metrics, previous_metrics):
    """Bound numerical replay drift while retaining the original strict result."""
    assert (
        replay.shape == previous.shape
        and np.isfinite(replay).all()
        and np.isfinite(previous).all()
    )
    errors = np.abs(replay - previous)
    maximum, p999 = float(errors.max()), float(np.quantile(errors, 0.999))
    assert maximum <= 1e-3, "Replay maximum probability error exceeds 0.001"
    assert p999 <= 2e-4, "Replay 99.9th percentile probability error exceeds 0.0002"
    for metric in ("auroc", "auprc"):
        assert abs(replay_metrics[metric] - previous_metrics[metric]) <= 1e-4
    return {
        "max_probability_error": maximum,
        "probability_error_p999": p999,
        "original_strict_check_passed": bool(maximum <= 2e-4),
        "original_strict_probability_atol": 2e-4,
        "original_strict_violation_count": int((errors > 2e-4).sum()),
        "accepted_rule": "max probability error <= 0.001; p99.9 <= 0.0002; AUROC and AP differences <= 0.0001",
    }


def analyze(data_root, checkpoint, output, fold=0, history="window"):
    """Compare original/full-rank/energy-truncated models on identical windows."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty analysis output directory")
    output.mkdir(parents=True, exist_ok=True)
    original = load_checkpoint("perceiver", checkpoint).cuda()
    original.eval()
    data = make_dataset(data_root, "perceiver", fold, history, "val")
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        reference = predict(original, "perceiver", data, 32)
        reference_metrics = metrics(data.labels, reference)
        rows, decompositions = spectra(original)
        full_rank, _ = factorized_copy(original, rows, decompositions)
        replay = predict(full_rank, "perceiver", data, 32)
        replay_metrics = metrics(data.labels, replay)
        replay_check = validate_replay(
            replay, reference, replay_metrics, reference_metrics
        )
        del full_rank
        original_cost = profile_model("perceiver", copy.deepcopy(original))
        comparisons = [
            dict(
                variant="original",
                gflops_batch32=original_cost["flops_batch32"] / 1e9,
                parameters=sum(p.numel() for p in original.parameters()),
                metrics=reference_metrics,
            )
        ]
        np.save(output / "original_predictions.npy", reference)
        for energy in ENERGIES:
            compressed, changes = factorized_copy(
                original, rows, decompositions, energy
            )
            probability = predict(compressed, "perceiver", data, 32)
            cost = profile_model("perceiver", copy.deepcopy(compressed))
            comparisons.append(
                dict(
                    variant=f"{energy}% energy",
                    gflops_batch32=cost["flops_batch32"] / 1e9,
                    parameters=sum(p.numel() for p in compressed.parameters()),
                    metrics=metrics(data.labels, probability),
                    layers=changes,
                )
            )
            np.save(output / f"predictions_{energy}.npy", probability)
            del compressed
        plot_spectra(rows, output / "spectra.png")
        report = dict(
            checkpoint_sha256=sha(checkpoint),
            fold=fold,
            history=history,
            validation_windows=len(data),
            spectra=rows,
            comparisons=comparisons,
            cost_reference="Batch 32, 128 observed timestamps total; matrix/attention operations",
            full_rank_replay=replay_check,
            interpretation="Exploratory validation-fold compression, without fine-tuning; not a five-fold ranking",
        )
        write_json(output / "svd.json", report)
        return report
    finally:
        torch.set_float32_matmul_precision(previous_precision)
