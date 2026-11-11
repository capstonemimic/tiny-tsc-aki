"""Summarize paired five-fold scores, compute, and optimizer-step counts."""

from pathlib import Path
import json
import numpy as np
from scipy.stats import t
from evaluation import verify_result
from models import WINDOW_MODELS, HISTORY_MODELS
from utils import write_json


def interval(values):
    values = np.asarray(values, dtype=float)
    return (
        f"{values.mean():.4f} ± {t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5):.4f}"
    )


def summarize(data_root, output_root, destination):
    root, destination = Path(output_root), Path(destination)
    rows, flags = [], []
    selections = [(name, "window") for name in WINDOW_MODELS]
    selections += [
        ("micro_diet_gru", "window"),
        *[(name, "prefix") for name in HISTORY_MODELS],
    ]
    for name, history in selections:
        results = []
        costs = []
        for fold in range(5):
            folder = root / name / history / f"fold_{fold}"
            if not (folder / "result.json").exists():
                continue
            saved = verify_result(folder, data_root)
            results.append(saved)
            values = saved["metrics"]
            if (
                values["auroc"] > 0.95
                or values["auprc"] > 0.9
                or values["accuracy"] > 0.95
                or values["probability_std"] < 1e-4
                or values["sensitivity"] < 0.1
            ):
                flags.append(
                    dict(model=name, history=history, fold=fold, metrics=values)
                )
            if (folder / "profile.json").exists():
                profile = json.loads((folder / "profile.json").read_text())
                assert profile["checkpoint_sha256"] == saved["checkpoint_sha256"]
                costs.append(profile)
        row = dict(model=name, history=history, folds=len(results))
        if len(results) == 5:
            assert len({r["parameters"] for r in results}) == 1
            row.update(
                auroc=interval([r["metrics"]["auroc"] for r in results]),
                auprc=interval([r["metrics"]["auprc"] for r in results]),
                parameters=results[0]["trainable_parameters"],
                mean_updates_to_best=float(
                    np.mean([r["optimizer_steps_to_best"] for r in results])
                ),
            )
        if costs:
            first = costs[0]
            assert all(p["input_shapes"] == first["input_shapes"] for p in costs), (
                "Cannot mix FLOPs reference shapes"
            )
            assert all(
                p["parameters_total"] == r["parameters"] for p in costs for r in results
            )
            row.update(
                gflops_batch32=first["flops_batch32"] / 1e9,
                reference_shapes=first["input_shapes"],
            )
        rows.append(row)
    lines = [
        "# 24h model results",
        "",
        "AUROC and AP use the same selected checkpoint and complete validation predictions. "
        "Validation also selects the checkpoint. Intervals are 95% Student-t half-widths across five folds.",
        "",
        "| Model | History | Folds | AUROC | AUPRC (AP) | Trainable parameters | GFLOPs / batch 32 | Updates to best |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(key, "pending"))
                for key in [
                    "model",
                    "history",
                    "folds",
                    "auroc",
                    "auprc",
                    "parameters",
                    "gflops_batch32",
                    "mean_updates_to_best",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "FLOPs depend on the recorded reference shapes. Missing five-fold groups remain pending. "
        "Updates to the selected checkpoint are a retrospective proxy, not proof of convergence.",
        f"Screening flags: {len(flags)}; inspect the accompanying JSON.",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    write_json(destination.with_suffix(".json"), {"rows": rows, "flags": flags})
    return rows
