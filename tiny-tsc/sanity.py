"""Patient separation, causal histories, and model input integrity checks."""

from pathlib import Path
from contextlib import ExitStack
import numpy as np
import torch
from torch.nn import functional as F
from dataset import selected_windows, make_dataset, collate_stays, move_batch
from models import REGULAR, forward, loss, make_model


def check_folds(data_root):
    validation = set()
    rows = []
    for fold in range(5):
        membership, parts = selected_windows(Path(data_root), fold)
        current = set(parts["val"].sample_id)
        assert validation.isdisjoint(current)
        validation.update(current)
        for part in ("train", "val"):
            from safetensors import safe_open

            expected = membership.loc[
                membership.validation_fold.eq(fold)
                if part == "val"
                else membership.validation_fold.ne(fold)
            ]
            with safe_open(
                Path(data_root) / f"folds/fold_{fold}/{part}.safetensors",
                framework="numpy",
            ) as file:
                for key in ("subject_id", "hadm_id", "stay_id"):
                    np.testing.assert_array_equal(file.get_tensor(key), expected[key])
                np.testing.assert_array_equal(
                    file.get_tensor("labels"), expected.target.astype(int)
                )
        rows.append(
            dict(
                fold=fold,
                train=len(parts["train"]),
                val=len(parts["val"]),
                shared_subjects=0,
                shared_admissions=0,
                shared_stays=0,
            )
        )
    assert len(validation) == len(membership)
    return dict(
        windows=len(membership),
        prevalence=float(membership.target.mean()),
        folds=rows,
        every_validation_window_once=True,
        causal_prefixes=True,
    )


def check_model(data_root, name, history="window"):
    torch.manual_seed(42)
    model = make_model(name, 42).cuda()
    data = make_dataset(data_root, name, 0, history, "val")
    indices = np.r_[
        np.flatnonzero(data.labels == 0)[:2], np.flatnonzero(data.labels == 1)[:2]
    ]
    if history == "prefix":
        candidates = np.flatnonzero(data.ids.n_windows.to_numpy() > 1)
        if len(candidates):
            shortest = candidates[np.argmin(data.ids.n_windows.to_numpy()[candidates])]
            indices = np.unique(np.r_[indices, shortest])
    samples = [data[int(i)] for i in indices]
    if len(samples) < 2:
        raise ValueError("Sanity checks need examples from both classes")
    if name in REGULAR:
        batch = {
            "x": torch.stack([s[0] for s in samples]).cuda(),
            "labels": torch.stack([s[1] for s in samples]).cuda(),
        }
        if name == "minirocket":
            model.fit(batch["x"], chunksize=len(samples))
    else:
        batch = move_batch(collate_stays(samples))

    def run_samples(items):
        if name in REGULAR:
            packed = {"x": torch.stack([s[0] for s in items]).cuda()}
        else:
            packed = move_batch(collate_stays(items))
        return forward(model, name, packed)

    model.eval()
    history_delta = None
    with torch.no_grad(), ExitStack() as cleanup:
        cleanup.callback(
            torch.set_float32_matmul_precision, torch.get_float32_matmul_precision()
        )
        torch.set_float32_matmul_precision("highest")
        logits = forward(model, name, batch)
        torch.testing.assert_close(logits, forward(model, name, batch))
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(
            logits, run_samples(samples[::-1]).flip(0), rtol=2e-4, atol=2e-4
        )
        singles = torch.cat([run_samples([sample]) for sample in samples])
        torch.testing.assert_close(logits, singles, rtol=2e-4, atol=2e-4)
        if history == "prefix":
            for sample in samples:
                x, times, label = sample
                previous = (times < 0) if name == "micro_diet_gru" else (times < -1)
                if previous.any():
                    changed = x.clone()
                    if name == "micro_diet_gru":
                        changed[previous] = 0
                    else:
                        changed[previous, : x.shape[-1] // 2] = 0
                    difference = (
                        (run_samples([sample]) - run_samples([(changed, times, label)]))
                        .abs()
                        .max()
                    )
                    history_delta = float(difference)
                    assert history_delta > 1e-8, (
                        "History perturbation had no measurable effect"
                    )
                    break
            if history_delta is None:
                raise ValueError(
                    "No multi-window history available for the perturbation check"
                )
        # Ordered history batches flatten windows, so check their padding via their own dataset tests.
        if name not in REGULAR and name != "micro_diet_gru":
            padded = {
                **batch,
                "x": F.pad(batch["x"], (0, 0, 0, 7)),
                "times": torch.cat(
                    [batch["times"], batch["times"][:, -1:].expand(-1, 7)], 1
                ),
            }
            torch.testing.assert_close(
                logits, forward(model, name, padded), rtol=2e-4, atol=2e-4
            )
    model.train()
    objective = loss(model, name, batch)
    objective.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    if name.startswith("mwdn") or name == "micro_diet_gru":
        wavelet = [
            p.grad
            for key, p in model.named_parameters()
            if "mWDN_H" in key or "mWDN_L" in key
        ]
        assert wavelet and all(
            g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
            for g in wavelet
        )
    return dict(
        model=name,
        history=history,
        parameters=sum(p.numel() for p in model.parameters()),
        finite_gradients=True,
        deterministic_eval=True,
        batch_order_agreement=True,
        single_example_agreement=True,
        history_perturbation_max_logit_change=history_delta,
    )
