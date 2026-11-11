"""Paired validation metrics, inference, and checkpoint verification."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from dataset import make_dataset, collate_stays, move_batch
from models import REGULAR, make_model, forward
from utils import sha, write_json


def metrics(y, p):
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    if len(y) != len(p) or set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Evaluation requires matching predictions and both binary classes"
        )
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Invalid positive-class probabilities")
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "prevalence": float(np.mean(y)),
        "probability_range": [float(p.min()), float(p.max())],
        "probability_std": float(p.std()),
    }


from fastai.learner import Metric


def binary_scores(labels, probability):
    labels = np.asarray(labels).reshape(-1)
    probability = np.asarray(probability).reshape(-1)
    if len(labels) != len(probability) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            "Evaluation requires matching predictions and both binary classes"
        )
    if (
        not np.isfinite(probability).all()
        or not ((0 <= probability) & (probability <= 1)).all()
    ):
        raise ValueError("Invalid positive-class probabilities")
    tn, fp, fn, tp = confusion_matrix(labels, probability >= 0.5, labels=[0, 1]).ravel()
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    return {
        "roc_auc_multiclass": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "accuracy": float((tp + tn) / len(labels)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "prevalence": float(labels.mean()),
    }


class BinaryValidationMetric(Metric):
    """Never average batch AUROC, AP, or class recall."""

    def __init__(self, name):
        self._name = name
        self.reset()

    def reset(self):
        self.labels, self.probabilities = [], []

    def accumulate(self, learn):
        self.labels.append(learn.yb[0].detach().cpu().reshape(-1))
        self.probabilities.append(learn.pred.detach().float().softmax(-1)[:, 1].cpu())

    @property
    def value(self):
        if not self.labels:
            return None
        return binary_scores(
            torch.cat(self.labels).numpy(), torch.cat(self.probabilities).numpy()
        )[self._name]

    @property
    def name(self):
        return self._name


def validation_metrics():
    return [
        BinaryValidationMetric(name)
        for name in ("roc_auc_multiclass", "auprc", "balanced_accuracy")
    ]


def predict(model, name, data, batch_size=32):
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=None if name in REGULAR else collate_stays,
    )
    model.eval()
    predictions = []
    observed_labels = []
    with torch.no_grad():
        for raw in loader:
            batch = (
                {"x": raw[0].cuda(), "labels": raw[1].cuda()}
                if name in REGULAR
                else move_batch(raw)
            )
            logits = forward(model, name, batch)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite inference output")
            predictions.append(logits.float().softmax(-1)[:, 1].cpu().numpy())
            observed_labels.append(batch["labels"].cpu().numpy())
    np.testing.assert_array_equal(np.concatenate(observed_labels), data.labels)
    return np.concatenate(predictions)


def load_checkpoint(name, checkpoint, expected_hash=None):
    checkpoint = Path(checkpoint)
    if expected_hash is not None and sha(checkpoint) != expected_hash:
        raise ValueError("Checkpoint hash does not match its result manifest")
    model = make_model(name)
    # Fastai saves optimizer containers alongside model weights in older artifacts.
    from fastcore.foundation import L

    with torch.serialization.safe_globals([L]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved.get("model", saved), strict=True)
    return model


def save_predictions(data, probability, output):
    frame = data.ids.drop(
        columns=["positions", "window_indices"], errors="ignore"
    ).copy()
    frame["labels"], frame["probability"] = data.labels, probability
    frame.to_parquet(output, index=False)
    return metrics(data.labels, probability)


def verify_result(folder, data_root):
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    for name, key in [
        ("best.pt", "checkpoint_sha256"),
        ("val_predictions.parquet", "prediction_sha256"),
    ]:
        if sha(folder / name) != result[key]:
            raise ValueError(f"Changed artifact: {folder / name}")
    from dataset import selected_windows, stay_rows
    from models import REGULAR

    _, parts = selected_windows(Path(data_root), result["fold"])
    expected = parts["val"]
    if result["model"] in REGULAR:
        membership, _ = stay_rows(Path(data_root))
        expected = membership.loc[membership.validation_fold.eq(result["fold"])].copy()
        expected["sample_id"] = (
            expected.stay_id.astype(str)
            + ":"
            + expected.window_index.astype(int).astype(str)
        )
    frame = pd.read_parquet(folder / "val_predictions.parquet")
    assert frame.sample_id.is_unique and set(frame.sample_id) == set(expected.sample_id)
    expected = expected.set_index("sample_id").loc[frame.sample_id]
    for key in ("subject_id", "hadm_id", "stay_id"):
        np.testing.assert_array_equal(frame[key], expected[key])
    np.testing.assert_array_equal(frame.labels, expected.target)
    computed = metrics(frame.labels.to_numpy(), frame.probability.to_numpy())
    for key in ("auroc", "auprc"):
        assert abs(computed[key] - result["metrics"][key]) < 1e-10
    for relative, digest in result["identity"]["data"].items():
        assert sha(Path(data_root) / relative) == digest
    return result


def evaluate(data_root, folder, output):
    result = verify_result(folder, data_root)
    data = make_dataset(
        data_root, result["model"], result["fold"], result["history"], "val"
    )
    model = load_checkpoint(
        result["model"], Path(folder) / "best.pt", result["checkpoint_sha256"]
    ).cuda()
    probability = predict(
        model, result["model"], data, result["training"]["batch_size"]
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = save_predictions(data, probability, output)
    write_json(output.with_suffix(".json"), scores)
    return scores
