"""Verified 24h folds and causal within-stay histories."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import ROOT, load_config

DATA_CONFIG = load_config(ROOT / "configs/data.yaml")
WINDOW_HOURS = DATA_CONFIG["window_hours"]
CHANNELS = DATA_CONFIG["channels"]
GRID_POINTS = DATA_CONFIG["grid_points"]
N_FOLDS = DATA_CONFIG["n_folds"]


def stay_rows(root):
    m = pd.read_parquet(root / "window_membership.parquet").reset_index(drop=True)
    m["position"] = np.arange(len(m))
    assert m.validation_fold.isin(range(N_FOLDS)).all()
    assert m.groupby(DATA_CONFIG["group_column"]).validation_fold.nunique().eq(1).all()
    assert not m.duplicated(["stay_id", "window_index"]).any()
    ordered = m.sort_values(["stay_id", "window_index"])
    groups = ordered.groupby("stay_id", sort=True)
    for key in ("subject_id", "hadm_id", "validation_fold"):
        assert groups[key].nunique().eq(1).all(), key
    final = groups.tail(1).set_index("stay_id").sort_index()
    final["positions"] = groups.position.agg(list)
    final["window_indices"] = groups.window_index.agg(list)
    final["n_windows"] = groups.size()
    # Supplied query keeps no observation window at/after first AKI onset.
    cutoff = (m.window_index + 1) * WINDOW_HOURS
    assert (m.aki_time.isna() | (m.aki_time >= cutoff)).all()
    np.testing.assert_array_equal(
        m.target, (m.aki_time >= cutoff) & (m.aki_time < cutoff + WINDOW_HOURS)
    )
    # In this dataset, an existing positive label can occur only at the end.
    np.testing.assert_array_equal(groups.target.max(), final.target)
    final = final.reset_index()
    final["sample_id"] = final.stay_id.astype(str)
    assert final.n_windows.sum() == len(m)
    return m, final


def window_rows(root):
    """Keep every original target and attach only earlier/current windows of its stay."""
    m, _ = stay_rows(
        root
    )  # also validates the supplied onset labels and subject grouping
    rows = m.sort_values(["stay_id", "window_index"]).copy()
    positions, indices = [], []
    for _, group in rows.groupby("stay_id", sort=False):
        p = group.position.tolist()
        w = group.window_index.tolist()
        for i in range(len(group)):
            positions.append(p[: i + 1])
            indices.append(w[: i + 1])
    rows["positions"], rows["window_indices"] = positions, indices
    rows["n_windows"] = [len(p) for p in positions]
    rows["sample_id"] = (
        rows.stay_id.astype(str) + ":" + rows.window_index.astype(int).astype(str)
    )
    assert rows.sample_id.is_unique and len(rows) == len(m)
    assert sorted(rows.position.tolist()) == list(range(len(m)))
    assert all(p[-1] == end for p, end in zip(positions, rows.position))
    assert all(
        w[-1] == end and all(a < b for a, b in zip(w, w[1:]))
        for w, end in zip(indices, rows.window_index)
    )
    return m, rows.reset_index(drop=True)


def selected_windows(root, fold):
    if fold not in range(N_FOLDS):
        raise ValueError(f"Fold must be between 0 and {N_FOLDS - 1}")
    m, windows = window_rows(root)
    parts = {
        part: windows.loc[
            windows.validation_fold.eq(fold)
            if part == "val"
            else windows.validation_fold.ne(fold)
        ].reset_index(drop=True)
        for part in ("train", "val")
    }
    for key in ("subject_id", "hadm_id", "stay_id"):
        assert set(parts["train"][key]).isdisjoint(parts["val"][key]), key
    return m, parts


class WindowDataset(Dataset):
    def __init__(self, root, fold, history, part, model):
        self.history, self.model = history, model
        if history not in ("window", "prefix") or part not in ("train", "val"):
            raise ValueError("Choose window/prefix history and train/val split")
        m, parts = selected_windows(root, fold)
        self.ids = parts[part]
        self.labels = self.ids.target.to_numpy(dtype=np.int64)
        self.indices = self.ids.positions.tolist()
        self.days = self.ids.window_indices.tolist()
        fold_path = root / f"folds/fold_{fold}/{part}.safetensors"
        if model == "mwdn_micro_diet":
            original = m.loc[
                m.validation_fold.eq(fold)
                if part == "val"
                else m.validation_fold.ne(fold)
            ]
            self.lookup = np.full(len(m), -1, dtype=np.int64)
            self.lookup[original.position] = np.arange(len(original))
            with safe_open(fold_path, framework="numpy") as f:
                for key in ("subject_id", "hadm_id", "stay_id"):
                    np.testing.assert_array_equal(f.get_tensor(key), original[key])
                np.testing.assert_array_equal(
                    f.get_tensor("labels"), original.target.astype(int)
                )
                self.regular = f.get_tensor("features")
            assert self.regular.shape[1:] == (CHANNELS, GRID_POINTS)
        else:
            dest = root / "concat_inputs"
            self.arrays = {
                k: np.load(dest / f"{k}.npy", mmap_mode="r")
                for k in ("values", "mask", "times", "offsets")
            }
            features = json.loads((dest / "manifest.json").read_text())["features"]
            with safe_open(
                root / f"folds/fold_{fold}/train.safetensors", framework="numpy"
            ) as f:
                meta = f.metadata()
            self.mean = np.array(
                [float(meta[f"stat_{f}_mu"]) for f in features], np.float32
            )
            self.std = np.array(
                [float(meta[f"stat_{f}_sigma"]) or 1 for f in features], np.float32
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        members, days = self.indices[index], self.days[index]
        if self.history == "window":
            members, days = members[-1:], days[-1:]
        # The final prediction cutoff is time zero; preserve gaps in 24h units.
        days = np.asarray(days, np.float32)
        relative_starts = days - days[-1] - 1
        if self.model == "mwdn_micro_diet":
            indices = self.lookup[members]
            assert (indices >= 0).all()
            return (
                torch.from_numpy(self.regular[indices]),
                torch.from_numpy(relative_starts + 1),
                int(self.labels[index]),
            )
        inputs, times = [], []
        for member, shift in zip(members, relative_starts):
            start, end = self.arrays["offsets"][member : member + 2]
            mask = self.arrays["mask"][start:end]
            values = (
                np.clip(
                    (self.arrays["values"][start:end] - self.mean) / self.std, -4, 4
                )
                * mask
            )
            inputs.append(np.concatenate((values, mask), axis=1))
            times.append(self.arrays["times"][start:end] / WINDOW_HOURS + shift)
        return (
            torch.from_numpy(np.concatenate(inputs)),
            torch.from_numpy(np.concatenate(times)),
            int(self.labels[index]),
        )


def collate_stays(batch):
    xs, times, labels = zip(*batch)
    regular = xs[0].ndim == 3
    x = torch.cat(xs) if regular else pad_sequence(xs, batch_first=True)
    t = pad_sequence(times, batch_first=True)
    for i, item in enumerate(times):
        t[i, len(item) :] = item[-1]
    return {
        "x": x,
        "times": t,
        "lengths": torch.tensor([len(x) for x in xs]),
        "labels": torch.tensor(labels),
    }


class GridDataset(Dataset):
    """Individual normalized windows in the original safetensor row order."""

    def __init__(self, root, fold, part):
        membership, _ = selected_windows(root, fold)
        self.ids = membership.loc[
            membership.validation_fold.eq(fold)
            if part == "val"
            else membership.validation_fold.ne(fold)
        ].copy()
        self.ids["sample_id"] = (
            self.ids.stay_id.astype(str)
            + ":"
            + self.ids.window_index.astype(int).astype(str)
        )
        with safe_open(
            root / f"folds/fold_{fold}/{part}.safetensors", framework="numpy"
        ) as file:
            for key in ("subject_id", "hadm_id", "stay_id"):
                np.testing.assert_array_equal(file.get_tensor(key), self.ids[key])
            self.labels = file.get_tensor("labels").astype(np.int64)
            np.testing.assert_array_equal(self.labels, self.ids.target)
            self.features = file.get_tensor("features")
        assert self.features.shape == (len(self.ids), CHANNELS, GRID_POINTS)
        assert np.isfinite(self.features).all()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index]), torch.tensor(self.labels[index])


class IrregularWindowDataset(WindowDataset):
    def __getitem__(self, index):
        x, time, label = super().__getitem__(index)
        return x, time + 1, label


def make_dataset(root, name, fold, history, part):
    from models import ALL_MODELS, REGULAR, HISTORY_MODELS

    root = Path(root)
    if (
        name not in ALL_MODELS
        or history not in ("window", "prefix")
        or part not in ("train", "val")
    ):
        raise ValueError("Invalid model, history, or split")
    if history == "prefix" and name not in HISTORY_MODELS:
        raise ValueError(
            "Causal prefixes are supported by Perceiver and Micro-Diet + GRU"
        )
    if name in REGULAR:
        return GridDataset(root, fold, part)
    cls = WindowDataset if name in HISTORY_MODELS else IrregularWindowDataset
    kind = "mwdn_micro_diet" if name == "micro_diet_gru" else "perceiver"
    return cls(root, fold, history, part, kind)


def move_batch(batch):
    return {
        key: value.cuda() if key != "lengths" else value
        for key, value in batch.items()
    }
