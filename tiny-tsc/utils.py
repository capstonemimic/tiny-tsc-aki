"""Configuration, fingerprints, and atomic artifact writes."""

from pathlib import Path
import hashlib
import json
import random

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_config(name, path=None):
    settings = load_config(path or ROOT / "configs/training.yaml")
    config = {**settings["defaults"], **settings.get("models", {}).get(name, {})}
    for key in ("max_epochs", "batch_size", "accumulate"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["scheduler"] not in ("none", "cosine"):
        raise ValueError("Scheduler must be none or cosine")
    if config["precision"] not in ("float32", "bf16-mixed"):
        raise ValueError("Precision must be float32 or bf16-mixed")
    if config["early_stopping"] and config["patience"] < 1:
        raise ValueError("Early stopping requires a positive patience")
    return config


def fingerprint(data_root, name, fold, history, training):
    paths = [
        data_root / "window_membership.parquet",
        *[data_root / f"folds/fold_{fold}/{p}.safetensors" for p in ("train", "val")],
    ]
    from models import IRREGULAR

    if name in IRREGULAR:
        paths.append(data_root / "concat_inputs/manifest.json")
        paths.extend(
            data_root / f"concat_inputs/{field}.npy"
            for field in ("values", "mask", "times", "offsets")
        )
    return {
        "protocol": "aki24-every-window-v1",
        "model": name,
        "fold": fold,
        "history": history,
        "seed": 4242 + fold,
        "training": training,
        "sources": {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted(
                [
                    *ROOT.glob("*.py"),
                    *ROOT.glob("models/*.py"),
                    *ROOT.glob("configs/*.yaml"),
                ]
            )
        },
        "data": {str(p.relative_to(data_root)): sha(p) for p in paths},
    }
