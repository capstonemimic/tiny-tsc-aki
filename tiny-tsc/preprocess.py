"""Pack all observations for the prepared subject-grouped 24h folds."""

import json
import numpy as np
from safetensors import safe_open
from dataset import stay_rows
from utils import sha


def prepare_irregular(root):
    """Pack all observed timestamps, without truncation or interpolation, once."""
    import pyarrow.parquet as pq

    dest = root / "concat_inputs"
    if (dest / "manifest.json").exists():
        saved = json.loads((dest / "manifest.json").read_text())
        assert saved["membership_sha256"] == sha(root / "window_membership.parquet")
        assert saved["source_sha256"] == sha(
            root / "source/cohort_features_aki_24h_windowed.parquet"
        )
        for name, digest in saved["files_sha256"].items():
            assert sha(dest / f"{name}.npy") == digest
        return saved
    dest.mkdir(exist_ok=True)
    source = root / "source/cohort_features_aki_24h_windowed.parquet"
    m, stays = stay_rows(root)
    assert m.source_row.is_unique and m.source_row.is_monotonic_increasing, (
        "Membership must retain source-row order"
    )
    with safe_open(root / "folds/fold_0/train.safetensors", framework="numpy") as f:
        features = f.metadata()["feature_columns"].split(",")
    wanted = set(m.source_row.to_numpy())
    # Group duplicate feature/timestamp observations by their mean, matching the archived interpolation preprocessing.
    offsets, all_values, all_masks, all_times = [0], [], [], []
    source_row = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512, columns=features):
        for row in batch.to_pylist():
            index = source_row
            source_row += 1
            if index not in wanted:
                continue
            channels = []
            for feat in features:
                values = {}
                for event in row[feat] or []:
                    if event["v"] is None or np.isnan(event["v"]):
                        continue
                    assert np.isfinite(event["v"]) and 0 <= event["t"] < 24
                    values.setdefault(float(event["t"]), []).append(float(event["v"]))
                channels.append({t: float(np.mean(v)) for t, v in values.items()})
            times = sorted({t for values in channels for t in values})
            positions = {t: i for i, t in enumerate(times)}
            v = np.zeros((len(times), len(features)), np.float32)
            mask = np.zeros_like(v)
            for j, values in enumerate(channels):
                for t, value in values.items():
                    v[positions[t], j], mask[positions[t], j] = value, 1
            all_values.append(v)
            all_masks.append(mask)
            all_times.append(np.array(times, np.float32))
            offsets.append(offsets[-1] + len(times))
        if len(offsets) % 20000 < 512:
            print(f"Prepared {len(offsets) - 1}/{len(m)} irregular windows", flush=True)
    assert len(offsets) == len(m) + 1
    arrays = {
        "values": np.concatenate(all_values),
        "mask": np.concatenate(all_masks),
        "times": np.concatenate(all_times),
        "offsets": np.asarray(offsets, np.int64),
    }
    for name, a in arrays.items():
        np.save(dest / f"{name}.npy", a)
    result = {
        "windows": len(m),
        "stays": len(stays),
        "positive_stays": int(stays.target.sum()),
        "features": features,
        "timestamps": offsets[-1],
        "truncated_windows": 0,
        "source_sha256": sha(source),
        "membership_sha256": sha(root / "window_membership.parquet"),
        "files_sha256": {name: sha(dest / f"{name}.npy") for name in arrays},
    }
    (dest / "manifest.json").write_text(json.dumps(result, indent=2))
    return result
