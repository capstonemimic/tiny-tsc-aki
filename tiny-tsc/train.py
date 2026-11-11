"""Shared 24h training entry point for grid and irregular/history models."""

from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import make_dataset, collate_stays, move_batch
from evaluation import (
    metrics,
    predict,
    save_predictions,
    validation_metrics,
    verify_result,
)
from models import REGULAR, make_model, loss
from utils import (
    atomic_save,
    fingerprint,
    seed_everything,
    sha,
    training_config,
    write_json,
)


def train_regular(model, datasets, settings, folder):
    import tsai.all as tsai
    from fastai.callback.tracker import SaveModelCallback
    from fastai.callback.tensorboard import TensorBoardCallback
    from fastai.callback.progress import CSVLogger

    arrays = [datasets[part].features for part in ("train", "val")]
    labels = [datasets[part].labels for part in ("train", "val")]
    x, y, splits = tsai.combine_split_data(arrays, labels)
    combined = tsai.TSDatasets(x, y, splits=splits, device=torch.device("cuda"))
    loaders = tsai.TSDataLoaders.from_dsets(
        combined.train,
        combined.valid,
        bs=settings["batch_size"],
        drop_last=True,
        num_workers=0,
        device=torch.device("cuda"),
    )
    if len(loaders.train) == 0:
        raise ValueError("Training split is smaller than the configured batch size")
    if settings["accumulate"] != 1:
        raise ValueError(
            "Regular-grid training uses one physical batch per optimizer step"
        )
    if settings["early_stopping"] or settings["scheduler"] != "none":
        raise ValueError(
            "Regular-grid training uses a fixed epoch count and no LR scheduler"
        )
    callbacks = [
        SaveModelCallback(
            monitor="roc_auc_multiclass",
            comp=np.greater,
            fname="selected",
            with_opt=False,
        ),
        CSVLogger(fname="epochs.csv"),
        TensorBoardCallback(
            log_dir=str(folder / "tensorboard"),
            log_preds=False,
            projector=False,
            trace_model=False,
        ),
    ]
    learner = tsai.Learner(
        loaders,
        model,
        loss_func=tsai.CrossEntropyLossFlat(),
        metrics=[tsai.accuracy, *validation_metrics()],
        cbs=callbacks,
        path=folder,
    )
    if settings["precision"] == "bf16-mixed":
        learner = learner.to_bf16()
    learner.fit(
        settings["max_epochs"],
        lr=settings["learning_rate"],
        wd=settings["weight_decay"],
    )
    history = [
        dict(
            epoch=i + 1,
            loss=float(row[0]),
            auroc=float(row[3]),
            auprc=float(row[4]),
            optimizer_steps=(i + 1) * len(loaders.train),
        )
        for i, row in enumerate(learner.recorder.values)
    ]
    learner.load("selected", with_opt=False)
    probability, labels = learner.get_preds(ds_idx=1)
    np.testing.assert_array_equal(labels.cpu().numpy(), datasets["val"].labels)
    best_epoch = int(np.argmax([row["auroc"] for row in history])) + 1
    atomic_save({"model": model.state_dict(), "epoch": best_epoch}, folder / "best.pt")
    write_json(folder / "history.json", history)
    return probability.float().cpu().numpy()[:, 1], best_epoch, history


def train_sequence(model, name, datasets, settings, folder, identity):
    generator = torch.Generator().manual_seed(identity["seed"])
    loaders = {
        part: DataLoader(
            data,
            batch_size=settings["batch_size"],
            shuffle=part == "train",
            collate_fn=collate_stays,
            generator=generator,
            num_workers=0,
        )
        for part, data in datasets.items()
    }
    exempt = lambda n: name != "ode_rnn" and any(k in n for k in ("bias", "norm"))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if not exempt(n)],
                "weight_decay": settings["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if exempt(n)],
                "weight_decay": 0.0,
            },
        ],
        lr=settings["learning_rate"],
    )
    if name == "mtan_vae":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, settings["max_epochs"], eta_min=settings["learning_rate"] * 0.01
        )
        if settings["scheduler"] == "cosine"
        else None
    )
    history, best, stale, updates = [], -float("inf"), 0, 0
    resume = folder / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError(
                "Resume checkpoint was produced with different data, code, or settings"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler:
            scheduler.load_state_dict(saved["scheduler"])
        history, best, stale, updates = (
            saved["history"],
            saved["best"],
            saved["stale"],
            saved["updates"],
        )
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        generator.set_state(saved["loader_rng"])
    for epoch in range(len(history), settings["max_epochs"]):
        if settings["early_stopping"] and stale >= settings["patience"]:
            break
        start = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, seen = 0.0, 0
        for step, raw in enumerate(loaders["train"]):
            batch = move_batch(raw)
            objective = loss(model, name, batch)
            if not torch.isfinite(objective):
                raise ValueError(f"Non-finite loss at epoch {epoch + 1}, batch {step}")
            group_start = (
                step
                // settings["accumulate"]
                * settings["accumulate"]
                * settings["batch_size"]
            )
            denominator = min(
                settings["accumulate"] * settings["batch_size"],
                len(datasets["train"]) - group_start,
            )
            (objective * len(batch["labels"]) / denominator).backward()
            if (step + 1) % settings["accumulate"] == 0 or step + 1 == len(
                loaders["train"]
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            total += float(objective.detach()) * len(batch["labels"])
            seen += len(batch["labels"])
        assert seen == len(datasets["train"])
        probability = predict(
            model, name, datasets["val"], settings["batch_size"]
        )
        scores = metrics(datasets["val"].labels, probability)
        history.append(
            dict(
                epoch=epoch + 1,
                loss=total / seen,
                auroc=scores["auroc"],
                auprc=scores["auprc"],
                train_windows_seen=seen,
                optimizer_steps=updates,
                seconds=time.monotonic() - start,
            )
        )
        print(json.dumps(history[-1]), flush=True)
        if scores["auroc"] > best:
            best, stale = scores["auroc"], 0
            atomic_save(
                {"model": model.state_dict(), "epoch": epoch + 1}, folder / "best.pt"
            )
        else:
            stale += 1
        if scheduler:
            scheduler.step()
        write_json(folder / "history.json", history)
        atomic_save(
            dict(
                identity=identity,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler else None,
                history=history,
                best=best,
                stale=stale,
                updates=updates,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                loader_rng=generator.get_state(),
            ),
            resume,
        )
    selected = torch.load(folder / "best.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(selected["model"])
    probability = predict(model, name, datasets["val"], settings["batch_size"])
    assert abs(metrics(datasets["val"].labels, probability)["auroc"] - best) < 1e-10
    return probability, selected["epoch"], history


def train_model(
    data_root,
    output_root,
    name,
    fold=0,
    history="window",
    config_path=None,
    compiled=False,
):
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    if fold not in range(5):
        raise ValueError("Fold must be between 0 and 4")
    settings = training_config(name, config_path)
    settings["device"] = "cuda"
    settings["compiled"] = compiled
    identity = fingerprint(data_root, name, fold, history, settings)
    folder = output_root / name / history / f"fold_{fold}"
    if (folder / "result.json").exists():
        saved = verify_result(folder, data_root)
        if saved["identity"] != identity:
            raise ValueError(
                "Existing result uses different settings or sources; use another output directory"
            )
        return saved
    if name in REGULAR and folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Preserve the partial regular-model run before retrying")
    folder.mkdir(parents=True, exist_ok=True)
    seed_everything(4242 + fold)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    model = make_model(name, 4242 + fold, compiled=compiled).cuda()
    datasets = {
        part: make_dataset(data_root, name, fold, history, part)
        for part in ("train", "val")
    }
    if name == "minirocket":
        model.fit(
            torch.from_numpy(datasets["train"].features).cuda(), chunksize=128
        )
    if not (folder / "initial.pt").exists():
        atomic_save(model.state_dict(), folder / "initial.pt")
    if name in REGULAR:
        probability, best_epoch, rows = train_regular(
            model, datasets, settings, folder
        )
    else:
        probability, best_epoch, rows = train_sequence(
            model, name, datasets, settings, folder, identity
        )
    scores = save_predictions(
        datasets["val"], probability, folder / "val_predictions.parquet"
    )
    selected = rows[best_epoch - 1]
    for metric in ("auroc", "auprc"):
        assert abs(scores[metric] - selected[metric]) < 2e-5
    result = dict(
        model=name,
        fold=fold,
        history=history,
        identity=identity,
        training=settings,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        best_epoch=best_epoch,
        epochs=len(rows),
        metrics=scores,
        train_examples=len(datasets["train"]),
        val_examples=len(datasets["val"]),
        optimizer_steps_to_best=selected["optimizer_steps"],
        checkpoint_sha256=sha(folder / "best.pt"),
        prediction_sha256=sha(folder / "val_predictions.parquet"),
        evaluation="Every retained validation window; validation also selects the checkpoint",
    )
    write_json(folder / "result.json", result)
    (folder / "resume.pt").unlink(missing_ok=True)
    return verify_result(folder, data_root)
