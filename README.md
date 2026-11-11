# tiny-tsc

Repo with code and configs for "Tiny Time Series Classifiers for Acute Kidney Injury" (ML4H 2026).

Run the commands below from `tiny-tsc/`.

```sh
cd tiny-tsc
uv sync --locked
uv run main.py models
```

Supply a prepared subject-grouped snapshot:

```text
data/
  window_membership.parquet
  folds/fold_0/ ... folds/fold_4/
    train.safetensors
    val.safetensors
  source/cohort_features_aki_24h_windowed.parquet
```

Membership rows retain source order and include `source_row`, `subject_id`,
`hadm_id`, `stay_id`, `window_index`, `target`, `aki_time`, and `validation_fold`.
Safetensors contain aligned identifiers and labels, 33 × 128 feature grids,
`feature_columns`, and training-fold normalization statistics in metadata.
SQL extraction scripts are in `sql/`; prepared folds and membership are required inputs.

```sh
uv run main.py preprocess --data-root data
uv run main.py sanity --data-root data
uv run main.py train mwdn_micro_diet --data-root data --fold 0
uv run main.py reproduce --data-root data
```

`configs/training.yaml` and `configs/models.yaml` hold the settings. Model execution
requires CUDA. `--training-config PATH` selects alternate
settings for `train`. `reproduce` trains and profiles all window models on folds 0–4;
repeat `--model NAME` to select models. Completed fits are verified and reused.

Runs are saved under `outputs/MODEL/window/fold_N/`. Interrupted sequence
fits resume from completed epochs. Partial regular-grid fits need a new `--output-root`.

```sh
uv run main.py inference --data-root data \
  --run outputs/mwdn_micro_diet/window/fold_0 --output outputs/replay.parquet
uv run main.py profile mwdn_micro_diet \
  --checkpoint outputs/mwdn_micro_diet/window/fold_0/best.pt
uv run main.py --help
```
