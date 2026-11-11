"""CLI for 24h AKI training, evaluation, and analysis."""

from pathlib import Path
import json
import typer

app = typer.Typer(add_completion=False)


@app.command("models")
def list_models():
    """List available models."""
    from models import ALL_MODELS

    typer.echo("\n".join(ALL_MODELS))


@app.command()
def preprocess(data_root: Path = typer.Option(..., help="Prepared 24h fold snapshot")):
    """Pack all raw observations belonging to the prepared 24h windows."""
    from preprocess import prepare_irregular

    typer.echo(json.dumps(prepare_irregular(data_root.resolve()), indent=2))


@app.command()
def train(
    model: str,
    data_root: Path = typer.Option(...),
    output_root: Path = Path("outputs"),
    fold: int = typer.Option(0, min=0, max=4),
    history: str = "window",
    training_config: Path | None = None,
    compiled: bool = False,
):
    """Train one fold; history is window or causal prefix."""
    from train import train_model

    result = train_model(
        data_root, output_root, model, fold, history, training_config, compiled
    )
    typer.echo(json.dumps(result["metrics"], indent=2))


@app.command()
def reproduce(
    data_root: Path = typer.Option(...),
    output_root: Path = Path("outputs"),
    models: list[str] | None = typer.Option(None, "--model"),
):
    """Train and profile window models on folds 0–4; resumable."""
    from models import WINDOW_MODELS
    from profiling import measure
    from train import train_model
    from utils import write_json

    for name in models or WINDOW_MODELS:
        for fold in range(5):
            result = train_model(data_root, output_root, name, fold)
            folder = output_root / name / "window" / f"fold_{fold}"
            write_json(
                folder / "profile.json",
                measure(
                    name, folder / "best.pt", expected_hash=result["checkpoint_sha256"]
                ),
            )
            typer.echo(f"{name} fold {fold}: {json.dumps(result['metrics'])}")


@app.command()
def inference(
    data_root: Path = typer.Option(...),
    run: Path = typer.Option(...),
    output: Path = Path("outputs/validation_predictions.parquet"),
):
    """Replay a verified checkpoint on its complete validation fold."""
    from evaluation import evaluate

    typer.echo(json.dumps(evaluate(data_root, run, output), indent=2))


@app.command()
def profile(
    model: str,
    checkpoint: Path = typer.Option(...),
    output: Path = Path("outputs/profile.json"),
    timestamps: int = typer.Option(128, min=2),
    windows: int = typer.Option(1, min=1),
):
    """Count parameters and inference FLOPs from the selected checkpoint, batch 32."""
    from profiling import measure
    from utils import write_json

    result = measure(model, checkpoint, timestamps, windows)
    write_json(output, result)
    typer.echo(
        f"Parameters: {result['parameters_trainable']:,}; GFLOPs/batch32: {result['flops_batch32'] / 1e9:.4f}"
    )


@app.command()
def sanity(
    data_root: Path = typer.Option(...),
    model: str | None = None,
    history: str = "window",
):
    """Verify all five patient folds and optionally check model inputs/gradients."""
    from sanity import check_folds, check_model

    result = check_folds(data_root)
    if model:
        result["model"] = check_model(data_root, model, history)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def svd(
    data_root: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    output: Path = Path("outputs/svd"),
    fold: int = typer.Option(0, min=0, max=4),
    history: str = "window",
):
    """Analyze Perceiver weight spectra and compare 90/95/99% energy factorizations."""
    from analysis import analyze

    result = analyze(data_root, checkpoint, output, fold, history)
    typer.echo(json.dumps(result["comparisons"], indent=2))


@app.command()
def report(
    data_root: Path = typer.Option(...),
    output_root: Path = Path("outputs"),
    output: Path = Path("outputs/results.md"),
):
    """Build the five-fold scores, compute, and steps-to-checkpoint table."""
    from reporting import summarize

    summarize(data_root, output_root, output)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
