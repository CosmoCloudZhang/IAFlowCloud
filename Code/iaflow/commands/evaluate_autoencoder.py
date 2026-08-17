"""
Evaluate a selected autoencoder checkpoint on one stored split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from torch.utils.data import Dataset, Subset

from iaflow.artifacts import (
    load_compatible_autoencoder_checkpoint,
    portable_path,
    save_json,
)
from iaflow.config import ExperimentConfig, load_resolved_experiment_config
from iaflow.data import (
    CachedSurfaceDataset,
    build_dataloader,
)
from iaflow.evaluation import evaluate_autoencoder
from iaflow.training import resolve_device


def parse_arguments() -> argparse.Namespace:
    """
    Parse evaluation, device, split, and final-test safety arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Run containing ResolvedConfig.json and Best.pt.",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pca-metrics",
        type=Path,
        help="Optional full-validation PCA metrics for a matched-rank comparison.",
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Confirm that model selection is frozen before reading the complete test split.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _check_evaluation_request(
    arguments: argparse.Namespace,
) -> None:
    """
    Enforce the one-time complete-test evaluation policy.
    
    Arguments:
        arguments (argparse.Namespace):
            Parsed split, sample-limit, and final-test confirmation arguments.
    """
    if arguments.split != "test":
        if arguments.pca_metrics is not None and (
            arguments.split != "validation" or arguments.maximum_samples is not None
        ):
            raise ValueError(
                "PCA comparison requires the complete validation split."
            )
        return
    
    if arguments.pca_metrics is not None:
        raise ValueError("PCA comparison is available only for validation evaluation.")
    
    if not arguments.confirm_final_test:
        raise ValueError("Test evaluation requires --confirm-final-test.")
    
    if arguments.maximum_samples is not None:
        raise ValueError("Final test evaluation must use the complete stored test split.")


def _matched_pca_comparison(
    config: ExperimentConfig,
    path: Path,
    latent_dim: int,
    metrics: dict[str, float],
    normalization_scale: float,
) -> dict[str, object]:
    """
    Compare complete validation metrics with PCA at the same dimension.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        path (pathlib.Path):
            PCA validation-metrics artifact to load.
        latent_dim (int):
            Autoencoder latent dimension selecting the PCA rank.
        metrics (dict[str, float]):
            Complete autoencoder validation metrics.
        normalization_scale (float):
            Normalization scale loaded with the autoencoder checkpoint.
    
    Returns:
        comparison (dict[str, object]):
            Matched PCA metrics, metric differences, and outcome flag.
    """
    reference_path = config.resolve_path(path)
    with reference_path.open("r", encoding="utf-8") as stream:
        reference = json.load(stream)
    expected_metadata = {
        "split": "validation",
        "source_dataset": config.data.source_path,
        "target_dataset": config.data.target_dataset,
        "transform": config.data.transform,
        "normalization": config.data.normalization,
    }
    for name, expected in expected_metadata.items():
        if reference.get(name) != expected:
            raise ValueError(
                f"PCA metrics {name!r} does not match the experiment configuration."
            )
    pca_scale = float(reference.get("normalization_scale", float("nan")))
    if not math.isclose(
        pca_scale,
        normalization_scale,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "PCA metrics normalization scale does not match the autoencoder "
            "checkpoint."
        )
    ranks = reference.get("ranks")
    if not isinstance(ranks, dict) or str(latent_dim) not in ranks:
        raise ValueError(f"PCA metrics do not contain rank {latent_dim}.")
    pca_metrics = ranks[str(latent_dim)]
    if not isinstance(pca_metrics, dict):
        raise ValueError(f"PCA rank {latent_dim} metrics must be a mapping.")
    if int(pca_metrics.get("number_of_surfaces", -1)) != int(
        metrics["number_of_surfaces"]
    ):
        raise ValueError("PCA and autoencoder validation sample counts differ.")
    compared_names = (
        "normalized_mse",
        "log10_mse",
        "log10_rmse",
        "log10_mae",
        "variance_recovered",
        "mean_relative_error",
        "surface_relative_rmse_p95",
        "surface_relative_rmse_p99",
        "surface_relative_maximum_p95",
        "surface_relative_maximum_p99",
    )
    missing = [name for name in compared_names if name not in pca_metrics]
    if missing:
        raise ValueError(f"PCA metrics are missing required values: {missing}.")
    differences = {
        name: float(metrics[name]) - float(pca_metrics[name])
        for name in compared_names
    }
    return {
        "reference": portable_path(reference_path, config.project_root),
        "rank": latent_dim,
        "pca_metrics": pca_metrics,
        "autoencoder_minus_pca": differences,
        "autoencoder_outperforms_pca": bool(
            metrics["variance_recovered"] > pca_metrics["variance_recovered"]
            and metrics["log10_mse"] < pca_metrics["log10_mse"]
        ),
    }


def _resolve_evaluation_output(
    config: ExperimentConfig,
    run_directory: Path,
    split: str,
    explicit_output: Path | None,
) -> Path:
    """
    Resolve the default or explicitly requested evaluation result path.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        run_directory (pathlib.Path):
            Resolved run directory receiving default metrics.
        split (str):
            Stored data split being evaluated.
        explicit_output (pathlib.Path or None):
            Optional command-line result destination.
    
    Returns:
        output (pathlib.Path):
            Resolved metrics JSON destination.
    """
    if explicit_output is None:
        return run_directory / f"{split.capitalize()}Metrics.json"
    return config.resolve_path(explicit_output)


def _evaluation_dataset(
    config: ExperimentConfig,
    split: str,
    maximum_samples: int | None,
) -> Dataset:
    """
    Build one complete or deterministically truncated evaluation dataset.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        split (str):
            Stored split to evaluate.
        maximum_samples (int or None):
            Optional leading-sample limit for non-final evaluations.
    
    Returns:
        dataset (torch.utils.data.Dataset):
            Complete stored split or its leading deterministic subset.
    """
    dataset = CachedSurfaceDataset(config, split)
    if maximum_samples is None:
        return dataset
    
    if maximum_samples <= 0:
        raise ValueError("--maximum-samples must be positive.")
    return Subset(dataset, range(min(maximum_samples, len(dataset))))


def main() -> None:
    """
    Load a compatible checkpoint, evaluate one split, and save its metrics.
    """
    arguments = parse_arguments()
    run_directory = arguments.run_directory.expanduser().resolve()
    config = load_resolved_experiment_config(run_directory)
    _check_evaluation_request(arguments)
    device = resolve_device(arguments.device)
    checkpoint_path = run_directory / "Best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
    output = _resolve_evaluation_output(
        config,
        run_directory,
        arguments.split,
        arguments.output,
    )
    if arguments.split == "test" and output.exists():
        raise FileExistsError(
            f"Final test result already exists and will not be replaced: {output}"
        )
    model, normalization, checkpoint, _ = load_compatible_autoencoder_checkpoint(
        checkpoint_path,
        config,
        device=device,
    )
    
    dataset = _evaluation_dataset(
        config,
        arguments.split,
        arguments.maximum_samples,
    )
    loader = build_dataloader(
        dataset,
        config.data,
        training=False,
        seed=config.training.seed,
        device=device,
    )
    metrics = evaluate_autoencoder(
        model,
        loader,
        normalization,
        device,
        show_progress=not arguments.no_progress,
    )
    result = {
        "checkpoint": portable_path(checkpoint_path, config.project_root),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "latent_dim": model.config.latent_dim,
        "split": arguments.split,
        "final_test": arguments.split == "test",
        "device": str(device),
        "metrics": metrics,
    }
    if arguments.pca_metrics is not None:
        result["pca_comparison"] = _matched_pca_comparison(
            config,
            arguments.pca_metrics,
            model.config.latent_dim,
            metrics,
            normalization.scale,
        )
    save_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
