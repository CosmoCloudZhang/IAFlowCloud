"""
Evaluate a selected autoencoder checkpoint on one stored split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import Dataset, Subset

from ...comparison import matched_pca_comparison
from ...core.artifacts import portable_path, save_json
from ...core.data import CachedSurfaceDataset, build_dataloader
from ...core.evaluation import evaluate_autoencoder
from ...core.runtime import resolve_device
from ..artifacts import load_compatible_autoencoder_checkpoint
from ..config import (
    ExperimentConfig,
    load_resolved_experiment_config,
)


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
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
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
        result["pca_comparison"] = matched_pca_comparison(
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
