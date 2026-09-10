"""
Evaluate one selected PCA-AE checkpoint on a stored surface split.
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
from ..artifacts import (
    load_compatible_pca_autoencoder_checkpoint,
)
from ..config import load_resolved_pca_ae_config


def parse_arguments() -> argparse.Namespace:
    """
    Parse PCA-AE evaluation and final-test safety arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
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
    parser.add_argument("--pca-metrics", type=Path)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _check_request(
    arguments: argparse.Namespace,
) -> None:
    """
    Enforce the one-time complete-test policy for PCA-AE.
    
    Arguments:
        arguments (argparse.Namespace):
            Parsed evaluation arguments.
    """
    if arguments.maximum_samples is not None and arguments.maximum_samples <= 0:
        raise ValueError("--maximum-samples must be positive.")
    if arguments.split == "test":
        if not arguments.confirm_final_test:
            raise ValueError("PCA-AE test evaluation requires --confirm-final-test.")
        if arguments.maximum_samples is not None:
            raise ValueError("Final PCA-AE test evaluation must use the complete split.")
        if arguments.pca_metrics is not None:
            raise ValueError("PCA comparison is validation-only.")
    elif arguments.pca_metrics is not None and (
        arguments.split != "validation" or arguments.maximum_samples is not None
    ):
        raise ValueError("PCA comparison requires the complete validation split.")


def _dataset(
    config: object,
    split: str,
    maximum_samples: int | None,
) -> Dataset:
    """
    Build a complete or leading PCA-AE evaluation surface dataset.
    
    Arguments:
        config (PCAAEExperimentConfig):
            Checked resolved PCA-AE configuration.
        split (str):
            Stored split to evaluate.
        maximum_samples (int or None):
            Optional leading-sample smoke limit.
    
    Returns:
        dataset (torch.utils.data.Dataset):
            Complete or truncated surface dataset.
    """
    complete = CachedSurfaceDataset(config, split)
    if maximum_samples is None:
        return complete
    return Subset(complete, range(min(maximum_samples, len(complete))))


def main() -> None:
    """
    Load, evaluate, compare, and save one PCA-AE checkpoint result.
    """
    arguments = parse_arguments()
    _check_request(arguments)
    run_directory = arguments.run_directory.expanduser().resolve()
    config = load_resolved_pca_ae_config(run_directory)
    checkpoint_path = run_directory / "Best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PCA-AE checkpoint not found: {checkpoint_path}")
    output = (
        run_directory / f"{arguments.split.capitalize()}Metrics.json"
        if arguments.output is None
        else config.resolve_path(arguments.output)
    )
    if arguments.split == "test" and output.exists():
        raise FileExistsError(f"Final PCA-AE test result already exists: {output}")
    device = resolve_device(arguments.device)
    model, normalization, _coefficient_normalization, checkpoint, _metadata = (
        load_compatible_pca_autoencoder_checkpoint(
            checkpoint_path,
            config,
            device=device,
        )
    )
    dataset = _dataset(config, arguments.split, arguments.maximum_samples)
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
        "model_family": "PCA_AE",
        "pca_rank": int(config.model.pca_rank),
        "latent_dim": int(config.model.latent_dim),
        "split": arguments.split,
        "final_test": arguments.split == "test",
        "device": str(device),
        "metrics": metrics,
    }
    if arguments.pca_metrics is not None:
        result["pca_comparison"] = matched_pca_comparison(
            config,
            arguments.pca_metrics,
            config.model.latent_dim,
            metrics,
            normalization.scale,
        )
    save_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
