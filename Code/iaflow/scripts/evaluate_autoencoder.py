"""
Evaluate a selected autoencoder checkpoint on one stored split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import Dataset, Subset

from iaflow.artifacts import (
    load_compatible_autoencoder_checkpoint,
    portable_path,
    save_json,
)
from iaflow.config import ExperimentConfig, load_experiment_config
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--output", type=Path)
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
        return
    
    if not arguments.confirm_final_test:
        raise ValueError("Test evaluation requires --confirm-final-test.")
    
    if arguments.maximum_samples is not None:
        raise ValueError("Final test evaluation must use the complete stored test split.")


def _resolve_evaluation_output(
    config: ExperimentConfig,
    checkpoint_path: Path,
    split: str,
    explicit_output: Path | None,
) -> Path:
    """
    Resolve the default or explicitly requested evaluation result path.
    
    Arguments:
        config (ExperimentConfig):
            Checked experiment configuration.
        checkpoint_path (pathlib.Path):
            Resolved checkpoint evaluated by the command.
        split (str):
            Stored data split being evaluated.
        explicit_output (pathlib.Path or None):
            Optional command-line result destination.
    
    Returns:
        output (pathlib.Path):
            Resolved metrics JSON destination.
    """
    if explicit_output is None:
        return checkpoint_path.parent / f"{split.capitalize()}Metrics.json"
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
    config = load_experiment_config(arguments.config)
    _check_evaluation_request(arguments)
    device = resolve_device(arguments.device)
    checkpoint_path = config.resolve_path(arguments.checkpoint)
    output = _resolve_evaluation_output(
        config,
        checkpoint_path,
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
        "split": arguments.split,
        "final_test": arguments.split == "test",
        "device": str(device),
        "metrics": metrics,
    }
    save_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
