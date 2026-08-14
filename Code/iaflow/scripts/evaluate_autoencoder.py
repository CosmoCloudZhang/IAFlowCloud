"""
Evaluate a selected autoencoder checkpoint on one stored split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import Subset

from iaflow.artifacts import (
    load_compatible_autoencoder_checkpoint,
    portable_path,
    save_json,
)
from iaflow.config import load_experiment_config
from iaflow.data import (
    CachedSurfaceDataset,
    build_dataloader,
)
from iaflow.evaluation import evaluate_autoencoder
from iaflow.training import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "Config" / "NLA" / "AutoEncoderConv1D.yml"


def parse_arguments() -> argparse.Namespace:
    """
    Parse evaluation, device, split, and final-test safety arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def main() -> None:
    """
    Load a compatible checkpoint, evaluate one split, and save its metrics.
    """
    arguments = parse_arguments()
    config = load_experiment_config(arguments.config, project_root=PROJECT_ROOT)
    if arguments.split == "test":
        if not arguments.confirm_final_test:
            raise ValueError("Test evaluation requires --confirm-final-test.")
        
        if arguments.maximum_samples is not None:
            raise ValueError("Final test evaluation must use the complete stored test split.")
    device = resolve_device(arguments.device)
    checkpoint_path = arguments.checkpoint.expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = config.project_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    output = arguments.output
    if output is None:
        output = checkpoint_path.parent / f"{arguments.split.capitalize()}Metrics.json"
    elif not output.is_absolute():
        output = config.project_root / output
    output = output.expanduser().resolve()
    if arguments.split == "test" and output.exists():
        raise FileExistsError(
            f"Final test result already exists and will not be replaced: {output}"
        )
    model, normalization, checkpoint, _ = load_compatible_autoencoder_checkpoint(
        checkpoint_path,
        config,
        device=device,
    )
    
    dataset = CachedSurfaceDataset(config, arguments.split)
    if arguments.maximum_samples is not None:
        if arguments.maximum_samples <= 0:
            raise ValueError("--maximum-samples must be positive.")
        dataset = Subset(dataset, range(min(arguments.maximum_samples, len(dataset))))
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
