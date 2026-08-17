"""
Generate validation-tail maps and worst surfaces for one selected run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from iaflow.artifacts import load_compatible_autoencoder_checkpoint, save_json
from iaflow.config import load_resolved_experiment_config
from iaflow.data import CachedSurfaceDataset, build_dataloader
from iaflow.diagnostics import evaluate_tail_diagnostics
from iaflow.training import resolve_device


def parse_arguments() -> argparse.Namespace:
    """
    Parse run, device, output, and retained-tail arguments.
    
    Returns:
        arguments (argparse.Namespace):
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worst-count", type=int, default=12)
    parser.add_argument(
        "--maximum-samples",
        type=int,
        help="Optional leading validation subset for smoke checks only.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _save_npz(
    values: dict[str, object],
    path: Path,
) -> None:
    """
    Atomically save array-valued diagnostics.
    
    Arguments:
        values (dict[str, object]):
            Diagnostic arrays to store.
        path (pathlib.Path):
            Final compressed NPZ destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def main() -> None:
    """
    Evaluate the complete validation split and save reusable diagnostics.
    """
    arguments = parse_arguments()
    run_directory = arguments.run_directory.expanduser().resolve()
    config = load_resolved_experiment_config(run_directory)
    
    checkpoint_path = run_directory / "Best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
    
    device = resolve_device(arguments.device)
    model, normalization, _checkpoint, _metadata = (
        load_compatible_autoencoder_checkpoint(
            checkpoint_path,
            config,
            device=device,
        )
    )
    
    complete_dataset = CachedSurfaceDataset(config, "validation")
    if arguments.maximum_samples is not None and arguments.maximum_samples <= 0:
        raise ValueError("--maximum-samples must be positive.")
    
    sample_count = (
        len(complete_dataset)
        if arguments.maximum_samples is None
        else min(arguments.maximum_samples, len(complete_dataset))
    )
    dataset = Subset(complete_dataset, range(sample_count))
    
    loader = build_dataloader(
        dataset,
        config.data,
        training=False,
        seed=config.training.seed,
        device=device,
    )
    
    diagnostics = evaluate_tail_diagnostics(
        model,
        loader,
        normalization,
        device,
        source_indices=complete_dataset.source_indices[:sample_count],
        worst_count=arguments.worst_count,
        show_progress=not arguments.no_progress,
    )
    
    output = (
        config.resolve_path(arguments.output)
        if arguments.output is not None
        else run_directory
        / (
            "ValidationDiagnostics.npz"
            if arguments.maximum_samples is None
            else "ValidationDiagnosticsSmoke.npz"
        )
    )
    
    scalar_names = (
        "number_of_surfaces",
        "surface_relative_rmse_p95",
        "surface_relative_rmse_p99",
        "surface_relative_maximum_p95",
        "surface_relative_maximum_p99",
        "maximum_relative_error",
    )
    summary = {name: diagnostics.pop(name) for name in scalar_names}
    summary["complete_validation_split"] = arguments.maximum_samples is None
    
    _save_npz(diagnostics, output)
    summary["arrays"] = os.path.relpath(output, config.project_root)
    
    summary_path = run_directory / (
        "ValidationDiagnostics.json"
        if arguments.maximum_samples is None
        else "ValidationDiagnosticsSmoke.json"
    )
    save_json(summary, summary_path)
    
    print(f"Saved validation diagnostics: {output}")


if __name__ == "__main__":
    main()
