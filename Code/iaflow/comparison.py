"""
Run discovery and compact comparison tables for model selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["collect_ae_results", "smallest_qualified_model"]


def collect_ae_results(
    project_root: str | Path,
) -> list[dict[str, Any]]:
    """
    Collect completed AE run summaries with their resolved experiment axes.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
    
    Returns:
        results (list[dict[str, Any]]):
            Ordered architecture, depth, dense, latent, parameter, and metric records.
    """
    root = Path(project_root).expanduser().resolve()
    run_root = root / "Runs" / "NLA" / "AE"
    results: list[dict[str, Any]] = []
    if not run_root.is_dir():
        return results
    for summary_path in run_root.glob("*/*/Latent*/*/Summary.json"):
        run_directory = summary_path.parent
        config_path = run_directory / "ResolvedConfig.json"
        if not config_path.is_file():
            continue
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        metrics = summary.get("best_validation_metrics", {})
        results.append(
            {
                "architecture": config["model"]["name"],
                "depth": summary_path.parents[2].name,
                "dense_hidden": [
                    int(width) for width in config["model"]["dense_hidden"]
                ],
                "latent_dim": int(config["model"]["latent_dim"]),
                "run_directory": str(run_directory.relative_to(root)),
                "number_of_parameters": int(summary["number_of_parameters"]),
                "best_epoch": int(summary["best_epoch"]),
                "variance_recovered": float(metrics["variance_recovered"]),
                "log10_mse": float(metrics["log10_mse"]),
                "maximum_relative_error": float(
                    metrics.get("maximum_relative_error", float("nan"))
                ),
                "surface_relative_maximum_p99": float(
                    metrics.get("surface_relative_maximum_p99", float("nan"))
                ),
            }
        )
    return sorted(
        results,
        key=lambda result: (
            result["architecture"],
            result["depth"],
            result["latent_dim"],
            result["run_directory"],
        ),
    )


def smallest_qualified_model(
    results: list[dict[str, Any]],
    *,
    target_variance_recovered: float = 0.999,
) -> dict[str, Any] | None:
    """
    Select the smallest latent and then smallest parameter count meeting a target.
    
    Arguments:
        results (list[dict[str, Any]]):
            Records returned by collect_ae_results.
        target_variance_recovered (float):
            Minimum complete-validation variance recovery.
    
    Returns:
        result (dict[str, Any] or None):
            Best qualified record, or None when no run reaches the target.
    """
    qualified = [
        result
        for result in results
        if result["variance_recovered"] >= target_variance_recovered
    ]
    if not qualified:
        return None
    return min(
        qualified,
        key=lambda result: (
            result["latent_dim"],
            result["number_of_parameters"],
            result["log10_mse"],
        ),
    )
