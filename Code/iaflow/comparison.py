"""
Cross-family result discovery and reconstruction comparisons.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .core.artifacts import portable_path
from .core.metrics import (
    RECONSTRUCTION_COMPARISON_METRIC_NAMES,
    check_reconstruction_metrics,
)

__all__ = [
    "collect_ae_results",
    "collect_compressor_results",
    "matched_pca_comparison",
    "smallest_qualified_compressor",
    "smallest_qualified_model",
]


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


def collect_compressor_results(
    project_root: str | Path,
) -> list[dict[str, Any]]:
    """
    Collect direct-AE and PCA-AE validation summaries without migrating runs.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
    
    Returns:
        results (list[dict[str, Any]]):
            Ordered cross-family validation records.
    """
    root = Path(project_root).expanduser().resolve()
    results = []
    for result in collect_ae_results(root):
        results.append({"model_family": "Direct_AE", **result})
    run_root = root / "Runs" / "NLA" / "PCA_AE"
    if run_root.is_dir():
        for summary_path in run_root.glob("Depth*/Latent*/*/Summary.json"):
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
                    "model_family": "PCA_AE",
                    "architecture": "PCA_AE",
                    "depth": summary_path.parents[2].name,
                    "latent_dim": int(config["model"]["latent_dim"]),
                    "pca_rank": int(config["model"]["pca_rank"]),
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
            result["model_family"],
            result["architecture"],
            result["depth"],
            result["latent_dim"],
            result["run_directory"],
        ),
    )


def smallest_qualified_compressor(
    results: list[dict[str, Any]],
    *,
    target_variance_recovered: float = 0.999,
) -> dict[str, Any] | None:
    """
    Select the smallest validated cross-family compressor meeting the target.
    
    Arguments:
        results (list[dict[str, Any]]):
            Records returned by collect_compressor_results.
        target_variance_recovered (float):
            Minimum complete-validation variance recovery.
    
    Returns:
        result (dict[str, Any] or None):
            Smallest latent, then smallest parameter count, then lowest MSE.
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


def matched_pca_comparison(
    config: object,
    path: Path,
    latent_dim: int,
    metrics: dict[str, float],
    normalization_scale: float,
) -> dict[str, object]:
    """
    Compare complete validation metrics with PCA at the same dimension.
    
    Arguments:
        config (object):
            Checked direct-AE or PCA-AE experiment configuration.
        path (pathlib.Path):
            PCA validation-metrics artifact to load.
        latent_dim (int):
            Compressor latent dimension selecting the PCA rank.
        metrics (dict[str, float]):
            Complete compressor validation metrics.
        normalization_scale (float):
            Normalization scale paired with the compressor checkpoint.
    
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
        "centering": "training feature mean",
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
    check_reconstruction_metrics(
        metrics,
        name="Autoencoder validation metrics",
        normalization_scale=normalization_scale,
    )
    check_reconstruction_metrics(
        pca_metrics,
        name=f"PCA rank {latent_dim} validation metrics",
        normalization_scale=pca_scale,
    )
    if int(pca_metrics.get("number_of_surfaces", -1)) != int(
        metrics["number_of_surfaces"]
    ):
        raise ValueError("PCA and autoencoder validation sample counts differ.")
    differences = {
        name: float(metrics[name]) - float(pca_metrics[name])
        for name in RECONSTRUCTION_COMPARISON_METRIC_NAMES
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
