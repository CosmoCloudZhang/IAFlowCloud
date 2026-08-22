"""
Cross-family result discovery and reconstruction comparisons.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .core.artifacts import portable_path
from .core.data import NormalizationStats
from .core.metrics import (
    RECONSTRUCTION_COMPARISON_METRIC_NAMES,
    RECONSTRUCTION_FRACTIONAL_ERROR_METRIC_NAMES,
    check_reconstruction_metrics,
)

__all__ = [
    "PCA_COMPARISON_SCHEMA_VERSION",
    "check_matched_pca_comparison",
    "collect_ae_results",
    "collect_compressor_results",
    "load_complete_validation_record",
    "matched_pca_comparison",
    "smallest_qualified_compressor",
    "smallest_qualified_model",
]


PCA_COMPARISON_SCHEMA_VERSION = "2.0"


def _resolved_project_path(
    root: Path,
    value: object,
    *,
    name: str,
) -> Path:
    """
    Resolve a required path stored in a resolved experiment configuration.
    
    Arguments:
        root (pathlib.Path):
            IAFlowCloud repository root.
        value (object):
            Candidate absolute or repository-relative path value.
        name (str):
            Configuration field name used in validation errors.
    
    Returns:
        path (pathlib.Path):
            Absolute normalized path.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Resolved configuration {name} must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _complete_validation_record(
    root: Path,
    run_directory: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, object]] | None:
    """
    Load one complete validation artifact using the current comparison schema.
    
    Arguments:
        root (pathlib.Path):
            IAFlowCloud repository root.
        run_directory (pathlib.Path):
            Candidate trained-run directory.
        summary (dict[str, Any]):
            Parsed training summary paired with the run.
        config (dict[str, Any]):
            Parsed resolved experiment configuration.
    
    Returns:
        record (tuple[dict[str, float], dict[str, object]] or None):
            Checked validation metrics and PCA comparison, or None when the
            current complete-validation comparison has not been generated.
    """
    validation_path = run_directory / "ValidationMetrics.json"
    if not validation_path.is_file():
        return None
    with validation_path.open("r", encoding="utf-8") as stream:
        validation = json.load(stream)
    comparison = validation.get("pca_comparison")
    if (
        not isinstance(comparison, dict)
        or comparison.get("comparison_schema_version")
        != PCA_COMPARISON_SCHEMA_VERSION
    ):
        return None
    if validation.get("split") != "validation" or validation.get("final_test") is not False:
        raise ValueError(f"Invalid complete-validation artifact: {validation_path}")
    if int(validation.get("checkpoint_epoch", -1)) != int(summary["best_epoch"]):
        raise ValueError(
            f"Validation checkpoint does not match the best epoch: {validation_path}"
        )
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Resolved model configuration is invalid: {run_directory}")
    if int(validation.get("latent_dim", -1)) != int(model["latent_dim"]):
        raise ValueError(f"Validation latent dimension is invalid: {validation_path}")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Resolved data configuration is invalid: {run_directory}")
    cache_directory = _resolved_project_path(
        root,
        data.get("cache_directory"),
        name="data.cache_directory",
    )
    normalization = NormalizationStats.load(cache_directory / "Normalization.npz")
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Validation metrics are invalid: {validation_path}")
    check_matched_pca_comparison(
        comparison,
        metrics,
        normalization_scale=normalization.scale,
    )
    return metrics, comparison


def load_complete_validation_record(
    project_root: str | Path,
    run_directory: str | Path,
) -> tuple[dict[str, float], dict[str, object]] | None:
    """
    Load and check one current-schema complete-validation result.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
        run_directory (str or pathlib.Path):
            Absolute or repository-relative trained-run directory.
    
    Returns:
        record (tuple[dict[str, float], dict[str, object]] or None):
            Checked validation metrics and PCA comparison, or None when a
            current comparison artifact is absent.
    """
    root = Path(project_root).expanduser().resolve()
    directory = Path(run_directory).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    summary_path = directory / "Summary.json"
    config_path = directory / "ResolvedConfig.json"
    if not summary_path.is_file() or not config_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    return _complete_validation_record(root, directory, summary, config)


def _comparison_result_fields(
    metrics: dict[str, float],
    comparison: dict[str, object],
) -> dict[str, Any]:
    """
    Build common validated metric and matched-PCA result fields.
    
    Arguments:
        metrics (dict[str, float]):
            Checked complete validation metrics.
        comparison (dict[str, object]):
            Checked current-schema matched-PCA comparison.
    
    Returns:
        fields (dict[str, Any]):
            Common collector fields for result tables and model selection.
    """
    return {
        "validation_metrics": metrics,
        "pca_comparison": comparison,
        "comparison_schema_version": comparison["comparison_schema_version"],
        "fractional_error_reduction": comparison[
            "autoencoder_fractional_error_reduction"
        ],
        "variance_recovered_percentage_point_gain": comparison[
            "variance_recovered_percentage_point_gain"
        ],
        "variance_recovered": float(metrics["variance_recovered"]),
        "log10_mse": float(metrics["log10_mse"]),
        "maximum_relative_error": float(metrics["maximum_relative_error"]),
        "surface_relative_maximum_p99": float(
            metrics["surface_relative_maximum_p99"]
        ),
    }


def _fractional_error_reductions(
    metrics: dict[str, float],
    pca_metrics: dict[str, float],
) -> dict[str, float]:
    """
    Calculate signed fractional error reductions relative to matched PCA.
    
    Arguments:
        metrics (dict[str, float]):
            Complete compressor validation metrics.
        pca_metrics (dict[str, float]):
            Complete PCA validation metrics at the matched rank.
    
    Returns:
        reductions (dict[str, float]):
            Fractional reductions, where positive values mean lower model error.
    """
    reductions: dict[str, float] = {}
    for name in RECONSTRUCTION_FRACTIONAL_ERROR_METRIC_NAMES:
        pca_value = float(pca_metrics[name])
        if pca_value <= 0.0:
            raise ValueError(
                f"PCA comparison denominator {name!r} must be positive."
            )
        reduction = 1.0 - float(metrics[name]) / pca_value
        if not math.isfinite(reduction):
            raise ValueError(f"PCA fractional reduction {name!r} is not finite.")
        reductions[name] = reduction
    return reductions


def check_matched_pca_comparison(
    comparison: object,
    metrics: dict[str, float],
    *,
    normalization_scale: float,
) -> None:
    """
    Check a matched-PCA comparison and its reconstruction identities.
    
    Arguments:
        comparison (object):
            Candidate matched-PCA comparison mapping.
        metrics (dict[str, float]):
            Complete compressor validation metrics paired with the comparison.
        normalization_scale (float):
            Positive global RMS used for both reconstruction metric sets.
    """
    if not isinstance(comparison, dict):
        raise ValueError("PCA comparison must be a mapping.")
    required_names = {
        "comparison_schema_version",
        "reference",
        "rank",
        "pca_metrics",
        "autoencoder_minus_pca",
        "autoencoder_fractional_error_reduction",
        "variance_recovered_percentage_point_gain",
        "autoencoder_outperforms_pca",
    }
    if set(comparison) != required_names:
        raise ValueError(
            "PCA comparison must contain exactly "
            f"{sorted(required_names)}."
        )
    if comparison["comparison_schema_version"] != PCA_COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported PCA comparison schema version: "
            f"{comparison['comparison_schema_version']!r}."
        )
    if not isinstance(comparison["reference"], str) or not comparison["reference"]:
        raise ValueError("PCA comparison reference must be a non-empty path.")
    rank = comparison["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("PCA comparison rank must be a positive integer.")
    pca_metrics = comparison["pca_metrics"]
    if not isinstance(pca_metrics, dict):
        raise ValueError("PCA comparison metrics must be a mapping.")
    check_reconstruction_metrics(
        metrics,
        name="Autoencoder validation metrics",
        normalization_scale=normalization_scale,
    )
    check_reconstruction_metrics(
        pca_metrics,
        name=f"PCA rank {rank} validation metrics",
        normalization_scale=normalization_scale,
    )
    if int(pca_metrics["number_of_surfaces"]) != int(
        metrics["number_of_surfaces"]
    ):
        raise ValueError("PCA and autoencoder validation sample counts differ.")
    
    differences = comparison["autoencoder_minus_pca"]
    if not isinstance(differences, dict) or set(differences) != set(
        RECONSTRUCTION_COMPARISON_METRIC_NAMES
    ):
        raise ValueError("PCA comparison differences have an invalid schema.")
    for name in RECONSTRUCTION_COMPARISON_METRIC_NAMES:
        expected = float(metrics[name]) - float(pca_metrics[name])
        difference = differences[name]
        if (
            isinstance(difference, bool)
            or not isinstance(difference, (int, float))
            or not math.isfinite(float(difference))
            or not math.isclose(
                float(difference),
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(f"PCA comparison difference {name!r} is inconsistent.")
    
    reductions = comparison["autoencoder_fractional_error_reduction"]
    expected_reductions = _fractional_error_reductions(metrics, pca_metrics)
    if not isinstance(reductions, dict) or set(reductions) != set(
        RECONSTRUCTION_FRACTIONAL_ERROR_METRIC_NAMES
    ):
        raise ValueError("PCA fractional reductions have an invalid schema.")
    for name, expected in expected_reductions.items():
        reduction = reductions[name]
        if (
            isinstance(reduction, bool)
            or not isinstance(reduction, (int, float))
            or not math.isfinite(float(reduction))
            or not math.isclose(
                float(reduction),
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(f"PCA fractional reduction {name!r} is inconsistent.")
    
    variance_gain = comparison["variance_recovered_percentage_point_gain"]
    expected_variance_gain = 100.0 * (
        float(metrics["variance_recovered"])
        - float(pca_metrics["variance_recovered"])
    )
    if (
        isinstance(variance_gain, bool)
        or not isinstance(variance_gain, (int, float))
        or not math.isfinite(float(variance_gain))
        or not math.isclose(
            float(variance_gain),
            expected_variance_gain,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("PCA variance-recovery percentage-point gain is inconsistent.")
    
    pca_unrecovered_variance = 1.0 - float(pca_metrics["variance_recovered"])
    model_unrecovered_variance = 1.0 - float(metrics["variance_recovered"])
    if pca_unrecovered_variance <= 0.0:
        raise ValueError("PCA unrecovered variance must be positive.")
    variance_error_reduction = (
        1.0 - model_unrecovered_variance / pca_unrecovered_variance
    )
    if not math.isclose(
        variance_error_reduction,
        float(reductions["log10_mse"]),
        rel_tol=1.0e-6,
        abs_tol=1.0e-10,
    ):
        raise ValueError(
            "Log10 MSE reduction is inconsistent with unrecovered variance."
        )
    if not math.isclose(
        (1.0 - float(reductions["log10_rmse"])) ** 2,
        1.0 - float(reductions["log10_mse"]),
        rel_tol=1.0e-7,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Log10 RMSE and MSE reductions are inconsistent.")
    
    expected_outcome = bool(
        metrics["variance_recovered"] > pca_metrics["variance_recovered"]
        and metrics["log10_mse"] < pca_metrics["log10_mse"]
    )
    if comparison["autoencoder_outperforms_pca"] is not expected_outcome:
        raise ValueError("PCA comparison outcome flag is inconsistent.")


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
        validation_record = _complete_validation_record(
            root,
            run_directory,
            summary,
            config,
        )
        if validation_record is None:
            continue
        metrics, comparison = validation_record
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
                **_comparison_result_fields(metrics, comparison),
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
            validation_record = _complete_validation_record(
                root,
                run_directory,
                summary,
                config,
            )
            if validation_record is None:
                continue
            metrics, comparison = validation_record
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
                    **_comparison_result_fields(metrics, comparison),
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
            Versioned matched PCA metrics, absolute differences, fractional
            error reductions, variance gain, and outcome flag.
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
    comparison = {
        "comparison_schema_version": PCA_COMPARISON_SCHEMA_VERSION,
        "reference": portable_path(reference_path, config.project_root),
        "rank": latent_dim,
        "pca_metrics": pca_metrics,
        "autoencoder_minus_pca": differences,
        "autoencoder_fractional_error_reduction": (
            _fractional_error_reductions(metrics, pca_metrics)
        ),
        "variance_recovered_percentage_point_gain": 100.0
        * (
            float(metrics["variance_recovered"])
            - float(pca_metrics["variance_recovered"])
        ),
        "autoencoder_outperforms_pca": bool(
            metrics["variance_recovered"] > pca_metrics["variance_recovered"]
            and metrics["log10_mse"] < pca_metrics["log10_mse"]
        ),
    }
    check_matched_pca_comparison(
        comparison,
        metrics,
        normalization_scale=normalization_scale,
    )
    return comparison
