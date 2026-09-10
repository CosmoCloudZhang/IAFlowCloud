"""
Cross-family result discovery and reconstruction comparisons.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .autoencoder.config import load_experiment_template
from .core.artifacts import portable_path
from .core.config import config_to_dict
from .core.data import NormalizationStats
from .core.metrics import (
    RECONSTRUCTION_COMPARISON_METRIC_NAMES,
    RECONSTRUCTION_FRACTIONAL_ERROR_METRIC_NAMES,
    RECONSTRUCTION_METRIC_NAMES,
    check_reconstruction_metrics,
)
from .core.runs import discover_run_directories
from .pca_autoencoder.config import load_pca_ae_experiment_template

__all__ = [
    "audit_compressor_runs",
    "check_matched_pca_comparison",
    "collect_ae_results",
    "collect_compressor_results",
    "compressor_selection_input_sha256",
    "expected_compressor_cells",
    "load_complete_validation_record",
    "matched_pca_comparison",
    "notebook_source_sha256",
    "smallest_qualified_compressor",
    "smallest_qualified_model",
]


DEFAULT_LATENT_DIMENSIONS = (2, 4, 6, 8, 10)
_MATCHED_PCA_COMPARISON_FIELDS = (
    "reference",
    "rank",
    "pca_metrics",
    "autoencoder_minus_pca",
    "autoencoder_fractional_error_reduction",
    "variance_recovered_percentage_point_gain",
    "autoencoder_outperforms_pca",
)
_SELECTION_AUDIT_FIELDS = (
    "cell_id",
    "model_family",
    "architecture",
    "depth",
    "latent_dim",
    "template_path",
    "latent_root",
    "run_directory",
    "status",
    "current_configuration",
    "configuration_sources_match",
    "training_complete",
    "validation_ready",
    "diagnostics_ready",
    "latents_available",
    "eligible_for_comparison",
    "exclusion_reasons",
    "warnings",
)
_SELECTION_ELIGIBLE_ARTIFACTS = (
    "ResolvedConfig.json",
    "Architecture.json",
    "Summary.json",
    "ValidationMetrics.json",
    "ValidationDiagnostics.json",
)


def _load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    """
    Load one JSON object from an artifact path.
    
    Arguments:
        path (pathlib.Path):
            Existing JSON artifact path.
    
    Returns:
        values (dict[str, Any]):
            Parsed JSON mapping.
    """
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return values


def _sha256_file(
    path: Path,
) -> str:
    """
    Calculate the SHA256 digest of one existing file.
    
    Arguments:
        path (pathlib.Path):
            Existing regular file to hash.
    
    Returns:
        digest (str):
            Lowercase hexadecimal SHA256 digest.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required selection input does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def notebook_source_sha256(
    path: str | Path,
) -> str:
    """
    Hash the ordered markdown and code sources in one Jupyter notebook.
    
    Execution counts, outputs, and notebook metadata are deliberately excluded
    so a successful execution does not invalidate the authenticated analysis
    implementation.
    
    Arguments:
        path (str or pathlib.Path):
            Existing Jupyter notebook path.
    
    Returns:
        digest (str):
            Lowercase hexadecimal SHA256 digest of the canonical cell sources.
    """
    notebook_path = Path(path).expanduser().resolve()
    notebook = _load_json_mapping(notebook_path)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook cells must be a list: {notebook_path}")
    canonical_cells = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError(
                f"Notebook cell {index} must be a mapping: {notebook_path}"
            )
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError(
                f"Notebook cell {index} has an invalid type: {cell_type!r}"
            )
        source = cell.get("source", "")
        if isinstance(source, list) and all(
            isinstance(line, str) for line in source
        ):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError(
                f"Notebook cell {index} has invalid source text: {notebook_path}"
            )
        canonical_cells.append(
            {
                "cell_type": cell_type,
                "source": source,
            }
        )
    payload = json.dumps(
        canonical_cells,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repository_file_record(
    root: Path,
    value: str | Path,
    *,
    name: str,
) -> dict[str, str]:
    """
    Build a repository-relative path and digest record.
    
    Arguments:
        root (pathlib.Path):
            Resolved IAFlowCloud repository root.
        value (str or pathlib.Path):
            Absolute or repository-relative file path.
        name (str):
            Input name used in validation errors.
    
    Returns:
        record (dict[str, str]):
            Repository-relative POSIX path and SHA256 digest.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside the project root: {path}") from error
    return {
        "path": relative_path.as_posix(),
        "sha256": _sha256_file(path),
    }


def _canonical_matched_pca_comparison(
    comparison: object,
) -> dict[str, object] | None:
    """
    Retain the required scientific fields from a PCA comparison mapping.
    
    Extra historical metadata is deliberately ignored. A mapping missing any
    required scientific result is incomplete and cannot be used to skip
    validation evaluation.
    
    Arguments:
        comparison (object):
            Candidate matched-PCA comparison mapping.
    
    Returns:
        canonical (dict[str, object] or None):
            Deep-copied required fields, or None for an incomplete mapping.
    """
    if not isinstance(comparison, Mapping):
        return None
    if not set(_MATCHED_PCA_COMPARISON_FIELDS).issubset(comparison):
        return None
    return {
        name: copy.deepcopy(comparison[name])
        for name in _MATCHED_PCA_COMPARISON_FIELDS
    }


def _semantic_configuration(
    values: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """
    Separate scientific run settings from mutable provenance fields.
    
    The total epoch ceiling is initially excluded because compatible PCA-AE
    continuation may extend it without changing a trained architecture or data
    contract. Direct-AE auditing separately requires the current exact epoch
    ceiling. Configuration-source hashes remain available as a provenance
    check.
    
    Arguments:
        values (dict[str, Any]):
            JSON-safe resolved experiment configuration.
    
    Returns:
        result (tuple[dict[str, Any], list[dict[str, str]]]):
            Semantic configuration and ordered configuration-source records.
    """
    semantic = copy.deepcopy(values)
    runtime = semantic.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        semantic["runtime"] = runtime
    sources = runtime.pop("configuration_sources", [])
    if not isinstance(sources, list):
        sources = []
    training = semantic.get("training")
    if isinstance(training, dict):
        training.pop("epochs", None)
    return semantic, copy.deepcopy(sources)


def expected_compressor_cells(
    project_root: str | Path,
    *,
    latent_dimensions: Sequence[int] = DEFAULT_LATENT_DIMENSIONS,
) -> list[dict[str, Any]]:
    """
    Build the current direct-AE and PCA-AE comparison grid.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
        latent_dimensions (collections.abc.Sequence[int]):
            Positive bottleneck dimensions included in the comparison.
    
    Returns:
        cells (list[dict[str, Any]]):
            Ordered expected family, architecture, depth, latent, template,
            and run-root records.
    """
    root = Path(project_root).expanduser().resolve()
    dimensions = []
    for value in latent_dimensions:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("latent_dimensions must contain positive integers.")
        dimensions.append(value)
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("latent_dimensions cannot contain duplicates.")
    cells: list[dict[str, Any]] = []
    direct_root = root / "Config" / "NLA" / "AE"
    for template_path in sorted(direct_root.glob("*/Depth*.yaml")):
        architecture = template_path.parent.name
        depth = template_path.stem
        template = load_experiment_template(template_path, project_root=root)
        run_root = template.resolve_path(template.output.root_directory)
        for latent_dim in dimensions:
            cells.append(
                {
                    "cell_id": (
                        f"Direct_AE:{architecture}:{depth}:L{latent_dim:02d}"
                    ),
                    "model_family": "Direct_AE",
                    "architecture": architecture,
                    "depth": depth,
                    "latent_dim": latent_dim,
                    "template_path": str(template_path.relative_to(root)),
                    "latent_root": str(
                        (run_root / f"Latent{latent_dim:02d}").relative_to(root)
                    ),
                }
            )
    pca_ae_root = root / "Config" / "NLA" / "PCA_AE"
    for template_path in sorted(pca_ae_root.glob("Depth*.yaml")):
        depth = template_path.stem
        template = load_pca_ae_experiment_template(
            template_path,
            project_root=root,
        )
        run_root = template.resolve_path(template.output.root_directory)
        for latent_dim in dimensions:
            cells.append(
                {
                    "cell_id": f"PCA_AE:PCA_AE:{depth}:L{latent_dim:02d}",
                    "model_family": "PCA_AE",
                    "architecture": "PCA_AE",
                    "depth": depth,
                    "latent_dim": latent_dim,
                    "template_path": str(template_path.relative_to(root)),
                    "latent_root": str(
                        (run_root / f"Latent{latent_dim:02d}").relative_to(root)
                    ),
                }
            )
    return sorted(
        cells,
        key=lambda cell: (
            cell["model_family"],
            cell["architecture"],
            cell["depth"],
            cell["latent_dim"],
        ),
    )


def _expected_resolved_configuration(
    root: Path,
    cell: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """
    Resolve one current template for a concrete run directory.
    
    Arguments:
        root (pathlib.Path):
            IAFlowCloud repository root.
        cell (dict[str, Any]):
            Expected comparison-cell record.
        run_directory (pathlib.Path):
            Concrete candidate run directory.
    
    Returns:
        values (dict[str, Any]):
            JSON-safe expected resolved configuration.
    """
    template_path = root / cell["template_path"]
    if cell["model_family"] == "Direct_AE":
        template = load_experiment_template(template_path, project_root=root)
    else:
        template = load_pca_ae_experiment_template(
            template_path,
            project_root=root,
        )
    config = template.resolve(
        int(cell["latent_dim"]),
        run_directory,
        runtime={
            "maximum_train_samples": None,
            "maximum_validation_samples": None,
        },
    )
    return config_to_dict(config)


def _common_result_fields(
    summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract common training and architecture metadata for one result.
    
    Arguments:
        summary (dict[str, Any]):
            Checked training summary.
        config (dict[str, Any]):
            Persisted resolved configuration.
    
    Returns:
        fields (dict[str, Any]):
            Common training, stopping, device, and model metadata.
    """
    training = config["training"]
    model = config["model"]
    return {
        "dense_hidden": [int(width) for width in model["dense_hidden"]],
        "training_seed": int(training["seed"]),
        "deterministic": bool(training["deterministic"]),
        "configured_epoch_ceiling": int(training["epochs"]),
        "training_samples": int(summary["training_samples"]),
        "validation_samples": int(summary["validation_samples"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "best_epoch": int(summary["best_epoch"]),
        "stopped_early": bool(summary["stopped_early"]),
        "device": str(summary["device"]),
        "test_split_used_during_training": bool(
            summary["test_split_used_during_training"]
        ),
        "number_of_parameters": int(summary["number_of_parameters"]),
    }


def _audit_run_directory(
    root: Path,
    cell: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """
    Audit one candidate against its current template and artifact contract.
    
    Arguments:
        root (pathlib.Path):
            IAFlowCloud repository root.
        cell (dict[str, Any]):
            Expected comparison-cell record.
        run_directory (pathlib.Path):
            Candidate run directory within the cell's canonical run root.
    
    Returns:
        audit (dict[str, Any]):
            Eligibility flags, status, reasons, warnings, and optional result.
    """
    audit = {
        **copy.deepcopy(cell),
        "run_directory": str(run_directory.relative_to(root)),
        "status": "incomplete",
        "current_configuration": False,
        "configuration_sources_match": False,
        "training_complete": False,
        "validation_ready": False,
        "diagnostics_ready": False,
        "latents_available": (run_directory / "Latents.hdf5").is_file(),
        "eligible_for_comparison": False,
        "exclusion_reasons": [],
        "warnings": [],
        "result": None,
    }
    reasons = audit["exclusion_reasons"]
    warnings = audit["warnings"]
    config_path = run_directory / "ResolvedConfig.json"
    if not config_path.is_file():
        reasons.append("missing ResolvedConfig.json")
        return audit
    try:
        config = _load_json_mapping(config_path)
        expected = _expected_resolved_configuration(root, cell, run_directory)
        semantic, sources = _semantic_configuration(config)
        expected_semantic, expected_sources = _semantic_configuration(expected)
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"invalid resolved configuration: {error}")
        audit["status"] = "invalid"
        return audit
    audit["current_configuration"] = semantic == expected_semantic
    if cell["model_family"] == "Direct_AE":
        actual_training = config.get("training")
        expected_training = expected.get("training")
        if (
            not isinstance(actual_training, dict)
            or not isinstance(expected_training, dict)
            or actual_training.get("epochs") != expected_training.get("epochs")
        ):
            audit["current_configuration"] = False
    audit["configuration_sources_match"] = sources == expected_sources
    if not audit["current_configuration"]:
        reasons.append("resolved settings differ from the current template")
        audit["status"] = "legacy"
    if not audit["configuration_sources_match"]:
        warnings.append("configuration-source hashes differ from the current files")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict) or any(
        runtime.get(name) is not None
        for name in ("maximum_train_samples", "maximum_validation_samples")
    ):
        reasons.append("run used smoke-sample limits")
    required_training = (
        "Summary.json",
        "Best.pt",
        "Architecture.json",
        "History.json",
    )
    missing_training = [
        name for name in required_training if not (run_directory / name).is_file()
    ]
    if missing_training:
        if audit["status"] != "legacy":
            audit["status"] = "training"
        warnings.append("training artifacts pending: " + ", ".join(missing_training))
        return audit
    audit["training_complete"] = True
    try:
        summary = _load_json_mapping(run_directory / "Summary.json")
        architecture = _load_json_mapping(run_directory / "Architecture.json")
        with (run_directory / "History.json").open("r", encoding="utf-8") as stream:
            history = json.load(stream)
        if not isinstance(history, list) or not history:
            raise ValueError("History.json must contain a non-empty list.")
        if int(history[-1]["epoch"]) != int(summary["epochs_completed"]):
            raise ValueError("history endpoint does not match epochs_completed")
        if len(history) != int(summary["epochs_completed"]):
            raise ValueError("history length does not match epochs_completed")
        if sum(
            int(record["epoch"]) == int(summary["best_epoch"])
            for record in history
        ) != 1:
            raise ValueError("best_epoch is not represented exactly once in history")
        if int(architecture["number_of_parameters"]) != int(
            summary["number_of_parameters"]
        ):
            raise ValueError("architecture and summary parameter counts differ")
        if int(architecture["latent_dim"]) != int(cell["latent_dim"]):
            raise ValueError("architecture latent dimension differs from the grid")
        if int(summary["training_samples"]) != 80000:
            raise ValueError("training sample count is not 80000")
        if int(summary["validation_samples"]) != 10000:
            raise ValueError("validation sample count is not 10000")
        if summary["test_split_used_during_training"] is not False:
            raise ValueError("training summary reports test-split use")
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"inconsistent training artifacts: {error}")
        audit["status"] = "invalid"
        return audit
    try:
        validation_record = _complete_validation_record(
            root,
            run_directory,
            summary,
            config,
        )
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"invalid validation artifact: {error}")
        audit["status"] = "invalid"
        return audit
    if validation_record is None:
        if audit["status"] != "legacy":
            audit["status"] = "validation pending"
        warnings.append("current complete-validation artifact is pending")
        return audit
    metrics, comparison = validation_record
    reference_path = root / str(comparison["reference"])
    try:
        reference = _load_json_mapping(reference_path)
        reference_metrics = reference["ranks"][str(cell["latent_dim"])]
        if comparison["pca_metrics"] != reference_metrics:
            raise ValueError("embedded PCA metrics differ from the reference file")
        for name, value in summary["best_validation_metrics"].items():
            if not math.isclose(
                float(metrics[name]),
                float(value),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    f"summary and complete-validation metric {name!r} differ"
                )
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"inconsistent validation provenance: {error}")
        audit["status"] = "invalid"
        return audit
    audit["validation_ready"] = True
    diagnostic_json = run_directory / "ValidationDiagnostics.json"
    diagnostic_npz = run_directory / "ValidationDiagnostics.npz"
    if diagnostic_json.is_file() and diagnostic_npz.is_file():
        try:
            diagnostics = _load_json_mapping(diagnostic_json)
            if diagnostics["complete_validation_split"] is not True:
                raise ValueError("diagnostics are not marked complete")
            if int(diagnostics["number_of_surfaces"]) != 10000:
                raise ValueError("diagnostic surface count is not 10000")
            diagnostic_names = (
                "maximum_relative_error",
                "surface_relative_rmse_p95",
                "surface_relative_rmse_p99",
                "surface_relative_maximum_p95",
                "surface_relative_maximum_p99",
            )
            for name in diagnostic_names:
                if not math.isclose(
                    float(diagnostics[name]),
                    float(metrics[name]),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-8,
                ):
                    raise ValueError(f"diagnostic metric {name!r} is inconsistent")
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f"diagnostics are inconsistent: {error}")
        else:
            audit["diagnostics_ready"] = True
    result = {
        "model_family": cell["model_family"],
        "architecture": cell["architecture"],
        "depth": cell["depth"],
        "latent_dim": int(cell["latent_dim"]),
        "run_directory": str(run_directory.relative_to(root)),
        "diagnostics_ready": bool(audit["diagnostics_ready"]),
        "latents_available": bool(audit["latents_available"]),
        **_common_result_fields(summary, config),
        **_comparison_result_fields(metrics, comparison),
    }
    if cell["model_family"] == "PCA_AE":
        result["pca_rank"] = int(config["model"]["pca_rank"])
    audit["result"] = result
    if audit["current_configuration"] and not reasons:
        audit["eligible_for_comparison"] = True
        audit["status"] = "eligible"
    return audit


def audit_compressor_runs(
    project_root: str | Path,
    *,
    latent_dimensions: Sequence[int] = DEFAULT_LATENT_DIMENSIONS,
) -> list[dict[str, Any]]:
    """
    Audit current, incomplete, missing, and legacy compressor runs.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
        latent_dimensions (collections.abc.Sequence[int]):
            Positive bottleneck dimensions included in the comparison.
    
    Returns:
        audits (list[dict[str, Any]]):
            One record per discovered candidate, or one missing record for an
            expected cell with no candidate directories.
    """
    root = Path(project_root).expanduser().resolve()
    audits: list[dict[str, Any]] = []
    for cell in expected_compressor_cells(
        root,
        latent_dimensions=latent_dimensions,
    ):
        latent_root = root / cell["latent_root"]
        run_directories = discover_run_directories(latent_root)
        if not run_directories:
            audits.append(
                {
                    **copy.deepcopy(cell),
                    "run_directory": None,
                    "status": "missing",
                    "current_configuration": False,
                    "configuration_sources_match": False,
                    "training_complete": False,
                    "validation_ready": False,
                    "diagnostics_ready": False,
                    "latents_available": False,
                    "eligible_for_comparison": False,
                    "exclusion_reasons": [],
                    "warnings": [],
                    "result": None,
                }
            )
            continue
        for run_directory in run_directories:
            audits.append(_audit_run_directory(root, cell, run_directory))
    return sorted(
        audits,
        key=lambda audit: (
            audit["model_family"],
            audit["architecture"],
            audit["depth"],
            audit["latent_dim"],
            audit["run_directory"] or "",
        ),
    )


def compressor_selection_input_sha256(
    project_root: str | Path,
    *,
    policy: Mapping[str, Any],
    pca_metrics_path: str | Path,
    pca_metadata_path: str | Path,
    latent_dimensions: Sequence[int] = DEFAULT_LATENT_DIMENSIONS,
) -> str:
    """
    Fingerprint the scientific inputs and live eligibility for model selection.
    
    Incomplete runs contribute only their deterministic audit state. Immutable
    scientific artifacts are hashed only after a run becomes eligible, so
    mutable checkpoints and training histories do not destabilize the input
    identity while training is in progress.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
        policy (collections.abc.Mapping[str, Any]):
            JSON-safe model-selection policy included verbatim in the identity.
        pca_metrics_path (str or pathlib.Path):
            PCA validation-metrics artifact inside the repository.
        pca_metadata_path (str or pathlib.Path):
            PCA transform-metadata artifact inside the repository.
        latent_dimensions (collections.abc.Sequence[int]):
            Positive bottleneck dimensions included in the comparison.
    
    Returns:
        digest (str):
            Lowercase SHA256 digest of the canonical selection-input mapping.
    """
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be a JSON-safe mapping.")
    root = Path(project_root).expanduser().resolve()
    expected_records = []
    for cell in expected_compressor_cells(
        root,
        latent_dimensions=latent_dimensions,
    ):
        template = _repository_file_record(
            root,
            cell["template_path"],
            name=f"template for {cell['cell_id']}",
        )
        expected_records.append(
            {
                "cell_id": cell["cell_id"],
                "model_family": cell["model_family"],
                "architecture": cell["architecture"],
                "depth": cell["depth"],
                "latent_dim": int(cell["latent_dim"]),
                "template_path": template["path"],
                "template_sha256": template["sha256"],
                "latent_root": cell["latent_root"],
            }
        )
    expected_records.sort(
        key=lambda record: (
            record["model_family"],
            record["architecture"],
            record["depth"],
            record["latent_dim"],
            record["cell_id"],
        ),
    )
    audit_records = []
    for audit in audit_compressor_runs(
        root,
        latent_dimensions=latent_dimensions,
    ):
        record = {
            name: copy.deepcopy(audit[name])
            for name in _SELECTION_AUDIT_FIELDS
        }
        if audit["eligible_for_comparison"]:
            run_directory_value = audit["run_directory"]
            if not isinstance(run_directory_value, str):
                raise ValueError(
                    f"Eligible run {audit['cell_id']} has no run directory."
                )
            run_directory = (root / run_directory_value).resolve()
            record["eligible_artifact_sha256"] = {
                name: _sha256_file(run_directory / name)
                for name in _SELECTION_ELIGIBLE_ARTIFACTS
            }
        audit_records.append(record)
    audit_records.sort(
        key=lambda record: (
            record["model_family"],
            record["architecture"],
            record["depth"],
            record["latent_dim"],
            record["run_directory"] or "",
        ),
    )
    inputs = {
        "policy": copy.deepcopy(dict(policy)),
        "pca_metrics": _repository_file_record(
            root,
            pca_metrics_path,
            name="pca_metrics_path",
        ),
        "pca_metadata": _repository_file_record(
            root,
            pca_metadata_path,
            name="pca_metadata_path",
        ),
        "expected_cells": expected_records,
        "audits": audit_records,
    }
    canonical_json = json.dumps(
        inputs,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


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


def _check_matched_pca_reference(
    root: Path,
    config: dict[str, Any],
    comparison: dict[str, object],
    *,
    normalization_scale: float,
) -> None:
    """
    Authenticate embedded matched-PCA metrics against their reference file.
    
    Arguments:
        root (pathlib.Path):
            IAFlowCloud repository root.
        config (dict[str, Any]):
            Persisted resolved experiment configuration.
        comparison (dict[str, object]):
            Canonical content-complete matched-PCA comparison.
        normalization_scale (float):
            Positive global RMS paired with the model checkpoint.
    """
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Resolved data configuration must be a mapping.")
    reference_path = _resolved_project_path(
        root,
        comparison["reference"],
        name="pca_comparison.reference",
    )
    reference = _load_json_mapping(reference_path)
    expected_metadata = {
        "split": "validation",
        "source_dataset": data.get("source_path"),
        "target_dataset": data.get("target_dataset"),
        "transform": data.get("transform"),
        "centering": "training feature mean",
        "normalization": data.get("normalization"),
    }
    for name, expected in expected_metadata.items():
        if reference.get(name) != expected:
            raise ValueError(
                f"PCA comparison reference {name!r} differs from the run."
            )
    reference_scale = float(reference.get("normalization_scale", float("nan")))
    if not math.isclose(
        reference_scale,
        normalization_scale,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "PCA comparison reference normalization differs from the run."
        )
    ranks = reference.get("ranks")
    rank = comparison["rank"]
    if not isinstance(ranks, dict) or str(rank) not in ranks:
        raise ValueError(f"PCA comparison reference does not contain rank {rank}.")
    if comparison["pca_metrics"] != ranks[str(rank)]:
        raise ValueError(
            "Embedded PCA metrics differ from the matched reference metrics."
        )


def _complete_validation_record(
    root: Path,
    run_directory: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, object]] | None:
    """
    Load one content-complete validation artifact.
    
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
            required complete-validation comparison has not been generated.
    """
    validation_path = run_directory / "ValidationMetrics.json"
    if not validation_path.is_file():
        return None
    with validation_path.open("r", encoding="utf-8") as stream:
        validation = json.load(stream)
    comparison = _canonical_matched_pca_comparison(
        validation.get("pca_comparison")
    )
    if comparison is None:
        return None
    if validation.get("split") != "validation" or validation.get("final_test") is not False:
        raise ValueError(f"Invalid complete-validation artifact: {validation_path}")
    if int(summary.get("training_samples", -1)) != 80000:
        raise ValueError(f"Invalid full-training sample count: {run_directory}")
    if int(summary.get("validation_samples", -1)) != 10000:
        raise ValueError(f"Invalid full-validation sample count: {run_directory}")
    if summary.get("test_split_used_during_training") is not False:
        raise ValueError(f"Training summary reports test-split use: {run_directory}")
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
    if int(metrics.get("number_of_surfaces", -1)) != 10000:
        raise ValueError(f"Validation metrics are not complete: {validation_path}")
    check_matched_pca_comparison(
        comparison,
        metrics,
        normalization_scale=normalization.scale,
    )
    if int(comparison["rank"]) != int(model["latent_dim"]):
        raise ValueError(f"Matched-PCA rank is invalid: {validation_path}")
    _check_matched_pca_reference(
        root,
        config,
        comparison,
        normalization_scale=normalization.scale,
    )
    summary_metrics = summary.get("best_validation_metrics")
    check_reconstruction_metrics(
        summary_metrics,
        name="Training-summary validation metrics",
        normalization_scale=normalization.scale,
    )
    for name in RECONSTRUCTION_METRIC_NAMES:
        if name == "number_of_surfaces":
            agrees = int(summary_metrics[name]) == int(metrics[name])
        else:
            agrees = math.isclose(
                float(summary_metrics[name]),
                float(metrics[name]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        if not agrees:
            raise ValueError(
                f"Training summary and validation metric {name!r} differ."
            )
    return metrics, comparison


def load_complete_validation_record(
    project_root: str | Path,
    run_directory: str | Path,
) -> tuple[dict[str, float], dict[str, object]] | None:
    """
    Load and check one content-complete validation result.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
        run_directory (str or pathlib.Path):
            Absolute or repository-relative trained-run directory.
    
    Returns:
        record (tuple[dict[str, float], dict[str, object]] or None):
            Checked validation metrics and PCA comparison, or None when a
            complete comparison artifact is absent.
    """
    root = Path(project_root).expanduser().resolve()
    directory = Path(run_directory).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    summary_path = directory / "Summary.json"
    config_path = directory / "ResolvedConfig.json"
    if (
        not summary_path.is_file()
        or not config_path.is_file()
        or not (directory / "Best.pt").is_file()
    ):
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
            Checked content-complete matched-PCA comparison.
    
    Returns:
        fields (dict[str, Any]):
            Common collector fields for result tables and model selection.
    """
    return {
        "validation_metrics": metrics,
        "pca_comparison": comparison,
        "fractional_error_reduction": comparison[
            "autoencoder_fractional_error_reduction"
        ],
        "variance_recovered_percentage_point_gain": comparison[
            "variance_recovered_percentage_point_gain"
        ],
        "autoencoder_outperforms_pca": bool(
            comparison["autoencoder_outperforms_pca"]
        ),
        "normalized_mse": float(metrics["normalized_mse"]),
        "variance_recovered": float(metrics["variance_recovered"]),
        "log10_mse": float(metrics["log10_mse"]),
        "log10_rmse": float(metrics["log10_rmse"]),
        "log10_mae": float(metrics["log10_mae"]),
        "mean_relative_error": float(metrics["mean_relative_error"]),
        "maximum_relative_error": float(metrics["maximum_relative_error"]),
        "surface_relative_rmse_p95": float(
            metrics["surface_relative_rmse_p95"]
        ),
        "surface_relative_rmse_p99": float(
            metrics["surface_relative_rmse_p99"]
        ),
        "surface_relative_maximum_p95": float(
            metrics["surface_relative_maximum_p95"]
        ),
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
    
    Extra metadata is ignored so historical complete comparisons remain
    readable. Every required scientific field is still validated numerically.
    
    Arguments:
        comparison (object):
            Candidate matched-PCA comparison mapping.
        metrics (dict[str, float]):
            Complete compressor validation metrics paired with the comparison.
        normalization_scale (float):
            Positive global RMS used for both reconstruction metric sets.
    """
    canonical = _canonical_matched_pca_comparison(comparison)
    if canonical is None:
        if not isinstance(comparison, Mapping):
            raise ValueError("PCA comparison must be a mapping.")
        missing = sorted(
            set(_MATCHED_PCA_COMPARISON_FIELDS).difference(comparison)
        )
        raise ValueError(
            "PCA comparison is missing required scientific fields: "
            f"{missing}."
        )
    comparison = canonical
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
    Collect current full-validation direct-AE results.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
    
    Returns:
        results (list[dict[str, Any]]):
            Ordered current-configuration architecture, depth, latent,
            parameter, provenance, and validation records.
    """
    results: list[dict[str, Any]] = []
    for audit in audit_compressor_runs(project_root):
        if (
            audit["model_family"] != "Direct_AE"
            or not audit["eligible_for_comparison"]
        ):
            continue
        result = copy.deepcopy(audit["result"])
        result.pop("model_family")
        results.append(result)
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
    require_matched_pca_outperformance: bool = True,
) -> dict[str, Any] | None:
    """
    Select the smallest latent and then smallest parameter count meeting a target.
    
    Arguments:
        results (list[dict[str, Any]]):
            Records returned by collect_ae_results.
        target_variance_recovered (float):
            Minimum complete-validation variance recovery.
        require_matched_pca_outperformance (bool):
            Whether qualification also requires improvement over matched PCA
            in both validation variance recovery and log10 MSE.
    
    Returns:
        result (dict[str, Any] or None):
            Best qualified record, or None when no run reaches the target.
    """
    qualified = [
        result
        for result in results
        if result["variance_recovered"] >= target_variance_recovered
        and (
            not require_matched_pca_outperformance
            or result["pca_comparison"]["autoencoder_outperforms_pca"]
        )
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
    Collect current direct-AE and PCA-AE full-validation results.
    
    Arguments:
        project_root (str or pathlib.Path):
            IAFlowCloud repository root.
    
    Returns:
        results (list[dict[str, Any]]):
            Ordered current-configuration cross-family validation records.
    """
    results = [
        copy.deepcopy(audit["result"])
        for audit in audit_compressor_runs(project_root)
        if audit["eligible_for_comparison"]
    ]
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
    require_matched_pca_outperformance: bool = True,
) -> dict[str, Any] | None:
    """
    Select the smallest validated cross-family compressor meeting the target.
    
    Arguments:
        results (list[dict[str, Any]]):
            Records returned by collect_compressor_results.
        target_variance_recovered (float):
            Minimum complete-validation variance recovery.
        require_matched_pca_outperformance (bool):
            Whether qualification also requires improvement over matched PCA
            in both validation variance recovery and log10 MSE.
    
    Returns:
        result (dict[str, Any] or None):
            Smallest latent, then smallest parameter count, then lowest MSE.
    """
    qualified = [
        result
        for result in results
        if result["variance_recovered"] >= target_variance_recovered
        and (
            not require_matched_pca_outperformance
            or result["pca_comparison"]["autoencoder_outperforms_pca"]
        )
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
            Content-validated matched PCA metrics, absolute differences,
            fractional error reductions, variance gain, and outcome flag.
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
