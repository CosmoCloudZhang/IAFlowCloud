"""
Reusable shape-prior sampling utilities for the IA compression project.

The functions in this module keep prior design, physical-model evaluation, diagnostics, stress tests, data splitting, and HDF5 serialization independent of the notebooks.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import h5py
import numpy

from ..utilities.coordinates import validate_coordinate_grid
from ..utilities.data_split import (
    SPLIT_NAMES,
    build_dataset_split_indices,
    validate_dataset_split_indices,
    validate_integer_scalar,
)
from .model import (
    Z_STAR,
    NLAModel,
    compose_shape_amplitude,
    pivot_redshift_ratio,
)

__all__ = [
    "DEFAULT_SELECTION_CRITERIA",
    "assess_prior_summary",
    "build_corner_parameter_samples",
    "build_dataset_split_indices",
    "calculate_prior_diagnostics",
    "evaluate_prior_candidate",
    "generate_component_samples",
    "generate_unit_samples",
    "get_shape_prior",
    "save_shape_dataset_in_batches",
    "summarize_prior",
    "transform_unit_samples",
    "validate_prior_specification",
    "validate_shape_sample_batch",
]


# -----------------------------------------------------------------------------
# 1. Canonical priors and the explicit decision rule
# -----------------------------------------------------------------------------

DEFAULT_SELECTION_CRITERIA = {
    "minimum_random_valid_fraction": 1.0,
    "minimum_corner_valid_fraction": 1.0,
    "maximum_random_q99_log_dynamic_range": 1.75,
    "maximum_corner_log_dynamic_range": 3.0,
    "minimum_median_pointwise_shape_spread": 1.0,
}


def get_shape_prior():
    """
    Return a fresh copy of the frozen shape-parameter training prior.
    
    These are documented data-generation ranges, not observational constraints.
    A0 is deliberately absent: it is an external analytic normalization and
    is therefore neither a sampled input nor a compression target.
    
    Returns:
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered shape-parameter bounds and sampling distributions.
    """
    return {
        "eta": {"minimum": -1.5, "maximum": 1.5, "distribution": "uniform"},
        "xi": {"minimum": -1.5, "maximum": 1.5, "distribution": "uniform"},
        "s": {"minimum": 1.0, "maximum": 10.0, "distribution": "uniform"},
        "z_q": {"minimum": 0.5, "maximum": 2.5, "distribution": "uniform"},
        "q": {"minimum": 0.0, "maximum": 2.0, "distribution": "uniform"},
        "n_star": {"minimum": 1.0, "maximum": 5.0, "distribution": "uniform"},
        "k_t_star": {"minimum": 0.1, "maximum": 1.0, "distribution": "log_uniform"},
        "alpha": {"minimum": -0.4, "maximum": 0.4, "distribution": "uniform"},
        "m": {"minimum": 1.0, "maximum": 4.0, "distribution": "uniform"},
        "gamma_t": {"minimum": -0.5, "maximum": 0.5, "distribution": "uniform"},
        "gamma_n": {"minimum": -0.35, "maximum": 0.35, "distribution": "uniform"},
        "gamma_alpha": {"minimum": -0.25, "maximum": 0.25, "distribution": "uniform"},
        "gamma_m": {"minimum": -0.2, "maximum": 0.2, "distribution": "uniform"},
    }


def validate_prior_specification(
    prior_specification,
):
    """
    Validate the complete ordered NLA shape prior and its sampling rules.
    
    Arguments:
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered parameter prior to validate.
    
    """
    if not prior_specification:
        raise ValueError("prior_specification cannot be empty.")
    
    prior_parameter_names = tuple(prior_specification)
    if not all(
        isinstance(name, str) and name for name in prior_parameter_names
    ):
        raise ValueError("Prior parameter names must be non-empty strings.")
    
    if prior_parameter_names != NLAModel.SHAPE_PARAMETER_NAMES:
        raise ValueError(
            "The prior must contain the complete ordered NLA shape-parameter "
            f"scope: {NLAModel.SHAPE_PARAMETER_NAMES}."
        )
    
    for parameter_name, specification in prior_specification.items():
        required_keys = {"minimum", "maximum", "distribution"}
        if set(specification) != required_keys:
            raise ValueError(
                f"{parameter_name} must define exactly {sorted(required_keys)}."
            )
        
        minimum = float(specification["minimum"])
        maximum = float(specification["maximum"])
        distribution = specification["distribution"]
        
        if not numpy.isfinite(minimum) or not numpy.isfinite(maximum):
            raise ValueError(f"{parameter_name} prior limits must be finite.")
        
        if minimum >= maximum:
            raise ValueError(f"{parameter_name} requires minimum < maximum.")
        
        if distribution not in {"uniform", "log_uniform"}:
            raise ValueError(f"Unsupported distribution for {parameter_name}.")
        
        if distribution == "log_uniform" and minimum <= 0.0:
            raise ValueError(
                f"{parameter_name} requires positive log-uniform bounds."
            )

# -----------------------------------------------------------------------------
# 2. Reproducible unit-cube sampling and transformation to parameter space
# -----------------------------------------------------------------------------

def generate_unit_samples(
    number_of_models,
    random_seed,
    number_of_parameters,
):
    """
    Draw reproducible samples from a unit hypercube.
    
    Arguments:
        number_of_models (int):
            Number of parameter rows to draw.
        random_seed (int):
            Seed for the NumPy random-number generator.
        number_of_parameters (int):
            Number of sampled parameter dimensions.
    
    Returns:
        unit_samples (numpy.ndarray):
            Unit-cube samples with shape (N_models, N_parameters).
    """
    number_of_models = validate_integer_scalar(number_of_models, "number_of_models")
    number_of_parameters = validate_integer_scalar(
        number_of_parameters,
        "number_of_parameters",
    )
    
    if number_of_models <= 0:
        raise ValueError("number_of_models must be positive.")
    
    if number_of_parameters <= 0:
        raise ValueError("number_of_parameters must be positive.")
    
    rng = numpy.random.default_rng(random_seed)
    return rng.random(
        (number_of_models, number_of_parameters)
    )


def transform_unit_samples(
    unit_samples,
    prior_specification,
):
    """
    Transform shared unit-cube samples into a requested parameter prior.
    
    Arguments:
        unit_samples (numpy.ndarray):
            Samples in the unit hypercube, ordered like prior_specification.
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered bounds and distributions used for the transformation.
    
    Returns:
        parameter_samples (numpy.ndarray):
            Physical parameters with the same shape as unit_samples.
    """
    validate_prior_specification(prior_specification)
    unit_samples = numpy.asarray(unit_samples, dtype=float)
    expected_columns = len(prior_specification)
    
    if unit_samples.ndim != 2 or unit_samples.shape[1] != expected_columns:
        raise ValueError(
            f"unit_samples must have shape (number_of_models, {expected_columns})."
        )
    
    if not numpy.all(numpy.isfinite(unit_samples)):
        raise ValueError("unit_samples must contain finite values.")
    
    if numpy.any((unit_samples < 0.0) | (unit_samples > 1.0)):
        raise ValueError("unit_samples must lie between zero and one.")
    
    parameter_samples = numpy.empty_like(unit_samples)
    
    for parameter_index, specification in enumerate(prior_specification.values()):
        minimum = float(specification["minimum"])
        maximum = float(specification["maximum"])
        
        if specification["distribution"] == "uniform":
            parameter_samples[:, parameter_index] = (
                minimum
                + unit_samples[:, parameter_index] * (maximum - minimum)
            )
        else:
            log_minimum = numpy.log10(minimum)
            log_maximum = numpy.log10(maximum)
            parameter_samples[:, parameter_index] = 10.0 ** (
                log_minimum
                + unit_samples[:, parameter_index]
                * (log_maximum - log_minimum)
            )
    
    return parameter_samples


# -----------------------------------------------------------------------------
# 3. Deterministic full-corner stress set
# -----------------------------------------------------------------------------

def build_corner_parameter_samples(
    prior_specification,
):
    """
    Build every simultaneous lower and upper shape-prior corner.
    
    The complete ordered shape prior produces 2**N_parameters deterministic
    stress-test models.
    
    Arguments:
        prior_specification (dict[str, dict[str, float or str]]):
            Complete ordered NLA shape prior.
    
    Returns:
        corner_samples (numpy.ndarray):
            Full-corner parameter rows in prior-specification order.
    """
    validate_prior_specification(prior_specification)
    bound_pairs = [
        (
            float(specification["minimum"]),
            float(specification["maximum"]),
        )
        for specification in prior_specification.values()
    ]
    return numpy.asarray(list(product(*bound_pairs)), dtype=float)


# -----------------------------------------------------------------------------
# 4. Physical-model evaluation: parameters become factors and surfaces
# -----------------------------------------------------------------------------

def _validate_sampling_coordinates(
    z,
    k,
):
    """
    Validate and return the coordinate grids used for sampled surfaces.
    
    Arguments:
        z (numpy.ndarray):
            Candidate redshift grid.
        k (numpy.ndarray):
            Candidate wavenumber grid in Mpc^-1.
    
    Returns:
        coordinates (tuple[numpy.ndarray, numpy.ndarray]):
            Strictly increasing redshift and positive wavenumber grids.
    """
    z = validate_coordinate_grid(z, "z")
    k = validate_coordinate_grid(k, "k")
    
    if numpy.any(z <= -1.0):
        raise ValueError("z values must satisfy z > -1.")
    
    if numpy.any(k <= 0.0):
        raise ValueError("k values must be positive.")
    
    return z, k


def _validate_shape_parameter_samples(
    parameter_samples,
    *,
    prior_specification=None,
):
    """
    Validate and return a canonical matrix of NLA shape parameters.
    
    Arguments:
        parameter_samples (numpy.ndarray):
            Candidate shape-parameter rows.
        prior_specification (dict[str, dict[str, float or str]] or None):
            Optional prior whose bounds and parameter order must be satisfied.
    
    Returns:
        samples (numpy.ndarray):
            Finite shape-parameter matrix in canonical order.
    """
    parameter_samples = numpy.asarray(parameter_samples, dtype=float)
    expected_columns = len(NLAModel.SHAPE_PARAMETER_NAMES)
    
    if (
        parameter_samples.ndim != 2
        or parameter_samples.shape[0] == 0
        or parameter_samples.shape[1] != expected_columns
    ):
        raise ValueError(
            "parameter_samples must have non-empty shape "
            f"(number_of_models, {expected_columns})."
        )
    
    if not numpy.all(numpy.isfinite(parameter_samples)):
        raise ValueError("parameter_samples must contain finite values.")
    
    if prior_specification is not None:
        validate_prior_specification(prior_specification)
        for parameter_index, specification in enumerate(
            prior_specification.values()
        ):
            values = parameter_samples[:, parameter_index]
            if numpy.any(values < specification["minimum"]) or numpy.any(
                values > specification["maximum"]
            ):
                raise ValueError("A sampled parameter lies outside its prior limits.")
    
    return parameter_samples


def generate_component_samples(
    parameter_samples,
    z,
    k,
    *,
    z_star=Z_STAR,
):
    """
    Generate R_z, R_L, S_k_z, and A_theta for every parameter row.
    
    Arguments:
        parameter_samples (numpy.ndarray):
            Shape-parameter rows in NLAModel.SHAPE_PARAMETER_NAMES order.
        z (numpy.ndarray):
            Strictly increasing redshift grid.
        k (numpy.ndarray):
            Strictly increasing positive wavenumber grid in Mpc^-1.
        z_star (float or int):
            Pivot redshift shared by every generated model.
    
    Returns:
        result (tuple[dict, numpy.ndarray, dict]):
            Component arrays, validity mask, and failure messages keyed by row.
    """
    parameter_samples = _validate_shape_parameter_samples(parameter_samples)
    z, k = _validate_sampling_coordinates(z, k)
    
    number_of_samples = len(parameter_samples)
    redshift_shape = (number_of_samples, len(z))
    surface_shape = (number_of_samples, len(z), len(k))
    components = {
        "R_z": numpy.full(redshift_shape, numpy.nan),
        "R_L": numpy.full(redshift_shape, numpy.nan),
        "S_k_z": numpy.full(surface_shape, numpy.nan),
        "A_theta": numpy.full(surface_shape, numpy.nan),
    }
    valid_mask = numpy.zeros(number_of_samples, dtype=bool)
    failure_messages = {}
    for model_index, values in enumerate(parameter_samples):
        try:
            model = NLAModel.from_shape_array(values, z_star=z_star)
            model_components = model.shape_components(z, k)
            
            for component_name in components:  # noqa: PLC0206
                components[component_name][model_index] = model_components[
                    component_name
                ]
            valid_mask[model_index] = True
        
        except (FloatingPointError, TypeError, ValueError) as error:
            failure_messages[model_index] = str(error)
    
    return components, valid_mask, failure_messages


# -----------------------------------------------------------------------------
# 5. Strict validation of parameter rows and physical surface tensors
# -----------------------------------------------------------------------------

def _expected_component_shapes(
    number_of_models,
    z,
    k,
):
    """
    Return the required shapes of every sampled physical component.
    
    Arguments:
        number_of_models (int):
            Number of sampled parameter rows.
        z (numpy.ndarray):
            Redshift grid.
        k (numpy.ndarray):
            Wavenumber grid.
    
    Returns:
        shapes (dict[str, tuple[int, ...]]):
            Expected array shape for each physical component.
    """
    redshift_shape = (number_of_models, len(z))
    surface_shape = (number_of_models, len(z), len(k))
    return {
        "R_z": redshift_shape,
        "R_L": redshift_shape,
        "S_k_z": surface_shape,
        "A_theta": surface_shape,
    }


def _validate_shape_components(
    component_samples,
    expected_component_shapes,
):
    """
    Validate and return a complete collection of sampled shape components.
    
    Arguments:
        component_samples (dict[str, numpy.ndarray]):
            Candidate R_z, R_L, S_k_z, and A_theta arrays.
        expected_component_shapes (dict[str, tuple[int, ...]]):
            Required shape for each component.
    
    Returns:
        components (dict[str, numpy.ndarray]):
            Validated finite and positive component arrays.
    """
    if set(component_samples) != set(expected_component_shapes):
        raise ValueError(
            f"component_samples must contain {sorted(expected_component_shapes)}."
        )
    
    components = {}
    for component_name, expected_shape in expected_component_shapes.items():
        values = numpy.asarray(component_samples[component_name])
        if values.shape != expected_shape:
            raise ValueError(
                f"{component_name} has shape {values.shape}; expected {expected_shape}."
            )
        
        if not numpy.all(numpy.isfinite(values)):
            raise ValueError(f"{component_name} must contain only finite values.")
        
        if not numpy.all(values > 0.0):
            raise ValueError(f"{component_name} must be strictly positive.")
        
        components[component_name] = values
    
    return components


def validate_shape_sample_batch(
    parameter_samples,
    component_samples,
    valid_mask,
    prior_specification,
    z,
    k,
):
    """
    Validate a persistable batch of shape parameters and physical components.
    
    Arguments:
        parameter_samples (numpy.ndarray):
            Shape-parameter rows in canonical order.
        component_samples (dict[str, numpy.ndarray]):
            Generated R_z, R_L, S_k_z, and A_theta arrays.
        valid_mask (numpy.ndarray):
            Boolean validity flag for every parameter row.
        prior_specification (dict[str, dict[str, float or str]]):
            Prior used to generate the parameter rows.
        z (numpy.ndarray):
            Redshift grid used for the components.
        k (numpy.ndarray):
            Wavenumber grid used for the components.
    
    """
    parameter_samples = _validate_shape_parameter_samples(
        parameter_samples,
        prior_specification=prior_specification,
    )
    valid_mask = numpy.asarray(valid_mask, dtype=bool)
    z, k = _validate_sampling_coordinates(z, k)
    number_of_models = len(parameter_samples)
    
    if valid_mask.shape != (number_of_models,) or not numpy.all(valid_mask):
        raise ValueError("Every stored model must pass the validity checks.")
    
    expected_shapes = _expected_component_shapes(number_of_models, z, k)
    components = _validate_shape_components(
        component_samples,
        expected_shapes,
    )
    expected_A_theta = compose_shape_amplitude(
        components["R_z"],
        components["R_L"],
        components["S_k_z"],
    )
    if not numpy.allclose(component_samples["A_theta"], expected_A_theta):
        raise ValueError("A_theta does not match R_z * R_L * S_k_z.")


# -----------------------------------------------------------------------------
# 6. Shape diagnostics and prior summaries
# -----------------------------------------------------------------------------


def calculate_prior_diagnostics(
    component_samples,
    sample_valid_mask,
):
    """
    Calculate diagnostics for the positive A_theta shape surface.
    
    Arguments:
        component_samples (dict[str, numpy.ndarray]):
            Generated component arrays for a collection of models.
        sample_valid_mask (numpy.ndarray):
            Boolean flag identifying successfully generated models.
    
    Returns:
        diagnostics (dict[str, numpy.ndarray]):
            Per-model amplitude limits, dynamic ranges, and extreme flags.
    """
    sample_valid_mask = numpy.asarray(sample_valid_mask, dtype=bool)
    A_theta = numpy.asarray(component_samples["A_theta"], dtype=float)
    number_of_samples = len(A_theta)
    
    if sample_valid_mask.shape != (number_of_samples,):
        raise ValueError("sample_valid_mask has the wrong shape.")
    
    valid_indices = numpy.flatnonzero(sample_valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("At least one valid model is required for diagnostics.")
    
    valid_amplitudes = A_theta[valid_indices]
    if not numpy.all(numpy.isfinite(valid_amplitudes)):
        raise ValueError("Valid A_theta samples must contain only finite values.")
    
    if numpy.any(valid_amplitudes <= 0.0):
        raise ValueError("Valid A_theta samples must be strictly positive.")
    
    minimum_A_theta = numpy.full(number_of_samples, numpy.nan)
    maximum_A_theta = numpy.full(number_of_samples, numpy.nan)
    log_dynamic_range = numpy.full(number_of_samples, numpy.nan)
    extreme_flag = numpy.zeros(number_of_samples, dtype=bool)
    
    minimum_A_theta[valid_indices] = valid_amplitudes.min(axis=(1, 2))
    maximum_A_theta[valid_indices] = valid_amplitudes.max(axis=(1, 2))
    log_dynamic_range[valid_indices] = (
        numpy.log10(maximum_A_theta[valid_indices])
        - numpy.log10(minimum_A_theta[valid_indices])
    )
    
    extreme_threshold = numpy.quantile(log_dynamic_range[valid_indices], 0.99)
    extreme_flag[valid_indices] = (
        log_dynamic_range[valid_indices] >= extreme_threshold
    )
    
    return {
        "minimum_A_theta": minimum_A_theta,
        "maximum_A_theta": maximum_A_theta,
        "log_dynamic_range": log_dynamic_range,
        "extreme_flag": extreme_flag,
    }


def summarize_prior(
    candidate_name,
    component_samples,
    sample_valid_mask,
    diagnostics,
    *,
    corner_valid_mask=None,
    corner_diagnostics=None,
):
    """
    Summarize the comparison metrics used to select one training prior.
    
    Arguments:
        candidate_name (str):
            Human-readable name of the candidate prior.
        component_samples (dict[str, numpy.ndarray]):
            Components generated from random samples.
        sample_valid_mask (numpy.ndarray):
            Validity mask for the random samples.
        diagnostics (dict[str, numpy.ndarray]):
            Diagnostics calculated for the random samples.
        corner_valid_mask (numpy.ndarray or None):
            Optional validity mask for full-corner samples.
        corner_diagnostics (dict[str, numpy.ndarray] or None):
            Optional diagnostics for full-corner samples.
    
    Returns:
        summary (dict[str, float or int or str]):
            Scalar validity, dynamic-range, and shape-diversity statistics.
    """
    sample_valid_mask = numpy.asarray(sample_valid_mask, dtype=bool)
    valid_ranges = diagnostics["log_dynamic_range"][sample_valid_mask]
    if len(valid_ranges) == 0:
        raise ValueError("At least one valid random sample is required.")
    
    A_theta = numpy.asarray(component_samples["A_theta"], dtype=float)
    valid_A_theta = A_theta[sample_valid_mask]
    shape_quantiles = numpy.quantile(
        numpy.log10(valid_A_theta),
        [0.01, 0.99],
        axis=0,
    )
    pointwise_shape_spread = shape_quantiles[1] - shape_quantiles[0]
    
    summary = {
        "candidate": str(candidate_name),
        "number_of_random_models": len(sample_valid_mask),
        "valid_fraction": float(sample_valid_mask.mean()),
        "median_log_dynamic_range": float(numpy.median(valid_ranges)),
        "q99_log_dynamic_range": float(numpy.quantile(valid_ranges, 0.99)),
        "maximum_log_dynamic_range": float(numpy.max(valid_ranges)),
        "median_pointwise_shape_spread": float(
            numpy.median(pointwise_shape_spread)
        ),
        "maximum_pointwise_shape_spread": float(
            numpy.max(pointwise_shape_spread)
        ),
    }
    
    if corner_valid_mask is not None or corner_diagnostics is not None:
        if corner_valid_mask is None or corner_diagnostics is None:
            raise ValueError(
                "corner_valid_mask and corner_diagnostics must be supplied together."
            )
        corner_valid_mask = numpy.asarray(corner_valid_mask, dtype=bool)
        valid_corner_ranges = corner_diagnostics["log_dynamic_range"][
            corner_valid_mask
        ]
        if len(valid_corner_ranges) == 0:
            raise ValueError("At least one valid corner model is required.")
        summary.update(
            {
                "number_of_corner_models": len(corner_valid_mask),
                "corner_valid_fraction": float(corner_valid_mask.mean()),
                "corner_q99_log_dynamic_range": float(
                    numpy.quantile(valid_corner_ranges, 0.99)
                ),
                "corner_maximum_log_dynamic_range": float(
                    numpy.max(valid_corner_ranges)
                ),
            }
        )
    
    return summary


def assess_prior_summary(
    summary,
    criteria=None,
):
    """
    Apply explicit training-prior selection gates to a prior summary.
    
    Arguments:
        summary (dict[str, float]):
            Scalar diagnostics returned by summarize_prior.
        criteria (dict[str, float] or None):
            Optional replacement for DEFAULT_SELECTION_CRITERIA.
    
    Returns:
        checks (dict[str, bool]):
            Individual gate results and the combined accepted decision.
    """
    if criteria is None:
        criteria = DEFAULT_SELECTION_CRITERIA
    
    required_summary_keys = {
        "valid_fraction",
        "corner_valid_fraction",
        "q99_log_dynamic_range",
        "corner_maximum_log_dynamic_range",
        "median_pointwise_shape_spread",
    }
    missing_keys = required_summary_keys.difference(summary)
    if missing_keys:
        raise ValueError(f"Prior summary is missing {sorted(missing_keys)}.")
    
    checks = {
        "random_validity": (
            summary["valid_fraction"]
            >= criteria["minimum_random_valid_fraction"]
        ),
        "corner_validity": (
            summary["corner_valid_fraction"]
            >= criteria["minimum_corner_valid_fraction"]
        ),
        "random_tail_control": (
            summary["q99_log_dynamic_range"]
            <= criteria["maximum_random_q99_log_dynamic_range"]
        ),
        "corner_tail_control": (
            summary["corner_maximum_log_dynamic_range"]
            <= criteria["maximum_corner_log_dynamic_range"]
        ),
        "shape_diversity": (
            summary["median_pointwise_shape_spread"]
            >= criteria["minimum_median_pointwise_shape_spread"]
        ),
    }
    checks["accepted"] = bool(all(checks.values()))
    return {name: bool(value) for name, value in checks.items()}


def _evaluate_parameter_samples(
    parameter_samples,
    z,
    k,
    *,
    z_star,
):
    """
    Generate components and diagnostics for one parameter collection.
    
    Arguments:
        parameter_samples (numpy.ndarray):
            Shape-parameter rows in canonical order.
        z (numpy.ndarray):
            Redshift evaluation grid.
        k (numpy.ndarray):
            Wavenumber evaluation grid in Mpc^-1.
        z_star (float or int):
            Pivot redshift used by every model.
    
    Returns:
        evaluation (dict[str, object]):
            Components, validity mask, failures, and diagnostics.
    """
    components, valid_mask, failures = generate_component_samples(
        parameter_samples,
        z,
        k,
        z_star=z_star,
    )
    diagnostics = calculate_prior_diagnostics(components, valid_mask)
    return {
        "components": components,
        "valid_mask": valid_mask,
        "failures": failures,
        "diagnostics": diagnostics,
    }


def evaluate_prior_candidate(
    candidate_name,
    prior_specification,
    unit_samples,
    z,
    k,
    *,
    z_star=Z_STAR,
    include_corner_components=False,
    criteria=None,
):
    """
    Evaluate a candidate prior with random samples and all full corners.
    
    Arguments:
        candidate_name (str):
            Human-readable candidate name.
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered shape-parameter prior.
        unit_samples (numpy.ndarray):
            Shared unit-cube random samples.
        z (numpy.ndarray):
            Redshift evaluation grid.
        k (numpy.ndarray):
            Wavenumber evaluation grid in Mpc^-1.
        z_star (float or int):
            Pivot redshift used by every model.
        include_corner_components (bool):
            Whether to retain the large corner-component arrays in the result.
        criteria (dict[str, float] or None):
            Optional prior-selection criteria.
    
    Returns:
        evaluation (dict[str, object]):
            Generated samples, diagnostics, summary, and acceptance checks.
    """
    validate_prior_specification(prior_specification)
    parameter_samples = transform_unit_samples(unit_samples, prior_specification)
    random_evaluation = _evaluate_parameter_samples(
        parameter_samples,
        z,
        k,
        z_star=z_star,
    )
    
    corner_parameters = build_corner_parameter_samples(prior_specification)
    corner_evaluation = _evaluate_parameter_samples(
        corner_parameters,
        z,
        k,
        z_star=z_star,
    )
    summary = summarize_prior(
        candidate_name,
        random_evaluation["components"],
        random_evaluation["valid_mask"],
        random_evaluation["diagnostics"],
        corner_valid_mask=corner_evaluation["valid_mask"],
        corner_diagnostics=corner_evaluation["diagnostics"],
    )
    
    evaluation = {
        "prior": prior_specification,
        "parameters": parameter_samples,
        **random_evaluation,
        "corner_parameters": corner_parameters,
        "corner_valid_mask": corner_evaluation["valid_mask"],
        "corner_failures": corner_evaluation["failures"],
        "corner_diagnostics": corner_evaluation["diagnostics"],
        "summary": summary,
        "assessment": assess_prior_summary(summary, criteria=criteria),
    }
    if include_corner_components:
        evaluation["corner_components"] = corner_evaluation["components"]
    
    return evaluation


# -----------------------------------------------------------------------------
# 7. Reproducible ML splits and HDF5 serialization
# -----------------------------------------------------------------------------

def _write_shape_dataset_metadata(
    dataset,
    shape_parameter_names,
    prior_specification,
    metadata,
):
    """
    Write the scientific parameter scope and generation metadata.
    
    Arguments:
        dataset (h5py.File):
            Open output dataset.
        shape_parameter_names (tuple[str, ...]):
            Canonical shape-parameter names.
        prior_specification (dict[str, dict[str, float or str]]):
            Prior used to generate the parameter rows.
        metadata (dict):
            Scientific generation metadata.
    """
    dataset.attrs["sample_scope"] = (
        f"{len(shape_parameter_names)} A_theta shape parameters; "
        "A0 is external and not sampled"
    )
    dataset.attrs["normalization_parameter"] = "A0 (external; not stored)"
    dataset.attrs["shape_parameter_names"] = json.dumps(shape_parameter_names)
    dataset.attrs["A_theta_definition"] = "R_z * R_L * S_k_z"
    dataset.attrs["A_IA_definition"] = "-A0 * A_omega * A_theta"
    dataset.attrs["prior_specification"] = json.dumps(prior_specification)
    dataset.attrs["metadata"] = json.dumps(metadata)


def _create_shape_dataset_structure(
    dataset,
    parameter_samples,
    shape_parameter_names,
    z,
    k,
    z_star,
    split_indices,
    storage_dtype,
):
    """
    Create the complete HDF5 structure before sampled batches are written.
    
    Arguments:
        dataset (h5py.File):
            Open output dataset.
        parameter_samples (numpy.ndarray):
            Complete shape-parameter matrix.
        shape_parameter_names (tuple[str, ...]):
            Canonical parameter names.
        z (numpy.ndarray):
            Redshift grid.
        k (numpy.ndarray):
            Wavenumber grid in Mpc^-1.
        z_star (float or int):
            Pivot redshift shared by every model.
        split_indices (dict[str, numpy.ndarray] or None):
            Optional complete sample partition.
        storage_dtype (numpy.dtype):
            Floating-point dtype used for stored arrays.
    
    Returns:
        handles (tuple[dict, dict, dict]):
            Component datasets, diagnostic datasets, and compression options.
    """
    compression_options = {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }
    coordinates_group = dataset.create_group("coordinates")
    coordinates_group.create_dataset("z", data=z)
    coordinates_group.create_dataset("k", data=k)
    coordinates_group.create_dataset(
        "r_star",
        data=pivot_redshift_ratio(z, z_star=z_star),
    )
    
    parameters_group = dataset.create_group("parameters")
    parameters_group.create_dataset(
        "values",
        data=parameter_samples.astype(storage_dtype),
        **compression_options,
    )
    string_type = h5py.string_dtype(encoding="utf-8")
    parameters_group.create_dataset(
        "names",
        data=numpy.asarray(shape_parameter_names, dtype=object),
        dtype=string_type,
    )
    
    expected_shapes = _expected_component_shapes(len(parameter_samples), z, k)
    components_group = dataset.create_group("components")
    component_datasets = {
        name: components_group.create_dataset(
            name,
            shape=shape,
            dtype=storage_dtype,
            **compression_options,
        )
        for name, shape in expected_shapes.items()
    }
    
    diagnostics_group = dataset.create_group("diagnostics")
    diagnostics_group.attrs["component"] = "A_theta"
    diagnostic_datasets = {
        "minimum": diagnostics_group.create_dataset(
            "minimum",
            shape=(len(parameter_samples),),
            dtype=storage_dtype,
        ),
        "maximum": diagnostics_group.create_dataset(
            "maximum",
            shape=(len(parameter_samples),),
            dtype=storage_dtype,
        ),
        "log_dynamic_range": diagnostics_group.create_dataset(
            "log_dynamic_range",
            shape=(len(parameter_samples),),
            dtype=storage_dtype,
        ),
    }
    
    if split_indices is not None:
        splits_group = dataset.create_group("splits")
        for split_name in SPLIT_NAMES:
            splits_group.create_dataset(
                split_name,
                data=numpy.asarray(split_indices[split_name], dtype=numpy.int64),
                **compression_options,
            )
    
    return component_datasets, diagnostic_datasets, compression_options


def _write_shape_dataset_batch(
    parameter_samples,
    prior_specification,
    z,
    k,
    z_star,
    batch_start,
    batch_stop,
    component_datasets,
    diagnostic_datasets,
):
    """
    Generate, validate, and write one bounded-memory sample batch.
    
    Arguments:
        parameter_samples (numpy.ndarray):
            Complete shape-parameter matrix.
        prior_specification (dict[str, dict[str, float or str]]):
            Prior used to generate the parameter rows.
        z (numpy.ndarray):
            Redshift grid.
        k (numpy.ndarray):
            Wavenumber grid in Mpc^-1.
        z_star (float or int):
            Pivot redshift shared by every model.
        batch_start (int):
            Inclusive source-row offset.
        batch_stop (int):
            Exclusive source-row offset.
        component_datasets (dict[str, h5py.Dataset]):
            Writable component datasets.
        diagnostic_datasets (dict[str, h5py.Dataset]):
            Writable diagnostic datasets.
    """
    batch_parameters = parameter_samples[batch_start:batch_stop]
    batch_components, batch_valid_mask, batch_failures = (
        generate_component_samples(
            batch_parameters,
            z,
            k,
            z_star=z_star,
        )
    )
    if batch_failures:
        first_index, message = next(iter(batch_failures.items()))
        absolute_index = batch_start + first_index
        raise RuntimeError(
            f"Model {absolute_index} failed during generation: {message}"
        )
    
    validate_shape_sample_batch(
        batch_parameters,
        batch_components,
        batch_valid_mask,
        prior_specification,
        z,
        k,
    )
    batch_diagnostics = calculate_prior_diagnostics(
        batch_components,
        batch_valid_mask,
    )
    batch_slice = slice(batch_start, batch_stop)
    for component_name, component_values in batch_components.items():
        component_datasets[component_name][batch_slice] = component_values
    
    diagnostic_sources = {
        "minimum": "minimum_A_theta",
        "maximum": "maximum_A_theta",
        "log_dynamic_range": "log_dynamic_range",
    }
    for diagnostic_name, source_name in diagnostic_sources.items():
        diagnostic_datasets[diagnostic_name][batch_slice] = batch_diagnostics[
            source_name
        ]


def save_shape_dataset_in_batches(
    output_file,
    parameter_samples,
    prior_specification,
    metadata,
    *,
    z,
    k,
    split_indices=None,
    z_star=Z_STAR,
    storage_dtype="float32",
    batch_size=2048,
):
    """
    Generate and save a large shape-surface dataset in bounded memory.
    
    Every batch is generated and strictly validated before it is written. Both
    S_k_z and A_theta are retained in the schema, even though one can be
    reconstructed from the other factors. The file is first written with a
    .partial suffix and atomically promoted only after every row succeeds.
    
    Arguments:
        output_file (str or pathlib.Path):
            Final HDF5 destination.
        parameter_samples (numpy.ndarray):
            Shape-parameter rows in prior-specification order.
        prior_specification (dict[str, dict[str, float or str]]):
            Prior used to generate parameter_samples.
        metadata (dict):
            Scientific generation metadata stored in the HDF5 attributes.
        z (numpy.ndarray):
            Redshift coordinate grid.
        k (numpy.ndarray):
            Wavenumber coordinate grid in Mpc^-1.
        split_indices (dict[str, numpy.ndarray] or None):
            Optional complete train, validation, and test partition.
        z_star (float or int):
            Pivot redshift used for every generated model.
        storage_dtype (str or numpy.dtype):
            Floating-point dtype used for stored numerical arrays.
        batch_size (int):
            Number of models generated and validated per batch.
    
    Returns:
        output_path (pathlib.Path):
            Path to the atomically completed HDF5 dataset.
    """
    parameter_samples = _validate_shape_parameter_samples(
        parameter_samples,
        prior_specification=prior_specification,
    )
    z, k = _validate_sampling_coordinates(z, k)
    storage_dtype = numpy.dtype(storage_dtype)
    batch_size = validate_integer_scalar(batch_size, "batch_size")
    
    number_of_models = len(parameter_samples)
    if storage_dtype.kind != "f":
        raise ValueError("storage_dtype must be a floating-point dtype.")
    
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    
    if split_indices is not None:
        split_indices = validate_dataset_split_indices(
            split_indices,
            number_of_models,
        )
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_output_file = output_file.with_suffix(output_file.suffix + ".partial")
    partial_output_file.unlink(missing_ok=True)
    shape_parameter_names = tuple(NLAModel.SHAPE_PARAMETER_NAMES)
    
    with h5py.File(partial_output_file, "w") as dataset:
        _write_shape_dataset_metadata(
            dataset,
            shape_parameter_names,
            prior_specification,
            metadata,
        )
        (
            component_datasets,
            diagnostic_datasets,
            compression_options,
        ) = _create_shape_dataset_structure(
            dataset,
            parameter_samples,
            shape_parameter_names,
            z,
            k,
            z_star,
            split_indices,
            storage_dtype,
        )
        
        for batch_start in range(0, number_of_models, batch_size):
            batch_stop = min(batch_start + batch_size, number_of_models)
            _write_shape_dataset_batch(
                parameter_samples,
                prior_specification,
                z,
                k,
                z_star,
                batch_start,
                batch_stop,
                component_datasets,
                diagnostic_datasets,
            )
        
        dynamic_ranges = diagnostic_datasets["log_dynamic_range"][:]
        extreme_threshold = numpy.quantile(dynamic_ranges, 0.99)
        dataset["diagnostics"].create_dataset(
            "extreme_flag",
            data=dynamic_ranges >= extreme_threshold,
            **compression_options,
        )
    partial_output_file.replace(output_file)
    return output_file
