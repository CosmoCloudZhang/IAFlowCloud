"""
Reusable shape-prior sampling utilities for the IA compression project.

The functions in this module keep prior design, physical-model evaluation,
diagnostics, stress tests, data splitting, and HDF5 serialization independent
of the notebooks.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import h5py
import numpy

from ..general.coordinates import validate_coordinate_grid
from ..general.data_split import build_dataset_split_indices, validate_dataset_split_indices
from .model import Z_STAR, NLAModel, pivot_redshift_ratio

__all__ = [
    "DEFAULT_SELECTION_CRITERIA",
    "assess_prior_summary",
    "build_boundary_parameter_samples",
    "build_corner_parameter_samples",
    "build_dataset_split_indices",
    "calculate_prior_diagnostics",
    "combine_positive_model_factors",
    "evaluate_prior_candidate",
    "generate_component_samples",
    "generate_unit_samples",
    "get_final_prior",
    "save_nuisance_dataset_in_batches",
    "summarize_prior",
    "transform_unit_samples",
    "validate_generated_dataset",
    "validate_prior_specification",
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


def get_final_prior():
    """
    Return a fresh copy of the frozen 13-parameter training prior.
    
    These are documented data-generation ranges, not observational constraints.
    ``A0`` is deliberately absent: it is an external analytic normalization and
    is therefore neither a sampled input nor a compression target.
    
    Returns:
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered nuisance-parameter bounds and sampling distributions.
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
    Validate parameter names, bounds, sampling rules, and canonical order.
    
    Arguments:
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered nuisance-parameter prior to validate.
    
    Returns:
        valid (bool):
            True when every prior entry satisfies the required contract.
    """
    expected_names = tuple(NLAModel.SHAPE_PARAMETER_NAMES)
    if tuple(prior_specification) != expected_names:
        raise ValueError(
            "The prior parameters must follow NLAModel.SHAPE_PARAMETER_NAMES."
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
    
    return True


# -----------------------------------------------------------------------------
# 2. Reproducible unit-cube sampling and transformation to parameter space
# -----------------------------------------------------------------------------

def generate_unit_samples(
    number_of_models,
    random_seed,
):
    """
    Draw reproducible unit-cube samples in the canonical parameter order.
    
    Arguments:
        number_of_models (int):
            Number of parameter rows to draw.
        random_seed (int):
            Seed for the NumPy random-number generator.
    
    Returns:
        unit_samples (numpy.ndarray):
            Unit-cube samples with shape (N_models, N_parameters).
    """
    number_of_models = int(number_of_models)
    if number_of_models <= 0:
        raise ValueError("number_of_models must be positive.")
    
    rng = numpy.random.default_rng(random_seed)
    return rng.random(
        (number_of_models, len(NLAModel.SHAPE_PARAMETER_NAMES))
    )


def transform_unit_samples(
    unit_samples,
    prior_specification,
):
    """
    Transform shared unit-cube samples into a requested nuisance prior.
    
    Arguments:
        unit_samples (numpy.ndarray):
            Samples in the unit hypercube, ordered by the canonical parameters.
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered bounds and distributions used for the transformation.
    
    Returns:
        parameter_samples (numpy.ndarray):
            Physical nuisance parameters with the same shape as unit_samples.
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


def _prior_midpoint(
    specification,
):
    """
    Return the natural midpoint of one uniform or log-uniform prior.
    
    Arguments:
        specification (dict[str, float or str]):
            Bounds and distribution for one nuisance parameter.
    
    Returns:
        midpoint (float):
            Arithmetic or geometric midpoint of the prior.
    """
    minimum = float(specification["minimum"])
    maximum = float(specification["maximum"])
    if specification["distribution"] == "log_uniform":
        return numpy.sqrt(minimum * maximum)
    return 0.5 * (minimum + maximum)


# -----------------------------------------------------------------------------
# 3. Deterministic boundary and full-corner stress sets
# -----------------------------------------------------------------------------

def build_boundary_parameter_samples(
    prior_specification,
    *,
    active_parameter_names=None,
):
    """
    Build midpoint and one-at-a-time boundaries for active parameters.
    
    By default all 13 parameters that change A_theta are varied. The returned
    set therefore contains ``1 + 2 * 13 = 27`` rows.
    
    Arguments:
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered nuisance-parameter prior.
        active_parameter_names (tuple[str, ...] or None):
            Parameters to move to their lower and upper bounds.
    
    Returns:
        boundary_samples (numpy.ndarray):
            Midpoint and one-at-a-time boundary parameter rows.
    """
    validate_prior_specification(prior_specification)
    if active_parameter_names is None:
        active_parameter_names = NLAModel.SHAPE_PARAMETER_NAMES
    active_parameter_names = tuple(active_parameter_names)
    invalid_names = set(active_parameter_names).difference(prior_specification)
    if invalid_names:
        raise ValueError(f"Unknown active parameters: {sorted(invalid_names)}.")
    
    midpoint = numpy.asarray(
        [
            _prior_midpoint(specification)
            for specification in prior_specification.values()
        ]
    )
    
    boundary_samples = [midpoint]
    parameter_names = list(prior_specification)
    for parameter_name in active_parameter_names:
        parameter_index = parameter_names.index(parameter_name)
        specification = prior_specification[parameter_name]
        lower_sample = midpoint.copy()
        upper_sample = midpoint.copy()
        lower_sample[parameter_index] = specification["minimum"]
        upper_sample[parameter_index] = specification["maximum"]
        boundary_samples.extend([lower_sample, upper_sample])
    
    return numpy.asarray(boundary_samples)


def build_corner_parameter_samples(
    prior_specification,
    *,
    active_parameter_names=None,
):
    """
    Build all simultaneous lower and upper active-parameter corners.
    
    By default the 13 A_theta shape parameters produce ``2**13 = 8192``
    deterministic stress-test models.
    
    Arguments:
        prior_specification (dict[str, dict[str, float or str]]):
            Ordered nuisance-parameter prior.
        active_parameter_names (tuple[str, ...] or None):
            Parameters varied simultaneously between their bounds.
    
    Returns:
        corner_samples (numpy.ndarray):
            Full-corner parameter rows in canonical order.
    """
    validate_prior_specification(prior_specification)
    if active_parameter_names is None:
        active_parameter_names = NLAModel.SHAPE_PARAMETER_NAMES
    active_parameter_names = tuple(active_parameter_names)
    invalid_names = set(active_parameter_names).difference(prior_specification)
    if invalid_names:
        raise ValueError(f"Unknown active parameters: {sorted(invalid_names)}.")
    if not active_parameter_names:
        raise ValueError("At least one active parameter is required.")
    
    parameter_names = list(prior_specification)
    midpoint = numpy.asarray(
        [
            _prior_midpoint(specification)
            for specification in prior_specification.values()
        ]
    )
    bound_pairs = [
        (
            float(prior_specification[name]["minimum"]),
            float(prior_specification[name]["maximum"]),
        )
        for name in active_parameter_names
    ]
    corner_values = list(product(*bound_pairs))
    corner_samples = numpy.repeat(
        midpoint[None, :],
        len(corner_values),
        axis=0,
    )
    for active_index, parameter_name in enumerate(active_parameter_names):
        parameter_index = parameter_names.index(parameter_name)
        corner_samples[:, parameter_index] = numpy.asarray(corner_values)[
            :, active_index
        ]
    return corner_samples


# -----------------------------------------------------------------------------
# 4. Physical-model evaluation: parameters become factors and surfaces
# -----------------------------------------------------------------------------

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
    parameter_samples = numpy.asarray(parameter_samples, dtype=float)
    z = validate_coordinate_grid(z, "z")
    k = validate_coordinate_grid(k, "k")
    
    expected_columns = len(NLAModel.SHAPE_PARAMETER_NAMES)
    if parameter_samples.ndim != 2 or parameter_samples.shape[1] != expected_columns:
        raise ValueError(
            f"parameter_samples must have shape (number_of_models, {expected_columns})."
        )
    if numpy.any(z <= -1.0):
        raise ValueError("z values must satisfy z > -1.")
    if numpy.any(k <= 0.0):
        raise ValueError("k values must be positive.")
    if not numpy.all(numpy.isfinite(parameter_samples)):
        raise ValueError("parameter_samples must contain finite values.")
    
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
            with numpy.errstate(over="raise", invalid="raise", divide="raise"):
                model = NLAModel.from_shape_array(values, z_star=z_star)
                model_components = {
                    "R_z": model.redshift_factor(z),
                    "R_L": model.luminosity_factor(z),
                    "S_k_z": model.scale_dependence(z, k),
                    "A_theta": model.model_amplitude(z, k),
                }
    
            for component_name, component_values in model_components.items():
                if not numpy.all(numpy.isfinite(component_values)):
                    raise ValueError(f"{component_name} contains non-finite values.")
    
            for component_name in ("R_z", "R_L", "S_k_z", "A_theta"):
                if not numpy.all(model_components[component_name] > 0.0):
                    raise ValueError(f"{component_name} must remain positive.")
    
            expected_A_theta = (
                model_components["R_z"][:, None]
                * model_components["R_L"][:, None]
                * model_components["S_k_z"]
            )
            if not numpy.allclose(model_components["A_theta"], expected_A_theta):
                raise ValueError("A_theta must equal R_z * R_L * S_k_z.")
    
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

def validate_generated_dataset(
    parameter_samples,
    component_samples,
    valid_mask,
    prior_specification,
    z,
    k,
):
    """
    Validate a generated nuisance-parameter dataset and its components.
    
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
    
    Returns:
        valid (bool):
            True when all array, prior, positivity, and factorization checks pass.
    """
    validate_prior_specification(prior_specification)
    parameter_samples = numpy.asarray(parameter_samples, dtype=float)
    valid_mask = numpy.asarray(valid_mask, dtype=bool)
    z = numpy.asarray(z, dtype=float)
    k = numpy.asarray(k, dtype=float)
    number_of_models = len(parameter_samples)
    
    expected_parameter_shape = (
        number_of_models,
        len(NLAModel.SHAPE_PARAMETER_NAMES),
    )
    expected_redshift_shape = (number_of_models, len(z))
    expected_surface_shape = (number_of_models, len(z), len(k))
    
    if parameter_samples.shape != expected_parameter_shape:
        raise ValueError(
            f"parameter_samples has shape {parameter_samples.shape}; "
            f"expected {expected_parameter_shape}."
        )
    if valid_mask.shape != (number_of_models,) or not numpy.all(valid_mask):
        raise ValueError("Every stored model must pass the validity checks.")
    
    expected_component_shapes = {
        "R_z": expected_redshift_shape,
        "R_L": expected_redshift_shape,
        "S_k_z": expected_surface_shape,
        "A_theta": expected_surface_shape,
    }
    if set(component_samples) != set(expected_component_shapes):
        raise ValueError(
            f"component_samples must contain {sorted(expected_component_shapes)}."
        )
    
    for component_name, expected_shape in expected_component_shapes.items():
        values = numpy.asarray(component_samples[component_name])
        if values.shape != expected_shape:
            raise ValueError(
                f"{component_name} has shape {values.shape}; expected {expected_shape}."
            )
        if not numpy.all(numpy.isfinite(values)):
            raise ValueError(f"{component_name} must contain only finite values.")
    
    for parameter_index, specification in enumerate(prior_specification.values()):
        values = parameter_samples[:, parameter_index]
        if numpy.any(values < specification["minimum"]) or numpy.any(
            values > specification["maximum"]
        ):
            raise ValueError("A sampled parameter lies outside its prior limits.")
    
    for component_name in ("R_z", "R_L", "S_k_z", "A_theta"):
        if not numpy.all(component_samples[component_name] > 0.0):
            raise ValueError(f"{component_name} must be strictly positive.")
    
    expected_A_theta = (
        component_samples["R_z"][:, :, None]
        * component_samples["R_L"][:, :, None]
        * component_samples["S_k_z"]
    )
    if not numpy.allclose(component_samples["A_theta"], expected_A_theta):
        raise ValueError("A_theta does not match R_z * R_L * S_k_z.")
    
    return True


# -----------------------------------------------------------------------------
# 6. Shape diagnostics and prior summaries
# -----------------------------------------------------------------------------

def combine_positive_model_factors(
    component_samples,
):
    """
    Reconstruct the positive A_theta = R_z * R_L * S_k_z surface.
    
    Arguments:
        component_samples (dict[str, numpy.ndarray]):
            Component arrays containing R_z, R_L, and S_k_z.
    
    Returns:
        A_theta (numpy.ndarray):
            Reconstructed positive surface for every model.
    """
    return (
        component_samples["R_z"][:, :, None]
        * component_samples["R_L"][:, :, None]
        * component_samples["S_k_z"]
    )


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
    A_theta = component_samples["A_theta"]
    reconstructed_A_theta = combine_positive_model_factors(component_samples)
    number_of_samples = len(A_theta)
    
    if sample_valid_mask.shape != (number_of_samples,):
        raise ValueError("sample_valid_mask has the wrong shape.")
    
    valid_indices = numpy.flatnonzero(sample_valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("At least one valid model is required for diagnostics.")
    
    minimum_A_theta = numpy.full(number_of_samples, numpy.nan)
    maximum_A_theta = numpy.full(number_of_samples, numpy.nan)
    log_dynamic_range = numpy.full(number_of_samples, numpy.nan)
    extreme_flag = numpy.zeros(number_of_samples, dtype=bool)
    
    valid_amplitudes = A_theta[valid_indices]
    valid_model_factors = reconstructed_A_theta[valid_indices]
    minimum_A_theta[valid_indices] = valid_amplitudes.min(axis=(1, 2))
    maximum_A_theta[valid_indices] = valid_amplitudes.max(axis=(1, 2))
    log_dynamic_range[valid_indices] = (
        numpy.log10(valid_model_factors.max(axis=(1, 2)))
        - numpy.log10(valid_model_factors.min(axis=(1, 2)))
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
    
    model_factors = combine_positive_model_factors(component_samples)
    valid_model_factors = model_factors[sample_valid_mask]
    shape_quantiles = numpy.quantile(
        numpy.log10(valid_model_factors),
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
            Ordered nuisance-parameter prior.
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
    parameter_samples = transform_unit_samples(unit_samples, prior_specification)
    components, valid_mask, failures = generate_component_samples(
        parameter_samples,
        z,
        k,
        z_star=z_star,
    )
    diagnostics = calculate_prior_diagnostics(components, valid_mask)
    
    corner_parameters = build_corner_parameter_samples(prior_specification)
    corner_components, corner_valid_mask, corner_failures = generate_component_samples(
        corner_parameters,
        z,
        k,
        z_star=z_star,
    )
    corner_diagnostics = calculate_prior_diagnostics(
        corner_components,
        corner_valid_mask,
    )
    summary = summarize_prior(
        candidate_name,
        components,
        valid_mask,
        diagnostics,
        corner_valid_mask=corner_valid_mask,
        corner_diagnostics=corner_diagnostics,
    )
    
    evaluation = {
        "prior": prior_specification,
        "parameters": parameter_samples,
        "components": components,
        "valid_mask": valid_mask,
        "failures": failures,
        "diagnostics": diagnostics,
        "corner_parameters": corner_parameters,
        "corner_valid_mask": corner_valid_mask,
        "corner_failures": corner_failures,
        "corner_diagnostics": corner_diagnostics,
        "summary": summary,
        "assessment": assess_prior_summary(summary, criteria=criteria),
    }
    if include_corner_components:
        evaluation["corner_components"] = corner_components
    
    return evaluation


# -----------------------------------------------------------------------------
# 7. Reproducible ML splits and HDF5 serialization
# -----------------------------------------------------------------------------


def save_nuisance_dataset_in_batches(
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
    overwrite=False,
):
    """
    Generate and save a large dataset without holding all surfaces in memory.
    
    Every batch is generated and strictly validated before it is written.  Both
    ``S_k_z`` and ``A_theta`` are retained in the schema, even though one can be
    reconstructed from the other factors.  The file is first written with a
    ``.partial`` suffix and atomically promoted only after every row succeeds.
    
    Arguments:
        output_file (str or pathlib.Path):
            Final HDF5 destination.
        parameter_samples (numpy.ndarray):
            Shape-parameter rows in canonical order.
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
        overwrite (bool):
            Whether to replace an existing final or partial output.
    
    Returns:
        output_path (pathlib.Path):
            Path to the atomically completed HDF5 dataset.
    """
    validate_prior_specification(prior_specification)
    parameter_samples = numpy.asarray(parameter_samples, dtype=float)
    z = validate_coordinate_grid(z, "z")
    k = validate_coordinate_grid(k, "k")
    storage_dtype = numpy.dtype(storage_dtype)
    batch_size = int(batch_size)
    
    number_of_models = len(parameter_samples)
    expected_parameter_shape = (
        number_of_models,
        len(NLAModel.SHAPE_PARAMETER_NAMES),
    )
    if parameter_samples.shape != expected_parameter_shape:
        raise ValueError(
            f"parameter_samples has shape {parameter_samples.shape}; "
            f"expected {expected_parameter_shape}."
        )
    if numpy.any(z <= -1.0):
        raise ValueError("z values must satisfy z > -1.")
    if numpy.any(k <= 0.0):
        raise ValueError("k values must be positive.")
    if storage_dtype.kind != "f":
        raise ValueError("storage_dtype must be a floating-point dtype.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not numpy.all(numpy.isfinite(parameter_samples)):
        raise ValueError("parameter_samples must contain finite values.")
    
    for parameter_index, specification in enumerate(prior_specification.values()):
        values = parameter_samples[:, parameter_index]
        if numpy.any(values < specification["minimum"]) or numpy.any(
            values > specification["maximum"]
        ):
            raise ValueError("A sampled parameter lies outside its prior limits.")
    
    if split_indices is not None:
        split_indices = validate_dataset_split_indices(
            split_indices,
            number_of_models,
        )
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_output_file = output_file.with_suffix(output_file.suffix + ".partial")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists; set overwrite=True to replace it: {output_file}")
    if partial_output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Partial output exists; inspect it or set overwrite=True: {partial_output_file}"
        )
    if overwrite:
        partial_output_file.unlink(missing_ok=True)
    parameter_names = list(NLAModel.SHAPE_PARAMETER_NAMES)
    r_star = pivot_redshift_ratio(z, z_star=z_star)
    compression_options = {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }
    
    with h5py.File(partial_output_file, "w") as dataset:
        dataset.attrs["sample_scope"] = (
            "13 A_theta shape parameters; A0 is external and not sampled"
        )
        dataset.attrs["normalization_parameter"] = "A0 (external; not stored)"
        dataset.attrs["shape_parameter_names"] = json.dumps(parameter_names)
        dataset.attrs["A_theta_definition"] = "R_z * R_L * S_k_z"
        dataset.attrs["A_IA_definition"] = "-A0 * A_omega * A_theta"
        dataset.attrs["prior_specification"] = json.dumps(prior_specification)
        dataset.attrs["metadata"] = json.dumps(metadata)
    
        coordinates_group = dataset.create_group("coordinates")
        coordinates_group.create_dataset("z", data=z)
        coordinates_group.create_dataset("k", data=k)
        coordinates_group.create_dataset("r_star", data=r_star)
    
        parameters_group = dataset.create_group("parameters")
        parameters_group.create_dataset(
            "values",
            data=parameter_samples.astype(storage_dtype),
            **compression_options,
        )
        string_type = h5py.string_dtype(encoding="utf-8")
        parameters_group.create_dataset(
            "names",
            data=numpy.asarray(parameter_names, dtype=object),
            dtype=string_type,
        )
    
        redshift_shape = (number_of_models, len(z))
        surface_shape = (number_of_models, len(z), len(k))
        components_group = dataset.create_group("components")
        component_datasets = {
            "R_z": components_group.create_dataset(
                "R_z", shape=redshift_shape, dtype=storage_dtype, **compression_options
            ),
            "R_L": components_group.create_dataset(
                "R_L", shape=redshift_shape, dtype=storage_dtype, **compression_options
            ),
            "S_k_z": components_group.create_dataset(
                "S_k_z", shape=surface_shape, dtype=storage_dtype, **compression_options
            ),
            "A_theta": components_group.create_dataset(
                "A_theta", shape=surface_shape, dtype=storage_dtype, **compression_options
            ),
        }
    
        diagnostics_group = dataset.create_group("diagnostics")
        diagnostics_group.attrs["component"] = "A_theta"
        diagnostic_datasets = {
            "minimum": diagnostics_group.create_dataset(
                "minimum", shape=(number_of_models,), dtype=storage_dtype
            ),
            "maximum": diagnostics_group.create_dataset(
                "maximum", shape=(number_of_models,), dtype=storage_dtype
            ),
            "log_dynamic_range": diagnostics_group.create_dataset(
                "log_dynamic_range", shape=(number_of_models,), dtype=storage_dtype
            ),
        }
    
        if split_indices is not None:
            splits_group = dataset.create_group("splits")
            for split_name in ("train", "validation", "test"):
                splits_group.create_dataset(
                    split_name,
                    data=numpy.asarray(split_indices[split_name], dtype=numpy.int64),
                    **compression_options,
                )
    
        for batch_start in range(0, number_of_models, batch_size):
            batch_stop = min(batch_start + batch_size, number_of_models)
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
    
            validate_generated_dataset(
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
            for diagnostic_name, source_name in {
                "minimum": "minimum_A_theta",
                "maximum": "maximum_A_theta",
                "log_dynamic_range": "log_dynamic_range",
            }.items():
                diagnostic_datasets[diagnostic_name][batch_slice] = batch_diagnostics[source_name]
    
        dynamic_ranges = diagnostic_datasets["log_dynamic_range"][:]
        extreme_threshold = numpy.quantile(dynamic_ranges, 0.99)
        diagnostics_group.create_dataset(
            "extreme_flag",
            data=dynamic_ranges >= extreme_threshold,
            **compression_options,
        )
    partial_output_file.replace(output_file)
    return output_file
