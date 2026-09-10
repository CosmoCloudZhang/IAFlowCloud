"""
Shared visual conventions for validation-only compressor figures.

The module owns presentation mechanics only. Notebook code remains responsible
for artifact authentication, scientific selection, and preparation of plotted
values.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy
from matplotlib import pyplot
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

FAMILY_COLORS = {
    "PCA": "#111111",
    "PCA_AE": "#7B2CBF",
    "Conv1D": "#0072B2",
    "Conv2D": "#D55E00",
}

DEPTH_LINESTYLES = {
    "Depth03": "-",
    "Depth04": "--",
    "Depth05": ":",
}

DEPTH_MARKERS = {
    "Depth03": "o",
    "Depth04": "s",
    "Depth05": "^",
}

SELECTION_MARKERS = {
    "minimum_adequate": "*",
    "best_observed_accuracy": "X",
}

FIGURE_SIZES = {
    "summary": (12.4, 5.2),
    "selected_training": (11.5, 8.0),
    "selected_diagnostics": (12.0, 8.8),
    "depth_sensitivity": (14.5, 7.5),
    "worst_reconstructions": (14.5, 11.5),
    "latent_geometry": (11.5, 9.0),
    "latent_parameter_correlations": (12.5, 4.8),
}

METRIC_PRESENTATION = {
    "log10_rmse": {
        "scale": 1.0,
        "title": "Global log-space accuracy",
        "ylabel": "Validation log-RMSE [dex]",
    },
    "mean_relative_error": {
        "scale": 100.0,
        "title": "Typical physical-amplitude accuracy",
        "ylabel": "Mean physical relative error [%]",
    },
    "surface_relative_rmse_p95": {
        "scale": 100.0,
        "title": "Difficult-surface accuracy",
        "ylabel": "p95 surface relative RMS [%]",
    },
    "surface_relative_maximum_p99": {
        "scale": 100.0,
        "title": "Robust extreme local error",
        "ylabel": "p99 surface maximum error [%]",
    },
}

VALIDATION_RC_PARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "font.size": 10,
}

__all__ = [
    "DEPTH_LINESTYLES",
    "DEPTH_MARKERS",
    "FAMILY_COLORS",
    "FIGURE_SIZES",
    "METRIC_PRESENTATION",
    "SELECTION_MARKERS",
    "apply_validation_style",
    "empirical_survival",
    "plot_validation_metric_panel",
    "positive_log_limits",
    "safe_pearson_correlation",
    "save_validation_figure",
    "selection_legend_handles",
    "shared_positive_log_norm",
]


def apply_validation_style() -> None:
    """
    Apply the common Matplotlib style for compressor validation figures.
    """
    pyplot.rcParams.update(VALIDATION_RC_PARAMS)


def save_validation_figure(
    figure: matplotlib.figure.Figure,
    path: Path,
    *,
    creator: str,
) -> Path:
    """
    Atomically save one non-empty validation figure as PDF.
    
    Arguments:
        figure (matplotlib.figure.Figure):
            Completed Matplotlib figure.
        path (pathlib.Path):
            Destination PDF path.
        creator (str):
            Human-readable notebook identifier stored in PDF metadata.
    
    Returns:
        saved_path (pathlib.Path):
            Existing non-empty PDF path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    figure.savefig(
        temporary_path,
        format="pdf",
        metadata={"Creator": creator},
    )
    temporary_path.replace(path)
    
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not written: {path}")
    
    return path


def empirical_survival(
    values: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return the empirical survival curve of finite values.
    
    Arguments:
        values (numpy.ndarray):
            Values whose exceedance probabilities are required.
    
    Returns:
        sorted_values (numpy.ndarray):
            Finite values in ascending order.
        survival (numpy.ndarray):
            Fraction of samples greater than or equal to each sorted value.
    """
    finite_values = numpy.asarray(values, dtype=float)
    finite_values = finite_values[numpy.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("A survival curve requires at least one finite value.")
    
    sorted_values = numpy.sort(finite_values)
    survival = (
        sorted_values.size - numpy.arange(sorted_values.size)
    ) / sorted_values.size
    return sorted_values, survival


def positive_log_limits(
    values: Iterable[float],
    *,
    minimum_span_decades: float = 0.2,
    padding_fraction: float = 0.08,
) -> tuple[float, float]:
    """
    Calculate padded logarithmic limits from pooled positive values.
    
    Arguments:
        values (collections.abc.Iterable[float]):
            Pooled display-unit values from every figure being compared.
        minimum_span_decades (float):
            Minimum logarithmic span used for nearly identical values.
        padding_fraction (float):
            Fraction of the logarithmic span added at each boundary.
    
    Returns:
        limits (tuple[float, float]):
            Strictly positive lower and upper plotting limits.
    """
    array = numpy.asarray(list(values), dtype=float)
    positive = array[numpy.isfinite(array) & (array > 0.0)]
    if positive.size == 0:
        raise ValueError("Logarithmic limits require finite positive values.")
    if minimum_span_decades <= 0.0:
        raise ValueError("minimum_span_decades must be positive.")
    if padding_fraction < 0.0:
        raise ValueError("padding_fraction cannot be negative.")
    
    lower_log = float(numpy.min(numpy.log10(positive)))
    upper_log = float(numpy.max(numpy.log10(positive)))
    span = max(upper_log - lower_log, minimum_span_decades)
    padding = padding_fraction * span
    return 10.0 ** (lower_log - padding), 10.0 ** (upper_log + padding)


def shared_positive_log_norm(
    arrays: Iterable[numpy.ndarray],
    *,
    floor: float = 1.0e-3,
) -> LogNorm:
    """
    Build one logarithmic normalization from several positive arrays.
    
    Arguments:
        arrays (collections.abc.Iterable[numpy.ndarray]):
            Arrays expressed in the same display units.
        floor (float):
            Smallest display value admitted to the normalization.
    
    Returns:
        normalization (matplotlib.colors.LogNorm):
            Shared positive logarithmic colour normalization.
    """
    if floor <= 0.0:
        raise ValueError("floor must be positive.")
    
    finite_positive = []
    for values in arrays:
        array = numpy.asarray(values, dtype=float)
        selected = array[numpy.isfinite(array) & (array > 0.0)]
        if selected.size:
            finite_positive.append(selected)
    if not finite_positive:
        raise ValueError("A logarithmic normalization requires positive values.")
    
    pooled = numpy.concatenate(finite_positive)
    lower = max(floor, float(numpy.min(pooled)))
    upper = float(numpy.max(pooled))
    if upper <= lower:
        upper = 10.0 * lower
    return LogNorm(vmin=lower, vmax=upper)


def safe_pearson_correlation(
    first: numpy.ndarray,
    second: numpy.ndarray,
) -> float:
    """
    Calculate Pearson correlation while masking undefined coordinates.
    
    Arguments:
        first (numpy.ndarray):
            First one-dimensional sample vector.
        second (numpy.ndarray):
            Second one-dimensional sample vector of the same length.
    
    Returns:
        correlation (float):
            Pearson correlation, or NaN for insufficient or constant data.
    """
    first = numpy.asarray(first, dtype=float)
    second = numpy.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Pearson inputs must be same-length one-dimensional arrays.")
    
    finite = numpy.isfinite(first) & numpy.isfinite(second)
    first = first[finite]
    second = second[finite]
    if first.size < 2:
        return float("nan")
    if numpy.std(first) == 0.0 or numpy.std(second) == 0.0:
        return float("nan")
    return float(numpy.corrcoef(first, second)[0, 1])


def selection_legend_handles() -> list[Line2D]:
    """
    Return canonical legend handles for the two validation selections.
    
    Returns:
        handles (list[matplotlib.lines.Line2D]):
            Minimum-adequate and best-observed legend handles.
    """
    return [
        Line2D(
            [],
            [],
            color="black",
            marker=SELECTION_MARKERS["minimum_adequate"],
            linestyle="none",
            markersize=11,
            label="Family minimum adequate",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker=SELECTION_MARKERS["best_observed_accuracy"],
            linestyle="none",
            markersize=8,
            label="Family best observed accuracy",
        ),
    ]


def plot_validation_metric_panel(
    axis: matplotlib.axes.Axes,
    rows: Sequence[Mapping[str, object]],
    pca_rows: Sequence[Mapping[str, object]],
    latent_dimensions: Sequence[int],
    metric_name: str,
    *,
    minimum_adequate_cell_ids: set[str],
    best_accuracy_cell_ids: set[str],
    scale: float,
    ylabel: str,
    y_limits: tuple[float, float],
    include_pca: bool = True,
) -> None:
    """
    Plot neural configuration series and their matched PCA reference.
    
    Arguments:
        axis (matplotlib.axes.Axes):
            Target plotting axis.
        rows (collections.abc.Sequence[collections.abc.Mapping]):
            Authenticated aggregated neural result rows.
        pca_rows (collections.abc.Sequence[collections.abc.Mapping]):
            Matched-rank PCA result rows.
        latent_dimensions (collections.abc.Sequence[int]):
            Ordered latent dimensions shown on the horizontal axis.
        metric_name (str):
            Scalar validation metric field to plot.
        minimum_adequate_cell_ids (set[str]):
            Family minimum-adequate cell identifiers to mark with stars.
        best_accuracy_cell_ids (set[str]):
            Family best-observed cell identifiers to mark with X symbols.
        scale (float):
            Display-unit multiplier applied to every metric value.
        ylabel (str):
            Vertical-axis label including display units.
        y_limits (tuple[float, float]):
            Shared positive logarithmic limits.
        include_pca (bool):
            Whether to draw the matched-rank PCA reference.
    """
    series = {}
    for row in rows:
        key = (str(row["architecture"]), str(row["depth"]))
        series.setdefault(key, {})[int(row["latent_dim"])] = row
    
    for (architecture, depth), values in sorted(series.items()):
        if architecture not in FAMILY_COLORS:
            raise KeyError(f"No family colour is defined for {architecture}.")
        if depth not in DEPTH_LINESTYLES or depth not in DEPTH_MARKERS:
            raise KeyError(f"No depth style is defined for {depth}.")
        
        color = FAMILY_COLORS[architecture]
        y_values = numpy.asarray(
            [
                scale * float(values[latent_dim][metric_name])
                if latent_dim in values
                else numpy.nan
                for latent_dim in latent_dimensions
            ],
            dtype=float,
        )
        axis.plot(
            latent_dimensions,
            y_values,
            color=color,
            linestyle=DEPTH_LINESTYLES[depth],
            marker=DEPTH_MARKERS[depth],
            linewidth=1.8,
            markersize=5.5,
            label=(
                f"{architecture.replace('_', '-')} "
                f"{depth.replace('Depth', 'D')}"
            ),
        )
        
        for latent_dim in latent_dimensions:
            if latent_dim not in values:
                continue
            row = values[latent_dim]
            y_value = scale * float(row[metric_name])
            axis.scatter(
                [latent_dim],
                [y_value],
                marker=DEPTH_MARKERS[depth],
                s=36,
                facecolor=color if bool(row["target_met"]) else "white",
                edgecolor=color,
                linewidth=1.0,
                zorder=4,
            )
            if str(row["cell_id"]) in minimum_adequate_cell_ids:
                axis.scatter(
                    [latent_dim],
                    [y_value],
                    marker=SELECTION_MARKERS["minimum_adequate"],
                    s=150,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=7,
                )
            if str(row["cell_id"]) in best_accuracy_cell_ids:
                axis.scatter(
                    [latent_dim],
                    [y_value],
                    marker=SELECTION_MARKERS["best_observed_accuracy"],
                    s=90,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=7,
                )
    
    if include_pca:
        pca_by_latent = {
            int(row["latent_dim"]): row
            for row in pca_rows
        }
        axis.plot(
            latent_dimensions,
            [
                scale * float(pca_by_latent[latent_dim][metric_name])
                for latent_dim in latent_dimensions
            ],
            color=FAMILY_COLORS["PCA"],
            linestyle="-.",
            marker="D",
            linewidth=2.2,
            markersize=5,
            label="Matched rank-$L$ PCA",
        )
    
    axis.set_yscale("log")
    axis.set_ylim(*y_limits)
    axis.set_xticks(latent_dimensions)
    axis.set_xlabel("Latent dimension $L$")
    axis.set_ylabel(ylabel)
