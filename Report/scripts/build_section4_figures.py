#!/usr/bin/env python3
"""
Render coordinated Conv1D and PCA-AE architecture figures as vector artwork.

Projected sheets represent matrices; strips represent vectors. Their visual
extents are schematic, while the printed dimensions follow the report models.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.path import Path as DrawingPath


REPORT_DIRECTORY = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = REPORT_DIRECTORY / "figures"
SOURCE_DIRECTORY = FIGURE_DIRECTORY / "sources" / "schematics"
INK = "#20354A"
BLUE = "#2C6C9B"
FILLS = ("#E1EEF6", "#C9E0EE", "#ACCDDF")
GREY = "#66717C"
TOP = 78.0
BOTTOM = 26.0


def label(
    axes,
    x,
    y,
    value,
    *,
    size=8.8,
    color=INK,
    bold=False,
):
    """
    Place one centered, print-sized diagram label.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Diagram axes in millimetres.
        x (float):
            Horizontal text anchor.
        y (float):
            Vertical text anchor.
        value (str):
            Plain text or mathematical notation.
        size (float):
            Native font size in points.
        color (str):
            Text colour.
        bold (bool):
            Whether to emphasize a group heading.
    """
    axes.text(
        x, y, value, fontsize=size, color=color, ha="center", va="center",
        weight="bold" if bold else "normal", linespacing=1.25, zorder=6,
    )


def sheet(
    axes,
    x,
    y,
    width,
    height,
    *,
    fill="#F1F5F7",
    fixed=False,
):
    """
    Draw a projected matrix with a schematic row and column grid.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Diagram axes in millimetres.
        x (float):
            Horizontal matrix centre.
        y (float):
            Vertical matrix centre.
        width (float):
            Display width representing matrix columns.
        height (float):
            Display height representing matrix rows.
        fill (str):
            Matrix face colour.
        fixed (bool):
            Whether dashed grey edges identify a fixed transform.
    
    Returns:
        anchors (tuple[float, float]):
            Left and right flow anchors at the vertical centre.
    """
    edge = GREY if fixed else BLUE
    shear = 2.2
    corners = [
        (x - width / 2 - shear / 2, y - height / 2),
        (x + width / 2 - shear / 2, y - height / 2),
        (x + width / 2 + shear / 2, y + height / 2),
        (x - width / 2 + shear / 2, y + height / 2),
    ]
    shadow = [(px + 0.7, py - 0.7) for px, py in corners]
    axes.add_patch(Polygon(shadow, facecolor=INK, edgecolor="none", alpha=0.08))
    axes.add_patch(
        Polygon(
            corners, facecolor=fill, edgecolor=edge, linewidth=0.9,
            linestyle=(0, (3.0, 1.8)) if fixed else "solid", zorder=2,
        )
    )
    for index in range(1, 6):
        fraction = index / 6
        grid_y = y + (fraction - 0.5) * height
        left = x - width / 2 + (fraction - 0.5) * shear
        axes.plot(
            [left, left + width], [grid_y, grid_y], color=edge,
            alpha=0.25, linewidth=0.45, zorder=3,
        )
    
    for index in range(1, 8):
        fraction = index / 8
        lower = x + (fraction - 0.5) * width - shear / 2
        axes.plot(
            [lower, lower + shear], [y - height / 2, y + height / 2],
            color=edge, alpha=0.25, linewidth=0.45, zorder=3,
        )
    
    return x - width / 2, x + width / 2


def vector(
    axes,
    x,
    y,
    height,
    *,
    latent=False,
):
    """
    Draw a segmented vector strip, using an ellipsis for variable latent width.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Diagram axes in millimetres.
        x (float):
            Horizontal vector centre.
        y (float):
            Vertical vector centre.
        height (float):
            Display height indicating relative vector dimension.
        latent (bool):
            Whether to highlight the latent representation.
    
    Returns:
        anchors (tuple[float, float]):
            Left and right flow anchors.
    """
    width = 3.0 if latent else 2.6
    edge = "#A76725" if latent else BLUE
    fill = "#F5DAB3" if latent else FILLS[1]
    axes.add_patch(
        Rectangle(
            (x - width / 2 + 0.5, y - height / 2 - 0.5), width, height,
            facecolor=INK, edgecolor="none", alpha=0.08, zorder=1,
        )
    )
    axes.add_patch(
        Rectangle(
            (x - width / 2, y - height / 2), width, height,
            facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=2,
        )
    )
    if latent:
        divisions = (-height / 2 + 2.3, height / 2 - 2.3)
        label(axes, x, y, r"$\vdots$", size=8.5, color=edge)
    else:
        segments = max(3, min(9, round(height / 2.8)))
        divisions = [height * (index / segments - 0.5) for index in range(1, segments)]
    
    for division in divisions:
        axes.plot(
            [x - width / 2, x + width / 2], [y + division, y + division],
            color=edge, alpha=0.4, linewidth=0.5, zorder=3,
        )
    
    return x - width / 2, x + width / 2


def arrow(
    axes,
    start,
    end,
    *,
    text="",
    offset=6.0,
    fixed=False,
):
    """
    Connect representations with a directed operation.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Diagram axes in millimetres.
        start (tuple[float, float]):
            Arrow start.
        end (tuple[float, float]):
            Arrow end.
        text (str):
            Optional operation label.
        offset (float):
            Vertical label offset in millimetres.
        fixed (bool):
            Whether the operation is fixed.
    """
    color = GREY if fixed else BLUE
    axes.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=7.5,
            linewidth=0.85, color=color, shrinkA=2.0, shrinkB=2.0, zorder=4,
        )
    )
    if text:
        label(
            axes, (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + offset, text, size=8.5, color=color,
        )


def bottleneck(
    axes,
    x,
    dense_edge,
):
    """
    Route the encoder into one bottleneck and then into the decoder below.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Diagram axes in millimetres.
        x (float):
            Horizontal bottleneck centre.
        dense_edge (float):
            Right edge of the nearest dense vectors.
    """
    vector(axes, x, 52.0, 13.0, latent=True)
    paths = (
        [(dense_edge + 1.0, TOP), (x, TOP), (x, 59.3)],
        [(x, 44.7), (x, BOTTOM), (dense_edge + 1.0, BOTTOM)],
    )
    for points in paths:
        path = DrawingPath(
            points, [DrawingPath.MOVETO, DrawingPath.LINETO, DrawingPath.LINETO],
        )
        axes.add_patch(
            FancyArrowPatch(
                path=path, arrowstyle="-|>", mutation_scale=8.0,
                linewidth=1.0, color=BLUE, zorder=4,
            )
        )
    
    label(axes, x - 8.0, 54.0, r"$\mathbf{h}_i$", size=10.0)
    label(axes, x - 8.0, 47.0, r"$L$", size=9.2, color="#A76725")


def canvas():
    """
    Create a common native 160 mm canvas.
    
    Returns:
        result (tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]):
            Figure and axes with common row headings.
    """
    figure = pyplot.figure(figsize=(160.0 / 25.4, 110.0 / 25.4))
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set(xlim=(0.0, 160.0), ylim=(0.0, 110.0), aspect="equal")
    axes.axis("off")
    label(axes, 13.0, 104.0, "ENCODER", size=9.0, bold=True)
    label(axes, 13.0, 50.0, "DECODER", size=9.0, bold=True)
    label(
        axes, 80.0, 1.9,
        "Matrix extents and vector entries are schematic; labels give the dimensions.",
        size=8.5, color=GREY,
    )
    return figure, axes


def build_conv1d_diagram():
    """
    Draw the three-layer Conv1D example with exact intermediate dimensions.
    
    Returns:
        figure (matplotlib.figure.Figure):
            Completed Conv1D architecture artwork.
    """
    figure, axes = canvas()
    centers = (15.0, 43.0, 64.0, 83.0)
    sizes = ((20.0, 12.0), (13.0, 17.0), (8.0, 22.0), (5.6, 28.0))
    shapes = (r"$N_z\!\times\!N_k$", r"$64\!\times\!51$", r"$128\!\times\!26$", r"$256\!\times\!13$")
    dense_centers = (100.0, 115.0, 130.0, 141.0)
    dense_heights = (27.0, 22.0, 15.0, 10.0)
    dense_shapes = (r"$256\!\cdot\!13$", "256", "64", "16")
    label(axes, 57.0, 103.0, "Conv1D (kernel, stride)", size=8.5, color=BLUE)
    label(axes, 124.0, 103.0, "Fully connected (FC)", size=8.5, color=BLUE)
    label(axes, 58.0, 49.0, "Transposed Conv1D", size=8.5, color=BLUE)
    for y, label_y, encoder in ((TOP, 59.0, True), (BOTTOM, 7.8, False)):
        sheets = []
        for index, (x, size, text) in enumerate(zip(centers, sizes, shapes)):
            fill = "#F1F5F7" if index == 0 else FILLS[index - 1]
            sheets.append(sheet(axes, x, y, *size, fill=fill))
            label(axes, x, label_y, text)
        
        vectors = []
        for x, height, text in zip(dense_centers, dense_heights, dense_shapes):
            vectors.append(vector(axes, x, y, height))
            label(axes, x, label_y, text)
        
        for index, kernel in enumerate((5, 5, 3)):
            start, end = (sheets[index][1], y), (sheets[index + 1][0], y)
            arrow(axes, start if encoder else end, end if encoder else start, text=f"{kernel}, 2")
        
        start, end = (sheets[-1][1], y), (vectors[0][0], y)
        arrow(
            axes, start if encoder else end, end if encoder else start,
            text="flatten" if encoder else "reshape", offset=17.0,
        )
        for left, right in zip(vectors[:-1], vectors[1:]):
            start, end = (left[1], y), (right[0], y)
            arrow(axes, start if encoder else end, end if encoder else start, text="FC")
        
        symbol = r"$\mathbf{X}_i$" if encoder else r"$\widehat{\mathbf{X}}_i$"
        label(axes, centers[0], y + 13.0, symbol, size=10.0)
    
    label(axes, 57.0, 55.0, "learned channels × sequence length", size=8.5, color=GREY)
    bottleneck(axes, 155.0, dense_centers[-1] + 1.3)
    return figure


def build_pca_ae_diagram():
    """
    Draw fixed PCA transforms surrounding the learned coefficient autoencoder.
    
    Returns:
        figure (matplotlib.figure.Figure):
            Completed PCA-AE architecture artwork.
    """
    figure, axes = canvas()
    dense_centers = (68.0, 91.0, 115.0, 136.0)
    dense_heights = (16.0, 24.0, 18.0, 11.0)
    dense_shapes = (r"$R=30$", "256", "64", "16")
    label(axes, 45.0, 103.0, "Fixed PCA transform", size=8.5, color=GREY)
    label(axes, 114.0, 103.0, "Fully connected (FC)", size=8.5, color=BLUE)
    label(axes, 69.0, 95.5, "Coefficients", size=8.5, color=GREY)
    for y, label_y, encoder in ((TOP, 59.0, True), (BOTTOM, 7.8, False)):
        surface = sheet(axes, 15.0, y, 20.0, 12.0)
        size = (17.0, 12.0) if encoder else (12.0, 17.0)
        transform = sheet(axes, 44.0, y, *size, fill="#E6E9ED", fixed=True)
        vectors = []
        for x, height, text in zip(dense_centers, dense_heights, dense_shapes):
            vectors.append(vector(axes, x, y, height))
            label(axes, x, label_y, text)
        
        symbol = r"$\mathbf{X}_i$" if encoder else r"$\widehat{\mathbf{X}}_i$"
        basis = r"$\mathbf{V}_R^{\mathsf{T}}$" if encoder else r"$\mathbf{V}_R$"
        shape = r"$R\times(N_zN_k)$" if encoder else r"$(N_zN_k)\times R$"
        label(axes, 15.0, y + 13.0, symbol, size=10.0)
        label(axes, 15.0, label_y, r"$N_z\!\times\!N_k$")
        label(axes, 44.0, y + 13.5, basis, size=9.0, color=GREY)
        label(axes, 44.0, label_y, shape, size=8.5, color=GREY)
        start, end = (surface[1], y), (transform[0], y)
        arrow(
            axes, start if encoder else end, end if encoder else start,
            text="project" if encoder else "restore", offset=-12.5, fixed=True,
        )
        start, end = (transform[1], y), (vectors[0][0], y)
        arrow(
            axes, start if encoder else end, end if encoder else start,
            text="scale" if encoder else "unscale", offset=11.5, fixed=True,
        )
        for left, right in zip(vectors[:-1], vectors[1:]):
            start, end = (left[1], y), (right[0], y)
            arrow(axes, start if encoder else end, end if encoder else start, text="FC")
    
    label(axes, 73.0, 49.5, "Grey: fixed transforms", size=8.5, color=GREY)
    label(axes, 114.0, 49.5, "Blue: learned maps", size=8.5, color=BLUE)
    bottleneck(axes, 153.0, dense_centers[-1] + 1.3)
    return figure


def save_diagram(
    figure,
    stem,
    preview_directory,
):
    """
    Save a vector PDF, editable SVG, and optional external inspection PNG.
    
    Arguments:
        figure (matplotlib.figure.Figure):
            Completed diagram.
        stem (str):
            Common output basename.
        preview_directory (pathlib.Path or None):
            External PNG review directory, if requested.
    """
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIRECTORY / f"{stem}.pdf")
    figure.savefig(SOURCE_DIRECTORY / f"{stem}.svg")
    if preview_directory is not None:
        preview_directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_directory / f"{stem}.png", dpi=200)
    
    pyplot.close(figure)


def main():
    """
    Build both architecture diagrams using consistent native typography.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-directory", type=Path)
    arguments = parser.parse_args()
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "iaflow-section4-architectures",
    }
    with pyplot.rc_context(style):
        save_diagram(build_conv1d_diagram(), "ae_conv1d_architecture", arguments.preview_directory)
        save_diagram(build_pca_ae_diagram(), "pca_ae_architecture", arguments.preview_directory)


if __name__ == "__main__":
    main()
