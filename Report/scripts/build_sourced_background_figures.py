#!/usr/bin/env python3
"""
Render explanatory background diagrams and mathematical label overlays.

Published plot panels are preserved in figures/sources/. The companion
finalise_report_figures.py assembles them without added credit footers.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

REPORT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPORT_DIR / "figures"
INK = "#19364D"
BLUE = "#347EAE"
TEAL = "#358C82"
ORANGE = "#D38745"
PALE = "#F3F7FA"


def canvas(
    height,
):
    """
    Create a canvas at approximately the report's printed figure width.
    
    Arguments:
        height (float):
            Figure height in inches; the width is fixed at 6.05 inches.
    
    Returns:
        figure, axes (matplotlib.figure.Figure, matplotlib.axes.Axes):
            Figure and unit-square drawing coordinates.
    """
    figure, axes = pyplot.subplots(figsize=(6.05, height))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes.set(xlim=(0, 1), ylim=(0, 1))
    axes.axis("off")
    return figure, axes


def text(
    axes,
    x,
    y,
    value,
    size=8.5,
    bold=False,
    align="center",
):
    """
    Place a label with a controlled printed font size.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Drawing canvas.
        x (float):
            Horizontal label position.
        y (float):
            Vertical label position.
        value (str):
            Text, optionally including Matplotlib math notation.
        size (float):
            Font size in points at the native printed width.
        bold (bool):
            Whether to use a bold font.
        align (str):
            Horizontal alignment relative to x.
    """
    axes.text(
        x, y, value, fontsize=size, ha=align, va="center",
        color=INK, weight="bold" if bold else "normal", linespacing=1.4,
    )


def box(
    axes,
    x,
    y,
    width,
    height,
    fill=PALE,
    dashed=False,
):
    """
    Draw a panel or workflow box.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Drawing canvas.
        x (float):
            Left edge in drawing coordinates.
        y (float):
            Bottom edge in drawing coordinates.
        width (float):
            Horizontal extent.
        height (float):
            Vertical extent.
        fill (str):
            Background colour.
        dashed (bool):
            Whether to mark a future-work boundary with a dashed outline.
    """
    axes.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=0.9, edgecolor="#91A8B8", facecolor=fill,
        linestyle="--" if dashed else "-", zorder=0,
    ))


def arrow(
    axes,
    start,
    end,
    colour=INK,
    both=False,
):
    """
    Draw a directional connection or a wavelength interval.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Drawing canvas.
        start (tuple):
            Starting endpoint in drawing coordinates.
        end (tuple):
            Ending endpoint in drawing coordinates.
        colour (str):
            Stroke colour.
        both (bool):
            Whether to draw arrowheads at both ends.
    """
    axes.add_patch(FancyArrowPatch(
        start, end, arrowstyle="<->" if both else "-|>",
        mutation_scale=10, linewidth=1.2, color=colour,
        shrinkA=0, shrinkB=0, zorder=3,
    ))


def galaxy(
    axes,
    x,
    y,
    width=0.10,
    ratio=0.45,
    colour=BLUE,
    outline=False,
):
    """
    Draw a schematic galaxy or circular reference image.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Drawing canvas.
        x (float):
            Horizontal centre in drawing coordinates.
        y (float):
            Vertical centre in drawing coordinates.
        width (float):
            Horizontal diameter in drawing coordinates.
        ratio (float):
            Minor-to-major axis ratio on the printed page.
        colour (str):
            Fill or outline colour.
        outline (bool):
            Whether to draw only a dashed reference outline.
    """
    scale = axes.figure.get_figwidth() / axes.figure.get_figheight()
    scale *= numpy.diff(axes.get_ylim())[0] / numpy.diff(axes.get_xlim())[0]
    axes.add_patch(Ellipse(
        (x, y), width, width * ratio * scale,
        edgecolor=colour, facecolor="none" if outline else colour,
        alpha=0.9, linewidth=1.2, linestyle="--" if outline else "-",
        zorder=2,
    ))


def observer(
    axes,
    x,
    y,
):
    """
    Mark the observer without assigning a physical size.
    
    Arguments:
        axes (matplotlib.axes.Axes):
            Drawing canvas.
        x (float):
            Horizontal observer position.
        y (float):
            Vertical observer position.
    """
    axes.plot(x, y, "o", ms=9, mfc="white", mec=INK, mew=1.2, zorder=4)
    axes.plot(x, y, ".", ms=3, color=INK, zorder=5)


def save(
    figure,
    stem,
):
    """
    Save a PDF and an editable vector SVG without cropping the canvas.
    
    Arguments:
        figure (matplotlib.figure.Figure):
            Completed drawing.
        stem (str):
            Filename stem under Report/figures/.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = FIGURE_DIR / "sources" / "schematics"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf")
    figure.savefig(source_dir / f"{stem}.svg")
    pyplot.close(figure)


def build_lensing_diagram():
    """
    Illustrate deflection, image distortion, and the spatial meaning of k.
    """
    figure, axes = canvas(4.6)
    text(axes, 0.02, 0.977, "(a) Light deflection", size=10, bold=True, align="left")
    box(axes, 0.02, 0.74, 0.96, 0.21)
    galaxy(axes, 0.10, 0.906, width=0.065)
    galaxy(axes, 0.50, 0.831, width=0.11, ratio=0.60, colour=ORANGE)
    axes.plot([0.14, 0.50, 0.915], [0.922, 0.933, 0.865], color=BLUE, lw=1.7)
    axes.plot([0.14, 0.50, 0.915], [0.887, 0.902, 0.865], color=BLUE, lw=1.7)
    arrow(axes, (0.84, 0.877), (0.905, 0.867), colour=BLUE)
    observer(axes, 0.925, 0.865)
    text(axes, 0.10, 0.773, "Source")
    text(axes, 0.50, 0.763, "Intervening matter")
    text(axes, 0.90, 0.773, "Observer")
    text(axes, 0.76, 0.918, "Deflection exaggerated", size=8)
    
    text(axes, 0.02, 0.707, "(b) Convergence and shear", size=10, bold=True, align="left")
    box(axes, 0.02, 0.48, 0.96, 0.205)
    galaxy(axes, 0.17, 0.594, width=0.075, ratio=1)
    galaxy(axes, 0.50, 0.594, width=0.105, ratio=1)
    galaxy(axes, 0.50, 0.594, width=0.075, ratio=1, colour="white", outline=True)
    galaxy(axes, 0.83, 0.594, width=0.125, ratio=0.40)
    galaxy(axes, 0.83, 0.594, width=0.075, ratio=1, colour="#6B7886", outline=True)
    text(axes, 0.17, 0.505, "Unlensed circle")
    text(axes, 0.50, 0.505, r"Convergence: $\kappa>0$")
    text(axes, 0.83, 0.505, r"Shear: $\gamma\ne 0$")
    
    text(axes, 0.02, 0.445, "(c) Spatial scale and wavenumber", size=10, bold=True, align="left")
    box(axes, 0.02, 0.025, 0.96, 0.39)
    text(axes, 0.25, 0.36, r"Large spatial scale: small $k$")
    text(axes, 0.75, 0.36, r"Small spatial scale: large $k$")
    phase = numpy.linspace(0, 1, 400)
    axes.plot(0.08 + 0.33 * phase, 0.26 + 0.036 * numpy.sin(2 * numpy.pi * phase), color=TEAL, lw=1.8)
    axes.plot(0.59 + 0.33 * phase, 0.26 + 0.036 * numpy.sin(10 * numpy.pi * phase), color=ORANGE, lw=1.8)
    arrow(axes, (0.08, 0.19), (0.41, 0.19), both=True)
    arrow(axes, (0.59, 0.19), (0.656, 0.19), both=True)
    text(axes, 0.25, 0.148, r"One wavelength $\lambda_{\mathrm{spatial}}$")
    text(axes, 0.75, 0.148, r"One wavelength $\lambda_{\mathrm{spatial}}$")
    text(axes, 0.50, 0.091, r"$k=2\pi/\lambda_{\mathrm{spatial}}$  (equal distance intervals above)")
    text(axes, 0.50, 0.047, r"$P_{\delta\delta}(k,z)$ describes the strength of fluctuations at each scale.", size=8)
    save(figure, "background_lensing_density")


def build_intrinsic_alignment_diagram():
    """
    Show the ellipticity convention and the three correlation mechanisms.
    """
    figure, axes = canvas(3.65)
    axes.set_ylim(0.185, 0.98)
    box(axes, 0.02, 0.815, 0.96, 0.145, fill="#EDF4F8")
    text(axes, 0.5, 0.895, r"$e^{\mathrm{O}}\simeq e^{\mathrm{I}}+\gamma+n_e$", size=14)
    
    for x in (0.02, 0.35, 0.68):
        box(axes, x, 0.20, 0.30, 0.57)
    
    text(axes, 0.17, 0.737, r"(a) $\mathrm{GG}$", size=13, bold=True)
    text(axes, 0.17, 0.688, "Gravitational shears", size=8.4)
    galaxy(axes, 0.09, 0.616, width=0.075)
    galaxy(axes, 0.25, 0.616, width=0.075)
    galaxy(axes, 0.17, 0.465, width=0.13, colour=ORANGE)
    axes.plot([0.09, 0.12, 0.17], [0.59, 0.50, 0.28], color=BLUE, lw=1.4)
    axes.plot([0.25, 0.22, 0.17], [0.59, 0.50, 0.28], color=BLUE, lw=1.4)
    axes.text(
        0.17, 0.408, "Foreground\nmatter", fontsize=8, ha="center",
        va="center", color=INK, bbox={"facecolor": PALE, "edgecolor": "none", "pad": 1.5},
    )
    observer(axes, 0.17, 0.27)
    text(axes, 0.17, 0.227, "Observer", size=8)
    
    text(axes, 0.50, 0.737, r"(b) $\mathrm{II}$", size=13, bold=True)
    text(axes, 0.50, 0.688, "Intrinsic orientations", size=8.4)
    galaxy(axes, 0.445, 0.535, width=0.08)
    galaxy(axes, 0.555, 0.535, width=0.08)
    arrow(axes, (0.41, 0.535), (0.37, 0.535), colour=ORANGE)
    arrow(axes, (0.59, 0.535), (0.63, 0.535), colour=ORANGE)
    arrow(axes, (0.50, 0.63), (0.50, 0.58), colour=ORANGE)
    arrow(axes, (0.50, 0.435), (0.50, 0.485), colour=ORANGE)
    text(axes, 0.50, 0.379, "Anisotropic tidal field", size=8)
    text(axes, 0.50, 0.285, "Related orientations\nbefore lensing", size=8.2)
    
    text(axes, 0.83, 0.737, r"(c) $\mathrm{GI}$", size=13, bold=True)
    text(axes, 0.83, 0.688, "Intrinsic shape and shear", size=8.1)
    galaxy(axes, 0.865, 0.596, width=0.085)
    text(axes, 0.83, 0.644, r"Background $\mathrm{G}$", size=8)
    galaxy(axes, 0.84, 0.445, width=0.12, colour=ORANGE)
    galaxy(axes, 0.755, 0.43, width=0.075)
    axes.plot([0.865, 0.905, 0.845], [0.572, 0.47, 0.28], color=BLUE, lw=1.4)
    text(axes, 0.785, 0.355, r"Foreground $\mathrm{I}$", size=8)
    observer(axes, 0.845, 0.27)
    text(axes, 0.845, 0.227, "Observer", size=8)
    
    save(figure, "background_intrinsic_alignment")


def build_model_tradeoff_diagram():
    """
    Connect IA model flexibility to the bounded reconstruction experiment.
    """
    figure, axes = canvas(3.9)
    axes.set_ylim(0.235, 1.015)
    text(axes, 0.50, 0.967, "IA model flexibility and parameter constraints", size=10, bold=True)
    arrow(axes, (0.45, 0.933), (0.25, 0.90))
    arrow(axes, (0.55, 0.933), (0.75, 0.90))
    box(axes, 0.02, 0.66, 0.46, 0.235, fill="#ECF5F0")
    box(axes, 0.52, 0.66, 0.46, 0.235, fill="#FCF2E9")
    text(axes, 0.25, 0.856, "NLA: restricted response", size=9.2, bold=True)
    text(axes, 0.75, 0.856, "TATT: more response terms", size=9.2, bold=True)
    text(axes, 0.25, 0.753, "Few nuisance parameters\nRestricted response flexibility\nBias if the model is inadequate")
    text(axes, 0.75, 0.753, "Broader tidal responses\nMore adjustable parameters\nCan be harder to constrain")
    arrow(axes, (0.25, 0.655), (0.35, 0.619))
    arrow(axes, (0.75, 0.655), (0.65, 0.619))
    box(axes, 0.04, 0.55, 0.92, 0.065, fill="#EDF0FA")
    text(axes, 0.50, 0.583, "Compact reconstruction of an NLA-inspired response", size=9, bold=True)
    text(axes, 0.02, 0.516, "Experiment in this report", size=9.5, bold=True, align="left")
    
    for x in (0.02, 0.35, 0.68):
        box(axes, x, 0.395, 0.30, 0.086)
        box(axes, x, 0.255, 0.30, 0.086)
    
    text(axes, 0.17, 0.438, "Sampled\nshape parameters")
    text(axes, 0.50, 0.438, "$\\mathcal{A}_{\\Theta}(k,z)$\n$31\\times101$ grid")
    text(axes, 0.83, 0.438, "PCA / autoencoder\nvariants", size=8)
    text(axes, 0.83, 0.298, "$L$ latent\ncoordinates")
    text(axes, 0.50, 0.298, "Reconstructed\nresponse")
    text(axes, 0.17, 0.298, "Matched validation\nerrors")
    arrow(axes, (0.325, 0.438), (0.345, 0.438))
    arrow(axes, (0.655, 0.438), (0.675, 0.438))
    arrow(axes, (0.83, 0.390), (0.83, 0.347))
    arrow(axes, (0.675, 0.298), (0.655, 0.298))
    arrow(axes, (0.345, 0.298), (0.325, 0.298))
    
    save(figure, "background_model_tradeoff")


def build_power_labels():
    """
    Render replacement axis labels and the nonlinear matter-power title.
    """
    width, height = 820.536, 251.444
    figure = pyplot.figure(figsize=(width / 72, height / 72))
    replacements = (
        (17.2, r"$P_{\delta\delta}^{\mathrm{NL}}$"),
        (561.136, r"$P_{\mathrm{II}}$"),
    )
    
    for x, symbol in replacements:
        figure.text(
            x / width, 147.514 / height, symbol + r"$\ [{\rm Mpc}^{3}]$",
            fontsize=10, ha="center", va="baseline", rotation=90,
            rotation_mode="anchor", color="black",
        )
    
    figure.text(
        151.8 / width, 234.244 / height,
        r"(a) $P_{\delta\delta}^{\mathrm{NL}}$",
        fontsize=10, ha="center", va="baseline", color="black",
    )
    
    source_dir = FIGURE_DIR / "sources" / "schematics"
    source_dir.mkdir(parents=True, exist_ok=True)
    with pyplot.rc_context({"pdf.fonttype": 3}):
        figure.savefig(source_dir / "power_spectrum_labels.pdf", transparent=True, facecolor="none")
    pyplot.close(figure)


def main():
    """
    Rebuild selected diagrams and overlays without recomputing result curves.
    """
    builders = {
        "lensing": build_lensing_diagram,
        "ia": build_intrinsic_alignment_diagram,
        "models": build_model_tradeoff_diagram,
        "power-labels": build_power_labels,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=tuple(builders), default=list(builders))
    args = parser.parse_args()
    pyplot.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    
    for name in dict.fromkeys(args.figures):
        builders[name]()


if __name__ == "__main__":
    main()
