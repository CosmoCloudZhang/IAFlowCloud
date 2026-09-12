#!/usr/bin/env python3
"""Render the Section 4 Conv1D architecture schematic."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPORT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPORT_DIR / "figures"
INK = "#19364D"
PALE = "#F3F7FA"
ORANGE_FILL = "#F7E6D4"
CONV_FILLS = ("#E4F0F7", "#D3E6F2", "#C2DCEC")


def box(axes, x, y, width, height, fill=PALE, lw=1.05):
    """Draw one rounded rectangle in figure coordinates."""

    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            linewidth=lw,
            edgecolor=INK,
            facecolor=fill,
            mutation_aspect=0.6,
            zorder=2,
        )
    )


def arrow(axes, x0, y0, x1, y1):
    """Draw a short connector between two boxes."""

    axes.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.95,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def label(axes, x, y, text, size=7.2, bold=False, color=INK):
    """Place a centered label."""

    axes.text(
        x,
        y,
        text,
        fontsize=size,
        ha="center",
        va="center",
        color=color,
        weight="bold" if bold else "normal",
        linespacing=1.25,
        zorder=4,
    )


def build_architecture_schematic() -> None:
    """Draw the three-layer Conv1D encoder ending in a square latent block."""

    fig_w, fig_h = 6.35, 2.20
    figure, axes = pyplot.subplots(figsize=(fig_w, fig_h))
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.04, top=0.96)
    axes.set(xlim=(0, 1), ylim=(0, 1))
    axes.axis("off")
    aspect = fig_w / fig_h
    midline = 0.50

    def centered(width, height, x_left):
        return x_left, midline - height / 2, width, height

    # Feature-map rectangles: wider when the k-axis is longer, taller when
    # the channel count is larger.  Printed aspect is approximate, not exact.
    stages = [
        (*centered(0.112, 0.28, 0.018), PALE, "Input", r"$31\times101$"),
        (*centered(0.092, 0.36, 0.168), CONV_FILLS[0], "Conv 1", r"$64\times51$"),
        (*centered(0.080, 0.44, 0.298), CONV_FILLS[1], "Conv 2", r"$128\times26$"),
        (*centered(0.072, 0.52, 0.416), CONV_FILLS[2], "Conv 3", r"$256\times13$"),
        (*centered(0.070, 0.30, 0.528), PALE, "Flatten", r"$3{,}328$"),
        (*centered(0.058, 0.22, 0.636), PALE, "", "256"),
        (*centered(0.050, 0.17, 0.730), PALE, "", "64"),
        (*centered(0.044, 0.13, 0.816), PALE, "", "16"),
    ]
    boxes = []
    for x, y, width, height, fill, top, inside in stages:
        box(axes, x, y, width, height, fill=fill)
        label(axes, x + width / 2, y + height / 2, inside, size=7.0)
        if top:
            label(axes, x + width / 2, y + height + 0.055, top, size=6.8, bold=True)
        boxes.append((x, y, width, height))

    latent_w = 0.056
    latent_h = latent_w * aspect
    latent_x = 0.918
    latent_y = midline - latent_h / 2
    box(axes, latent_x, latent_y, latent_w, latent_h, fill=ORANGE_FILL, lw=1.25)
    label(axes, latent_x + latent_w / 2, midline + 0.012, r"$L$", size=10.0, bold=True)
    label(axes, latent_x + latent_w / 2, latent_y + latent_h + 0.055, "Latent", size=6.8, bold=True)
    boxes.append((latent_x, latent_y, latent_w, latent_h))

    for (x0, y0, w0, h0), (x1, y1, _w1, h1) in zip(boxes[:-1], boxes[1:]):
        arrow(
            axes,
            x0 + w0 + 0.004,
            y0 + h0 / 2,
            x1 - 0.004,
            y1 + h1 / 2,
        )

    # Operation labels sit under the three convolutions only.
    label(axes, 0.214, 0.14, r"$k{=}5$, stride 2", size=6.1)
    label(axes, 0.338, 0.14, r"$k{=}5$, stride 2", size=6.1)
    label(axes, 0.452, 0.14, r"$k{=}3$, stride 2", size=6.1)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / "ae_conv1d_architecture.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "ae_conv1d_architecture.preview.png", dpi=160, bbox_inches="tight")
    pyplot.close(figure)


if __name__ == "__main__":
    build_architecture_schematic()
