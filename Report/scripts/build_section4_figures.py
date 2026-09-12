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
            mutation_scale=8,
            linewidth=0.9,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def label(axes, x, y, text, size=6.8, bold=False):
    """Place a centered label."""

    axes.text(
        x,
        y,
        text,
        fontsize=size,
        ha="center",
        va="center",
        color=INK,
        weight="bold" if bold else "normal",
        linespacing=1.2,
        zorder=4,
    )


def build_architecture_schematic() -> None:
    """Draw the three-layer Conv1D encoder and its mirrored decoder."""

    fig_w, fig_h = 6.35, 3.55
    figure, axes = pyplot.subplots(figsize=(fig_w, fig_h))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95)
    axes.set(xlim=(0, 1), ylim=(0, 1))
    axes.axis("off")
    aspect = fig_w / fig_h

    columns = [
        (0.018, 0.100, 0.22, PALE, "Input", "Output", r"$31\times101$"),
        (0.140, 0.082, 0.28, CONV_FILLS[0], "Conv 1", "ConvT 1", r"$64\times51$"),
        (0.244, 0.072, 0.34, CONV_FILLS[1], "Conv 2", "ConvT 2", r"$128\times26$"),
        (0.338, 0.064, 0.40, CONV_FILLS[2], "Conv 3", "ConvT 3", r"$256\times13$"),
        (0.424, 0.062, 0.24, PALE, "Flatten", "Unflatten", r"$3{,}328$"),
        (0.508, 0.052, 0.18, PALE, "", "", "256"),
        (0.582, 0.046, 0.14, PALE, "", "", "64"),
        (0.650, 0.042, 0.11, PALE, "", "", "16"),
    ]

    def draw_row(midline, name_index):
        drawn = []
        for x, width, height, fill, enc_name, dec_name, inside in columns:
            y = midline - height / 2
            top = enc_name if name_index == 0 else dec_name
            box(axes, x, y, width, height, fill=fill)
            label(axes, x + width / 2, y + height / 2, inside, size=6.5)
            if top:
                offset = height / 2 + 0.045 if name_index == 0 else -(height / 2 + 0.045)
                label(axes, x + width / 2, midline + offset, top, size=6.2, bold=True)
            drawn.append((x, y, width, height))
        return drawn

    encoder = draw_row(0.73, 0)
    decoder = draw_row(0.27, 1)

    for left, right in zip(encoder[:-1], encoder[1:]):
        arrow(
            axes,
            left[0] + left[2] + 0.003,
            left[1] + left[3] / 2,
            right[0] - 0.003,
            right[1] + right[3] / 2,
        )
    for left, right in zip(decoder[:-1], decoder[1:]):
        arrow(
            axes,
            right[0] - 0.003,
            right[1] + right[3] / 2,
            left[0] + left[2] + 0.003,
            left[1] + left[3] / 2,
        )

    latent_w = 0.070
    latent_h = latent_w * aspect
    latent_x = 0.718
    latent_y = 0.50 - latent_h / 2
    box(axes, latent_x, latent_y, latent_w, latent_h, fill=ORANGE_FILL, lw=1.25)
    label(axes, latent_x + latent_w / 2, 0.508, r"$L$", size=10.0, bold=True)
    label(axes, latent_x + latent_w / 2, latent_y + latent_h + 0.042, "Latent", size=6.2, bold=True)

    enc16 = encoder[-1]
    dec16 = decoder[-1]
    arrow(
        axes,
        enc16[0] + enc16[2] + 0.004,
        enc16[1] + enc16[3] / 2,
        latent_x - 0.004,
        latent_y + latent_h - 0.012,
    )
    arrow(
        axes,
        latent_x - 0.004,
        latent_y + 0.012,
        dec16[0] + dec16[2] + 0.004,
        dec16[1] + dec16[3] / 2,
    )

    label(axes, 0.068, 0.935, "Encoder", size=7.4, bold=True)
    label(axes, 0.068, 0.065, "Decoder", size=7.4, bold=True)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / "ae_conv1d_architecture.pdf", bbox_inches="tight")
    figure.savefig(
        FIGURE_DIR / "ae_conv1d_architecture.preview.png",
        dpi=160,
        bbox_inches="tight",
    )
    pyplot.close(figure)


if __name__ == "__main__":
    build_architecture_schematic()
