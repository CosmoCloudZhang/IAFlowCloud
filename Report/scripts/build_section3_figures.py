"""Build the computational figures used in Section 3 of the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyccl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_PATH = PROJECT_ROOT / "Code"
FIGURE_PATH = PROJECT_ROOT / "Report" / "figures"
if str(CODE_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_PATH))

from ia_models.nla.model import (  # noqa: E402
    C0,
    Z_STAR,
    NLAModel,
    luminosity_factor,
    redshift_factor,
)


COLOURS = ["#b2182b", "#ef8a62", "#333333", "#67a9cf", "#2166ac"]


def load_cosmology() -> pyccl.Cosmology:
    """Create the Planck-like cosmology used for the illustrative panels.

    The notebook's CAMB setup is currently incompatible with the installed
    CAMB 2.0 interface.  Using the analytic Eisenstein--Hu transfer function
    keeps the same background parameters and normalization without changing
    the project's IA response functions.
    """

    parameter_path = PROJECT_ROOT / "Data" / "Cosmology" / "Planck.json"
    with parameter_path.open("r", encoding="utf-8") as file:
        parameter = json.load(file)

    return pyccl.Cosmology(
        h=parameter["H"],
        w0=parameter["W0"],
        wa=parameter["WA"],
        sigma8=parameter["SIGMA8"],
        n_s=parameter["NS"],
        m_nu=parameter["MNU"],
        T_CMB=parameter["TCMB"],
        Omega_k=parameter["OMEGAK"],
        Omega_c=parameter["OMEGAC"],
        Omega_b=parameter["OMEGAB"],
        mass_split="normal",
        transfer_function="eisenstein_hu",
        matter_power_spectrum="halofit",
    )


def build_power_spectrum_panels() -> None:
    """Combine the three power-spectrum plots from ``Power.ipynb``."""

    cosmology = load_cosmology()
    z = np.linspace(0.0, 3.0, 31)
    k = np.logspace(-2.0, 1.0, 101)
    plotted_redshifts = [0.0, 0.5, 1.0, 2.0, 3.0]
    plotted_indices = [
        int(np.flatnonzero(np.isclose(z, z_value))[0])
        for z_value in plotted_redshifts
    ]

    model = NLAModel(
        A0=1.0,
        eta=0.5,
        xi=0.0,
        s=2.0,
        z_q=1.0,
        q=1.0,
        n_star=2.0,
        k_t_star=0.5,
        alpha=0.3,
        m=2.0,
        gamma_t=0.4,
        gamma_n=0.2,
        gamma_alpha=0.0,
        gamma_m=0.0,
        constant=C0,
        z_star=Z_STAR,
    )
    amplitude = model.amplitude_components(cosmology, z, k)["A_IA"]
    matter_power = np.vstack(
        [
            pyccl.nonlin_matter_power(
                cosmology,
                k,
                1.0 / (1.0 + z_value),
            )
            for z_value in z
        ]
    )
    matter_intrinsic_power = amplitude * matter_power
    intrinsic_power = amplitude**2 * matter_power

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.45), sharex=True)
    spectra = (
        (matter_power, r"$P_\delta^{\rm nl}$", "(a) Nonlinear matter"),
        (-matter_intrinsic_power, r"$-P_{\delta I}$", "(b) Matter--intrinsic"),
        (intrinsic_power, r"$P_{II}$", "(c) Intrinsic--intrinsic"),
    )

    for axis, (spectrum, symbol, title) in zip(axes, spectra):
        for index, z_value, colour in zip(
            plotted_indices,
            plotted_redshifts,
            COLOURS,
        ):
            axis.plot(
                k,
                spectrum[index],
                color=colour,
                linewidth=1.5,
                label=rf"$z={z_value:.1f}$",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1.0e-2, 1.0e1)
        axis.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
        axis.set_ylabel(symbol + r"$\ [{\rm Mpc}^{3}]$")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.18, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.25, wspace=0.34)
    figure.savefig(
        FIGURE_PATH / "ia_power_spectra_panels.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def build_redshift_factor_panels() -> None:
    """Combine the redshift and luminosity-like plots from ``Formula.ipynb``."""

    z = np.linspace(0.0, 3.0, 301)
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), sharex=True)

    eta_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for eta, colour in zip(eta_values, COLOURS):
        axes[0].plot(
            z,
            redshift_factor(z, eta=eta, z_star=Z_STAR),
            color=colour,
            linewidth=1.5,
            label=rf"$\eta={eta:+.1f}$",
        )
    axes[0].set_title(r"(a) Broad trend $\eta$", fontsize=10)
    axes[0].set_ylabel(r"$R_z(z)$")

    xi_values = [-1.5, -0.75, 0.0, 0.75, 1.5]
    for xi, colour in zip(xi_values, COLOURS):
        axes[1].plot(
            z,
            luminosity_factor(
                z,
                xi=xi,
                s=2.0,
                z_q=1.5,
                z_star=Z_STAR,
            ),
            color=colour,
            linewidth=1.5,
            label=rf"$\xi={xi:+.2g}$",
        )
    axes[1].set_title(r"(b) Transition strength $\xi$", fontsize=10)
    axes[1].set_ylabel(r"$R_L(z)$")

    z_q_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    for z_q, colour in zip(z_q_values, COLOURS):
        axes[2].plot(
            z,
            luminosity_factor(
                z,
                xi=1.0,
                s=4.0,
                z_q=z_q,
                z_star=Z_STAR,
            ),
            color=colour,
            linewidth=1.5,
            label=rf"$z_q={z_q:.1f}$",
        )
    axes[2].set_title(r"(c) Transition location $z_q$", fontsize=10)
    axes[2].set_ylabel(r"$R_L(z)$")

    for axis in axes:
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.axvline(Z_STAR, color="#999999", linestyle=":", linewidth=0.8)
        axis.set_xlim(0.0, 3.0)
        axis.set_xlabel(r"$z$")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(frameon=False, fontsize=7.3, loc="best")

    figure.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.18, wspace=0.31)
    figure.savefig(
        FIGURE_PATH / "ia_redshift_factors_panels.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Create all Section 3 figures."""

    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    build_power_spectrum_panels()
    build_redshift_factor_panels()


if __name__ == "__main__":
    main()
