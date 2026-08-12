"""
Intrinsic-alignment amplitude factors used in this project.

The module separates the signed NLA amplitude into A_IA(k, z) = - A0 * A_omega(z) * A_theta(k, z). A_omega contains the standard cosmological scaling, A0 is the global IA normalization, and the positive A_theta contains only the model-dependent redshift, luminosity, and scale factors.
"""

from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy
import pyccl

C0 = 0.0134
Z_STAR = 0.5

__all__ = [
    "C0",
    "Z_STAR",
    "NLAModel",
    "amplitude_components",
    "cosmological_factor",
    "luminosity_factor",
    "model_amplitude",
    "pivot_redshift_ratio",
    "redshift_factor",
    "scale_transition",
    "tail_slope",
    "tail_smoothness",
    "transition_sharpness",
    "transition_wavenumber",
]


def _one_dimensional_array(
    values,
    name,
):
    """
    Convert a scalar or one-dimensional input to a finite float array.
    
    Arguments:
        values (float, list, tuple, or numpy.ndarray):
            Scalar or one-dimensional input.
        name (str):
            Name of the input.
    
    Returns:
        array (numpy.ndarray):
            A one-dimensional array of finite floats.
    """
    array = numpy.asarray(values, dtype=float)
    
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array.")
    
    if not numpy.all(numpy.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    
    return array


def _finite_scalar(
    value,
    name,
):
    """
    Return a finite scalar float.
    
    Arguments:
        value (float, int, or numpy.ndarray):
            Scalar input.
        name (str):
            Name of the input.
    
    Returns:
        scalar (float):
            A finite scalar float.
    """
    try:
        array = numpy.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real scalar.") from error
    
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    
    scalar = float(array)
    
    if not numpy.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    
    return scalar


def pivot_redshift_ratio(
    z,
    z_star=Z_STAR,
):
    """
    Calculate the redshift coordinate normalized at the pivot.
    
    The returned quantity is r_star(z) = (1 + z) / (1 + z_star), so r_star(z_star) = 1. Scalars remain scalars and arrays retain their input shape.
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        ratio (numpy.floating or numpy.ndarray):
            The redshift coordinate normalized at the pivot, with the same shape as z.
    """
    z_array = numpy.asarray(z, dtype=float)
    z_star = _finite_scalar(z_star, "z_star")
    
    if not numpy.all(numpy.isfinite(z_array)):
        raise ValueError("z must contain only finite values.")
    if numpy.any(z_array <= -1.0) or z_star <= -1.0:
        raise ValueError("z and z_star must satisfy z > -1.")
    
    return (1.0 + z_array) / (1.0 + z_star)


def cosmological_factor(
    cosmo,
    z,
    constant=C0,
):
    """
    Calculate the standard NLA cosmological factor.
    
    The returned quantity is A_omega(z) = constant * Omega_m / D(z), where D(z) is the linear growth factor. Scalars remain scalars and arrays retain their input shape.
    
    Arguments:
        cosmo (pyccl.Cosmology):
            Cosmology used to evaluate the linear growth factor and Omega_m.
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        constant (float or int):
            Conventional IA normalization. The default is C0 = 0.0134.
    
    Returns:
        A_omega (numpy.floating or numpy.ndarray):
            The cosmological factor A_omega(z), with the same shape as z.
    """
    z_array = numpy.asarray(z, dtype=float)
    constant = _finite_scalar(constant, "constant")
    omega_m = _finite_scalar(cosmo["Omega_m"], "Omega_m")
    
    if not numpy.all(numpy.isfinite(z_array)):
        raise ValueError("z must contain only finite values.")
    if numpy.any(z_array <= -1.0):
        raise ValueError("All redshifts must satisfy z > -1.")
    if omega_m <= 0.0:
        raise ValueError("Omega_m must be positive.")
    if constant <= 0.0:
        raise ValueError("Constant must be positive.")
    
    a_array = 1.0 / (1.0 + z_array)
    growth = numpy.asarray(
        pyccl.growth_factor(cosmo, a_array),
        dtype=float,
    )
    
    if not numpy.all(numpy.isfinite(growth)):
        raise ValueError("The linear growth factor must contain only finite values.")
    if numpy.any(growth <= 0.0):
        raise ValueError("The linear growth factor must be positive.")
    
    with numpy.errstate(over="ignore", divide="ignore", invalid="ignore"):
        A_omega = constant * omega_m / growth
    
    if not numpy.all(numpy.isfinite(A_omega)):
        raise ValueError("A_omega must contain only finite values.")
    
    return A_omega


def redshift_factor(
    z,
    eta=0.0,
    z_star=Z_STAR,
):
    """
    Calculate the normalized power-law redshift factor.
    
    The returned quantity is R_z(z; eta, z_star) = (1 + z)**eta / (1 + z_star)**eta, so R_z(z_star; eta, z_star) = 1. Scalars remain scalars and arrays retain their input shape.
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        eta (float or int):
            Additional power-law index controlling redshift evolution.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        R_z (numpy.floating or numpy.ndarray):
            The redshift factor R_z(z; eta, z_star), with the same shape as z.
    """
    eta = _finite_scalar(eta, "eta")
    r_star = pivot_redshift_ratio(z, z_star=z_star)
    
    if eta == 0.0:
        R_z = numpy.ones_like(r_star)
    else:
        with numpy.errstate(over="ignore", invalid="ignore"):
            R_z = r_star**eta
    
    if not numpy.all(numpy.isfinite(R_z)):
        raise ValueError("R_z must contain only finite values.")
    if numpy.any(R_z <= 0.0):
        raise ValueError("R_z must be positive.")
    
    return R_z


def luminosity_factor(
    z,
    xi=1.0,
    s=2.0,
    z_q=1.5,
    z_star=Z_STAR,
):
    """
    Calculate the normalized smooth broken power-law luminosity factor.
    
    The returned quantity is R_L(z; xi, s, z_q, z_star) = ((1 + (r_star(z) / r_q_star)**s) / (1 + (1 / r_q_star)**s))**(xi / s), where r_q_star = r_star(z_q), so R_L(z_star; xi, s, z_q, z_star) = 1. Scalars remain scalars and arrays retain their input shape.
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        xi (float or int):
            Change in logarithmic slope across the transition.
        s (float or int):
            Transition sharpness. It must be positive.
        z_q (float or int):
            Redshift at the center of the transition. It must be greater than -1.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        R_L (numpy.floating or numpy.ndarray):
            The luminosity factor R_L(z; xi, s, z_q), with the same shape as z.
    """
    xi = _finite_scalar(xi, "xi")
    s = _finite_scalar(s, "s")
    z_q = _finite_scalar(z_q, "z_q")
    z_star = _finite_scalar(z_star, "z_star")
    
    if s <= 0.0:
        raise ValueError("s must be positive.")
    if z_q <= -1.0:
        raise ValueError("z_q must satisfy z_q > -1.")
    
    r_star = pivot_redshift_ratio(z, z_star=z_star)
    r_q_star = pivot_redshift_ratio(z_q, z_star=z_star)
    
    log_ratio = numpy.log(r_star) - numpy.log(r_q_star)
    log_star_ratio = -numpy.log(r_q_star)
    
    # Evaluate log(1 + ratio**s) without numerical overflow.
    log_numerator = numpy.logaddexp(0.0, s * log_ratio)
    log_denominator = numpy.logaddexp(0.0, s * log_star_ratio)
    
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        R_L = numpy.exp(
            (xi / s) * (log_numerator - log_denominator)
        )
    
    if not numpy.all(numpy.isfinite(R_L)):
        raise ValueError("R_L must contain only finite values.")
    if numpy.any(R_L <= 0.0):
        raise ValueError("R_L must be positive.")
    
    return R_L


def transition_wavenumber(
    z,
    k_t_star=0.5,
    gamma_t=0.4,
    z_star=Z_STAR,
):
    """
    Calculate the transition wavenumber.
    
    The returned quantity is k_t(z) = k_t_star * r_star(z)**gamma_t, so k_t(z_star) = k_t_star. Scalars remain scalars and arrays retain their input shape.
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        k_t_star (float or int):
            Transition wavenumber at z_star. It must be positive.
        gamma_t (float or int):
            Redshift-evolution index of the transition wavenumber.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        k_t (numpy.floating or numpy.ndarray):
            The transition wavenumber k_t(z), with the same shape as z.
    """
    k_t_star = _finite_scalar(k_t_star, "k_t_star")
    gamma_t = _finite_scalar(gamma_t, "gamma_t")
    
    if k_t_star <= 0.0:
        raise ValueError("k_t_star must be positive.")
    
    r_star = pivot_redshift_ratio(z, z_star=z_star)
    
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        k_t = k_t_star * r_star**gamma_t
    
    if not numpy.all(numpy.isfinite(k_t)):
        raise ValueError("k_t must contain only finite values.")
    if numpy.any(k_t <= 0.0):
        raise ValueError("k_t must be positive.")
    
    return k_t


def transition_sharpness(
    z,
    n_star=2.0,
    gamma_n=0.2,
    z_star=Z_STAR,
):
    """
    Calculate the transition sharpness.
    
    The returned quantity is n(z) = n_star * r_star(z)**gamma_n, so n(z_star) = n_star. Scalars remain scalars and arrays retain their input shape.
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        n_star (float or int):
            Transition sharpness at z_star. It must be positive.
        gamma_n (float or int):
            Redshift-evolution index of the transition sharpness.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        n (numpy.floating or numpy.ndarray):
            The transition sharpness n(z), with the same shape as z.
    """
    n_star = _finite_scalar(n_star, "n_star")
    gamma_n = _finite_scalar(gamma_n, "gamma_n")
    
    if n_star <= 0.0:
        raise ValueError("n_star must be positive.")
    
    r_star = pivot_redshift_ratio(z, z_star=z_star)
    
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        n = n_star * r_star**gamma_n
    
    if not numpy.all(numpy.isfinite(n)):
        raise ValueError("n must contain only finite values.")
    if numpy.any(n <= 0.0):
        raise ValueError("n must be positive.")
    
    return n


def tail_slope(
    z,
    alpha=0.3,
    gamma_alpha=0.0,
    z_star=Z_STAR,
):
    """
    Calculate the evolving high-k logarithmic slope.
    
    The returned quantity is alpha(z) = alpha * r_star(z)**gamma_alpha.  Consequently, alpha is the slope at the pivot redshift and its sign is preserved at every redshift.  Setting gamma_alpha = 0 recovers a constant slope.
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        alpha (float or int):
            Logarithmic slope at z_star.
        gamma_alpha (float or int):
            Redshift-evolution index of the logarithmic slope.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        alpha_z (numpy.floating or numpy.ndarray):
            The logarithmic slope alpha(z), with the same shape as z.
    """
    alpha = _finite_scalar(alpha, "alpha")
    gamma_alpha = _finite_scalar(gamma_alpha, "gamma_alpha")
    
    r_star = pivot_redshift_ratio(z, z_star=z_star)
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        alpha_z = alpha * r_star**gamma_alpha
    
    if not numpy.all(numpy.isfinite(alpha_z)):
        raise ValueError("alpha(z) must contain only finite values.")
    
    return alpha_z


def tail_smoothness(
    z,
    m=2.0,
    gamma_m=0.0,
    z_star=Z_STAR,
):
    """
    Calculate the evolving positive smoothness of the high-k tail.
    
    The returned quantity is m(z) = m * r_star(z)**gamma_m, so m is the smoothness at the pivot redshift.  The multiplicative evolution keeps m(z) positive whenever m is positive.
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Redshift value or array. Every value must be greater than -1 and finite.
        m (float or int):
            Smoothness at z_star. It must be positive.
        gamma_m (float or int):
            Redshift-evolution index of the smoothness.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        m_z (numpy.floating or numpy.ndarray):
            The smoothness m(z), with the same shape as z.
    """
    m = _finite_scalar(m, "m")
    gamma_m = _finite_scalar(gamma_m, "gamma_m")
    
    if m <= 0.0:
        raise ValueError("m must be positive.")
    
    r_star = pivot_redshift_ratio(z, z_star=z_star)
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        m_z = m * r_star**gamma_m
    
    if not numpy.all(numpy.isfinite(m_z)):
        raise ValueError("m(z) must contain only finite values.")
    
    if numpy.any(m_z <= 0.0):
        raise ValueError("m(z) must be positive.")
    
    return m_z


def scale_transition(
    z,
    k,
    *,
    q=1.0,
    n_star=2.0,
    k_t_star=0.5,
    alpha=0.3,
    m=2.0,
    gamma_t=0.4,
    gamma_n=0.2,
    gamma_alpha=0.0,
    gamma_m=0.0,
    z_star=Z_STAR,
):
    """
    Calculate the joint scale- and redshift-dependent factor S(k, z).
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Scalar or one-dimensional redshift grid.
        k (float, list, tuple, or numpy.ndarray):
            Wavenumber value or array in Mpc^-1. Every value must be positive.
        q (float or int):
            Strength of the correction at the transition.
        n_star (float or int):
            Transition sharpness at z_star. It must be positive.
        k_t_star (float or int):
            Transition wavenumber at z_star, in Mpc^-1. It must be positive.
        alpha (float or int):
            Logarithmic slope of the high-k tail.
        m (float or int):
            Smoothness of the high-k tail. It must be positive.
        gamma_t (float or int):
            Redshift-evolution index of k_t(z).
        gamma_n (float or int):
            Redshift-evolution index of n(z).
        gamma_alpha (float or int):
            Redshift-evolution index of alpha(z).
        gamma_m (float or int):
            Redshift-evolution index of m(z).
        z_star (float or int):
            Pivot redshift.
    
    Returns:
        S_k_z (numpy.ndarray):
            The scale factor S(k, z) with shape (N_z, N_k).
    """
    z_array = _one_dimensional_array(z, "z")
    k_array = _one_dimensional_array(k, "k")
    q = _finite_scalar(q, "q")
    
    if numpy.any(z_array <= -1.0):
        raise ValueError("All redshifts must satisfy z > -1.")
    if numpy.any(k_array <= 0.0):
        raise ValueError("All wavenumbers must be positive.")
    
    k_t_z = transition_wavenumber(
        z_array,
        k_t_star=k_t_star,
        gamma_t=gamma_t,
        z_star=z_star,
    )
    
    n_z = transition_sharpness(
        z_array,
        n_star=n_star,
        gamma_n=gamma_n,
        z_star=z_star,
    )
    
    alpha_z = tail_slope(
        z_array,
        alpha=alpha,
        gamma_alpha=gamma_alpha,
        z_star=z_star,
    )
    
    m_z = tail_smoothness(
        z_array,
        m=m,
        gamma_m=gamma_m,
        z_star=z_star,
    )
    
    if q == 0.0:
        return numpy.ones((len(z_array), len(k_array)), dtype=float)
    
    log_k_ratio = (
        numpy.log(k_array)[None, :]
        - numpy.log(k_t_z)[:, None]
    )
    
    log_transition_power = n_z[:, None] * log_k_ratio
    
    transition_weight = numpy.exp(
        -numpy.logaddexp(0.0, -log_transition_power)
    )
    
    log_tail_factor = (alpha_z / m_z)[:, None] * (
        numpy.logaddexp(0.0, m_z[:, None] * log_k_ratio)
        - numpy.log(2.0)
    )
    
    with numpy.errstate(over="ignore", invalid="ignore"):
        tail_factor = numpy.exp(log_tail_factor)
        S_k_z = 1.0 + q * transition_weight * tail_factor
    
    if not numpy.all(numpy.isfinite(S_k_z)):
        raise ValueError("S_k_z must contain only finite values.")
    
    if numpy.any(S_k_z <= 0.0):
        raise ValueError(
            "S_k_z must remain positive on the evaluated (z, k) grid."
        )
    
    return S_k_z


def model_amplitude(
    z,
    k,
    *,
    eta=0.0,
    xi=1.0,
    s=2.0,
    z_q=1.5,
    q=1.0,
    n_star=2.0,
    k_t_star=0.5,
    alpha=0.3,
    m=2.0,
    gamma_t=0.4,
    gamma_n=0.2,
    gamma_alpha=0.0,
    gamma_m=0.0,
    z_star=Z_STAR,
):
    """
    Calculate the positive model-dependent shape amplitude A_theta(k, z).
    
    Arguments:
        z (float, list, tuple, or numpy.ndarray):
            Scalar or one-dimensional redshift grid.
        k (float, list, tuple, or numpy.ndarray):
            Scalar or one-dimensional wavenumber grid in Mpc^-1.
        eta (float or int):
            Additional redshift power-law index.
        xi (float or int):
            Luminosity-factor change in logarithmic slope.
        s (float or int):
            Luminosity-transition sharpness.
        z_q (float or int):
            Luminosity-transition redshift.
        q (float or int):
            Strength of the nonlinear scale correction.
        n_star (float or int):
            Scale-transition sharpness at z_star.
        k_t_star (float or int):
            Scale-transition wavenumber at z_star, in Mpc^-1.
        alpha (float or int):
            Logarithmic slope of the high-k tail.
        m (float or int):
            Smoothness of the high-k tail.
        gamma_t (float or int):
            Redshift-evolution index of the transition wavenumber.
        gamma_n (float or int):
            Redshift-evolution index of the transition sharpness.
        gamma_alpha (float or int):
            Redshift-evolution index of the high-k slope.
        gamma_m (float or int):
            Redshift-evolution index of the tail smoothness.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        A_theta (numpy.ndarray):
            The model-dependent amplitude A_theta(k, z), with shape (N_z, N_k).
    """
    z_array = _one_dimensional_array(z, "z")
    k_array = _one_dimensional_array(k, "k")
    R_z = redshift_factor(
        z_array,
        eta=eta,
        z_star=z_star,
    )
    
    R_L = luminosity_factor(
        z_array,
        xi=xi,
        s=s,
        z_q=z_q,
        z_star=z_star,
    )
    
    S_k_z = scale_transition(
        z_array,
        k_array,
        q=q,
        n_star=n_star,
        k_t_star=k_t_star,
        alpha=alpha,
        m=m,
        gamma_t=gamma_t,
        gamma_n=gamma_n,
        gamma_alpha=gamma_alpha,
        gamma_m=gamma_m,
        z_star=z_star,
    )
    
    with numpy.errstate(over="ignore", invalid="ignore"):
        A_theta = (R_z * R_L)[:, None] * S_k_z
    
    if not numpy.all(numpy.isfinite(A_theta)):
        raise ValueError("A_theta must contain only finite values.")
    if numpy.any(A_theta <= 0.0):
        raise ValueError("A_theta must remain positive on the evaluated (k, z) grid.")
    
    return A_theta


def amplitude_components(
    cosmo,
    z,
    k,
    *,
    A0=1.0,
    eta=0.0,
    xi=1.0,
    s=2.0,
    z_q=1.5,
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
):
    """
    Calculate the factorized NLA amplitude components.
    
    Arguments:
        cosmo (pyccl.Cosmology):
            Cosmology used for A_omega.
        z (float, list, tuple, or numpy.ndarray):
            Scalar or one-dimensional redshift grid.
        k (float, list, tuple, or numpy.ndarray):
            Scalar or one-dimensional wavenumber grid in Mpc^-1.
        A0 (float or int):
            Global IA normalization applied outside A_omega and A_theta.
        eta (float or int):
            Additional redshift power-law index.
        xi (float or int):
            Luminosity-factor change in logarithmic slope.
        s (float or int):
            Luminosity-transition sharpness.
        z_q (float or int):
            Luminosity-transition redshift.
        q (float or int):
            Strength of the nonlinear scale correction.
        n_star (float or int):
            Scale-transition sharpness at z_star.
        k_t_star (float or int):
            Scale-transition wavenumber at z_star, in Mpc^-1.
        alpha (float or int):
            Logarithmic slope of the high-k tail.
        m (float or int):
            Smoothness of the high-k tail.
        gamma_t (float or int):
            Redshift-evolution index of the transition wavenumber.
        gamma_n (float or int):
            Redshift-evolution index of the transition sharpness.
        gamma_alpha (float or int):
            Redshift-evolution index of the high-k slope.
        gamma_m (float or int):
            Redshift-evolution index of the tail smoothness.
        constant (float or int):
            Conventional IA normalization.
        z_star (float or int):
            Normalization pivot. It must be greater than -1.
    
    Returns:
        components (dict[str, numpy.ndarray]):
            Dictionary containing A_omega with shape (N_z,) and A_theta(k, z)
            and A_IA(k, z) with shape (N_z, N_k).
            The keys are "A_omega", "A_theta", and "A_IA".
    """
    z_array = _one_dimensional_array(z, "z")
    k_array = _one_dimensional_array(k, "k")
    A0 = _finite_scalar(A0, "A0")
    
    A_omega = cosmological_factor(
        cosmo,
        z_array,
        constant=constant,
    )
    
    A_theta = model_amplitude(
        z_array,
        k_array,
        eta=eta,
        xi=xi,
        s=s,
        z_q=z_q,
        q=q,
        n_star=n_star,
        k_t_star=k_t_star,
        alpha=alpha,
        m=m,
        gamma_t=gamma_t,
        gamma_n=gamma_n,
        gamma_alpha=gamma_alpha,
        gamma_m=gamma_m,
        z_star=z_star,
    )
    
    with numpy.errstate(over="ignore", invalid="ignore"):
        A_IA = - A0 * A_omega[:, None] * A_theta
    
    for name, component in (
        ("A_theta", A_theta),
        ("A_IA", A_IA),
    ):
        if not numpy.all(numpy.isfinite(component)):
            raise ValueError(f"{name} must contain only finite values.")
    
    return {
        "A_omega": A_omega,
        "A_theta": A_theta,
        "A_IA": A_IA,
    }


@dataclass(frozen=True)
class NLAModel:
    """
    Immutable parameter container and interface for the NLA model.
    
    The class stores the IA normalization and nuisance parameters. Cosmology
    and evaluation coordinates remain external inputs to its methods. The
    module-level functions are the canonical implementations, and each method
    delegates to them so every mathematical definition lives in one place.
    """
    SHAPE_PARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
        "eta",
        "xi",
        "s",
        "z_q",
        "q",
        "n_star",
        "k_t_star",
        "alpha",
        "m",
        "gamma_t",
        "gamma_n",
        "gamma_alpha",
        "gamma_m",
    )
    FULL_PARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
        "A0",
        *SHAPE_PARAMETER_NAMES,
    )
    # Backwards-compatible name for the complete physical parameter vector.
    SAMPLED_PARAMETER_NAMES: ClassVar[tuple[str, ...]] = FULL_PARAMETER_NAMES
    
    # Model parameters.
    A0: float = 1.0
    eta: float = 0.0
    xi: float = 0.0
    s: float = 2.0
    z_q: float = 1.0
    q: float = 1.0
    n_star: float = 2.0
    k_t_star: float = 0.5
    alpha: float = 0.3
    m: float = 2.0
    gamma_t: float = 0.4
    gamma_n: float = 0.2
    gamma_alpha: float = 0.0
    gamma_m: float = 0.0
    constant: float = C0
    z_star: float = Z_STAR
    
    # Parameter validation.
    def __post_init__(self):
        """
        Normalize scalar fields and validate the complete model state.
        """
        for name in self.FULL_PARAMETER_NAMES + ("z_star", "constant"):
            scalar = _finite_scalar(getattr(self, name), name)
            object.__setattr__(self, name, scalar)
        
        # Validate the model parameters.
        if self.z_star <= -1.0 or self.z_q <= -1.0:
            raise ValueError("z_star and z_q must satisfy z > -1.")
        if self.s <= 0.0:
            raise ValueError("s must be positive.")
        if self.n_star <= 0.0:
            raise ValueError("n_star must be positive.")
        if self.k_t_star <= 0.0:
            raise ValueError("k_t_star must be positive.")
        if self.m <= 0.0:
            raise ValueError("m must be positive.")
        if self.constant <= 0.0:
            raise ValueError("constant must be positive.")
    
    # General model factors.
    def redshift_ratio(self, z):
        """
        Return r_star(z) for this model's pivot redshift.
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            r_star (numpy.floating or numpy.ndarray):
                The redshift ratio r_star(z), with the same shape as z.
        """
        return pivot_redshift_ratio(z, z_star=self.z_star)
    
    def cosmological_factor(self, cosmo, z):
        """
        Return A_omega(z) for an externally supplied cosmology.
        
        Arguments:
            cosmo (pyccl.Cosmology):
                Cosmology used to evaluate the linear growth factor and Omega_m.
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            A_omega (numpy.floating or numpy.ndarray):
                The cosmological factor A_omega(z), with the same shape as z.
        """
        return cosmological_factor(
            cosmo,
            z,
            constant=self.constant,
        )
    
    # Redshift-dependent model factor.
    def redshift_factor(self, z):
        """
        Return R_z(z) = r_star(z)**eta.
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            R_z (numpy.floating or numpy.ndarray):
                The redshift factor R_z(z), with the same shape as z.
        """
        return redshift_factor(z, eta=self.eta, z_star=self.z_star)
    
    # Luminosity-dependent model factor.
    def luminosity_factor(self, z):
        """
        Return the pivot-normalized luminosity-population factor.
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            R_L (numpy.floating or numpy.ndarray):
                The luminosity factor R_L(z), with the same shape as z.
        """
        return luminosity_factor(
            z,
            xi=self.xi,
            s=self.s,
            z_q=self.z_q,
            z_star=self.z_star,
        )
    
    # Scale-transition functions.
    def transition_wavenumber(self, z):
        """
        Return the evolving transition wavenumber k_t(z).
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            k_t (numpy.floating or numpy.ndarray):
                The transition wavenumber k_t(z), with the same shape as z.
        """
        return transition_wavenumber(
            z,
            k_t_star=self.k_t_star,
            gamma_t=self.gamma_t,
            z_star=self.z_star,
        )
    
    def transition_sharpness(self, z):
        """
        Return the evolving transition sharpness n(z).
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            n (numpy.floating or numpy.ndarray):
                The transition sharpness n(z), with the same shape as z.
        """
        return transition_sharpness(
            z,
            n_star=self.n_star,
            gamma_n=self.gamma_n,
            z_star=self.z_star,
        )
    
    def tail_slope(self, z):
        """
        Return the evolving high-k logarithmic slope alpha(z).
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            alpha_z (numpy.floating or numpy.ndarray):
                The tail slope alpha(z), with the same shape as z.
        """
        return tail_slope(
            z,
            alpha=self.alpha,
            gamma_alpha=self.gamma_alpha,
            z_star=self.z_star,
        )
    
    def tail_smoothness(self, z):
        """
        Return the evolving positive high-k smoothness m(z).
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
        
        Returns:
            m_z (numpy.floating or numpy.ndarray):
                The tail smoothness m(z), with the same shape as z.
        """
        return tail_smoothness(
            z,
            m=self.m,
            gamma_m=self.gamma_m,
            z_star=self.z_star,
        )
    
    # Scale-dependent model factor.
    def scale_dependence(self, z, k):
        """
        Return S_k_z with shape (N_z, N_k).
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
            k (float, list, tuple, or numpy.ndarray):
                Wavenumber value or array. Every value must be positive.
        
        Returns:
            S_k_z (numpy.ndarray):
                The scale dependence S_k_z(k, z), with shape (N_z, N_k).
        """
        return scale_transition(
            z,
            k,
            q=self.q,
            n_star=self.n_star,
            k_t_star=self.k_t_star,
            alpha=self.alpha,
            m=self.m,
            gamma_t=self.gamma_t,
            gamma_n=self.gamma_n,
            gamma_alpha=self.gamma_alpha,
            gamma_m=self.gamma_m,
            z_star=self.z_star,
        )
    
    # Model-amplitude functions.
    def model_amplitude(self, z, k):
        """
        Return the positive shape amplitude A_theta, excluding A0.
        
        Arguments:
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
            k (float, list, tuple, or numpy.ndarray):
                Wavenumber value or array. Every value must be positive.
        
        Returns:
            A_theta (numpy.ndarray):
                The positive model-dependent amplitude, with shape (N_z, N_k).
        """
        return model_amplitude(
            z,
            k,
            eta=self.eta,
            xi=self.xi,
            s=self.s,
            z_q=self.z_q,
            q=self.q,
            n_star=self.n_star,
            k_t_star=self.k_t_star,
            alpha=self.alpha,
            m=self.m,
            gamma_t=self.gamma_t,
            gamma_n=self.gamma_n,
            gamma_alpha=self.gamma_alpha,
            gamma_m=self.gamma_m,
            z_star=self.z_star,
        )
    
    def amplitude_components(self, cosmo, z, k):
        """
        Return the factorized amplitude components for a cosmology.
        
        Arguments:
            cosmo (pyccl.Cosmology):
                Cosmology used for A_omega.
            z (float, list, tuple, or numpy.ndarray):
                Redshift value or array. Every value must be greater than -1 and finite.
            k (float, list, tuple, or numpy.ndarray):
                Wavenumber value or array. Every value must be positive.
        
        Returns:
            components (dict[str, numpy.ndarray]):
                Dictionary containing A_omega with shape (N_z,) and A_theta and
                A_IA with shape (N_z, N_k). The keys are "A_omega", "A_theta",
                and "A_IA".
        """
        return amplitude_components(
            cosmo,
            z,
            k,
            A0=self.A0,
            eta=self.eta,
            xi=self.xi,
            s=self.s,
            z_q=self.z_q,
            q=self.q,
            n_star=self.n_star,
            k_t_star=self.k_t_star,
            alpha=self.alpha,
            m=self.m,
            gamma_t=self.gamma_t,
            gamma_n=self.gamma_n,
            gamma_alpha=self.gamma_alpha,
            gamma_m=self.gamma_m,
            constant=self.constant,
            z_star=self.z_star,
        )
    
    # Parameter-conversion utilities.
    def to_array(self):
        """
        Return the complete sampled parameter vector.
        
        Returns:
            values (numpy.ndarray):
                The complete sampled parameter vector, with shape (N_parameters,).
        """
        return numpy.asarray(
            [getattr(self, name) for name in self.FULL_PARAMETER_NAMES],
            dtype=float,
        )
    
    def to_shape_array(self):
        """
        Return only the shape parameters that determine A_theta.
        
        Returns:
            values (numpy.ndarray):
                The shape parameters that determine A_theta, with shape (N_parameters,).
        """
        return numpy.asarray(
            [getattr(self, name) for name in self.SHAPE_PARAMETER_NAMES],
            dtype=float,
        )
    
    def to_dict(self):
        """
        Return all model fields, including fixed normalization values.
        
        Returns:
            fields (dict[str, float]):
                Dictionary containing all model fields, including fixed normalization values.
        """
        return asdict(self)
    
    @classmethod
    def from_array(cls, values, *, z_star=Z_STAR, constant=C0):
        """
        Construct a model from the complete sampled parameter vector.
        
        Arguments:
            values (numpy.ndarray):
                The complete sampled parameter vector, with shape (N_parameters,).
            z_star (float or int):
                Normalization pivot. It must be greater than -1.
            constant (float or int):
                Conventional IA normalization.
        
        Returns:
            model (NLAModel):
                The constructed model.
        """
        values_array = _one_dimensional_array(values, "values")
        expected_size = len(cls.FULL_PARAMETER_NAMES)
        
        # Validate the input array.
        if len(values_array) != expected_size:
            raise ValueError(f"values must contain exactly {expected_size} parameters.")
        
        # Construct the model.
        parameters = dict(zip(cls.FULL_PARAMETER_NAMES, values_array))
        return cls(
            **parameters,
            z_star=z_star,
            constant=constant,
        )
    
    @classmethod
    def from_shape_array(
        cls,
        values,
        *,
        A0=1.0,
        z_star=Z_STAR,
        constant=C0,
    ):
        """
        Construct a model from the shape parameters.
        
        Arguments:
            values (numpy.ndarray):
                The shape parameters that determine A_theta, with shape (N_parameters,).
            A0 (float or int):
                The amplitude of the model.
            z_star (float or int):
                Normalization pivot. It must be greater than -1.
            constant (float or int):
                Conventional IA normalization.
        
        Returns:
            model (NLAModel):
                The constructed model.
        """
        values_array = _one_dimensional_array(values, "values")
        expected_size = len(cls.SHAPE_PARAMETER_NAMES)
        if len(values_array) != expected_size:
            raise ValueError(f"values must contain exactly {expected_size} parameters.")
        
        parameters = dict(zip(cls.SHAPE_PARAMETER_NAMES, values_array))
        return cls(
            A0=A0,
            **parameters,
            z_star=z_star,
            constant=constant,
        )
