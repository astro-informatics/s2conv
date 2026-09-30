r"""
Spin spherical harmonic and Wigner transforms, through s2fft, with a choice of backend.

The backends are:

- ``"jax"``: s2fft's JAX transforms with on-the-fly Wigner recursions.  They are
  JIT compiled, differentiable and run on GPUs, with memory :math:`\mathcal{O}(L^2)`,
  but their recursions are accurate only for spins and azimuthal band-limits below
  :data:`RECURSION_LIMIT`.
- ``"precompute"``: s2fft's precompute transforms.  The Wigner functions are computed
  once, by a recursion that is stable at every spin, and cached, so that the transforms
  are accurate for any spin and azimuthal band-limit at the price of
  :math:`\mathcal{O}(L^3)` (spherical) or :math:`\mathcal{O}(NL^3)` (Wigner) memory.  They
  are JIT compiled, differentiable and run on GPUs.
- ``"jax_ssht"``: s2fft's differentiable JAX frontend to the SSHT C library, which is
  accurate at every spin but runs on the CPU only (requires ``pyssht`` and ``ducc0``).
- ``"numpy"``: s2fft's NumPy reference transforms.

All transforms are exact, up to floating-point error, for band-limited signals.  All
backends except ``"numpy"`` support reverse-mode differentiation.  Forward mode
(``jax.jvp``) is supported by ``"precompute"``, and by ``"jax"`` with s2fft newer than
1.4.0 (the JAX transforms of s2fft 1.4.0 define only a custom VJP); ``"jax_ssht"``
supports reverse mode only.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from s2fft.precompute_transforms import construct as _construct
from s2fft.precompute_transforms import spherical as _precompute_spherical
from s2fft.precompute_transforms import wigner as _precompute_wigner
from s2fft.sampling import s2_samples

METHODS = ("jax", "precompute", "jax_ssht", "numpy")
"""Transform backends: s2fft's JAX, precompute, SSHT and NumPy transforms."""

RECURSION_LIMIT = 8
"""Spins and azimuthal band-limits from which the recursive backends lose accuracy."""


def _check_method(method: str, spin: int = 0, N: int = 1) -> None:
    if method not in METHODS:
        raise ValueError(f"Method {method!r} not recognised; choose from {METHODS}.")
    if method in ("jax", "numpy") and (
        abs(spin) >= RECURSION_LIMIT or N >= RECURSION_LIMIT
    ):
        raise ValueError(
            f"The {method!r} backend is accurate only for |spin| and N below "
            f"{RECURSION_LIMIT} (got spin {spin}, N {N}); use method='precompute' or "
            "method='jax_ssht'."
        )


def _fibre_weight(L: int) -> np.ndarray:
    # A function on the rotation group with N = 1 is constant on the fibres, and its
    # Wigner coefficients are F^l_{m0} = 2 pi sqrt(4 pi / (2l + 1)) f_lm, with f_lm the
    # scalar harmonic coefficients of its values on the sphere.
    return 2 * np.pi * np.sqrt(4 * np.pi / (2 * np.arange(L) + 1))


@functools.cache
def spherical_kernel(
    L: int,
    spin: int = 0,
    forward: bool = True,
    sampling: str = "mw",
    reality: bool = False,
    nside: int = None,
):
    r"""
    Cached kernel of the precomputed spin spherical harmonic transform.

    Args:
        L (int): Harmonic band-limit.
        spin (int, optional): Spin. Defaults to 0.
        forward (bool, optional): Kernel of the forward (True) or inverse transform.
            Defaults to True.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        reality (bool, optional): Whether the signal is real. Defaults to False.
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        jnp.ndarray: The kernel, of size :math:`\mathcal{O}(L^3)`.

    """
    # s2fft's NumPy construction is several times faster than its JAX construction and
    # agrees to machine precision.  The array is created eagerly even when first
    # requested while tracing, so that the cache never holds a tracer.
    kernel = _construct.spin_spherical_kernel(
        L, spin, reality, sampling, nside, forward
    )
    with jax.ensure_compile_time_eval():
        return jnp.asarray(kernel)


@functools.cache
def wigner_kernel(
    L: int,
    N: int,
    forward: bool = True,
    sampling: str = "mw",
    reality: bool = False,
    nside: int = None,
):
    r"""
    Cached kernel of the precomputed Wigner transform.

    Args:
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        forward (bool, optional): Kernel of the forward (True) or inverse transform.
            Defaults to True.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        reality (bool, optional): Whether the signal is real. Defaults to False.
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        jnp.ndarray: The kernel, of size :math:`\mathcal{O}(NL^3)`.

    """
    kernel = _construct.wigner_kernel(L, N, reality, sampling, nside, forward)
    with jax.ensure_compile_time_eval():
        return jnp.asarray(kernel)


def forward(
    f,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    method: str = "jax",
    reality: bool = False,
    nside: int = None,
):
    """
    Forward spin spherical harmonic transform.

    Args:
        f: Samples of a spin-:math:`s` field on the sphere.
        L (int): Harmonic band-limit.
        spin (int, optional): Spin. Defaults to 0.
        sampling (str, optional): Sampling scheme, one of {"mw", "mwss", "dh", "gl",
            "healpix"}. Defaults to "mw".
        method (str, optional): Backend, one of :data:`METHODS`. Defaults to "jax".
        reality (bool, optional): Whether the signal is real. Defaults to False.
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        Harmonic coefficients ``flm`` with shape ``[L, 2L - 1]``.

    """
    _check_method(method, spin=spin)
    if method == "jax":
        return s2fft.forward_jax(f, L, spin, nside, sampling, reality)
    if method == "precompute":
        kernel = spherical_kernel(L, spin, True, sampling, reality, nside)
        return _precompute_spherical.forward(
            f, L, spin, kernel, sampling, reality, "jax", nside
        )
    if method == "numpy":
        f = np.asarray(f)
    return s2fft.forward(
        f, L, spin=spin, nside=nside, sampling=sampling, method=method, reality=reality
    )


def inverse(
    flm,
    L: int,
    spin: int = 0,
    sampling: str = "mw",
    method: str = "jax",
    reality: bool = False,
    nside: int = None,
):
    """
    Inverse spin spherical harmonic transform.

    Args:
        flm: Harmonic coefficients with shape ``[L, 2L - 1]``.
        L (int): Harmonic band-limit.
        spin (int, optional): Spin. Defaults to 0.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        method (str, optional): Backend, one of :data:`METHODS`. Defaults to "jax".
        reality (bool, optional): Whether the signal is real. Defaults to False.
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        Samples of the spin-:math:`s` field on the sphere.

    """
    _check_method(method, spin=spin)
    if method == "jax":
        return s2fft.inverse_jax(flm, L, spin, nside, sampling, reality)
    if method == "precompute":
        kernel = spherical_kernel(L, spin, False, sampling, reality, nside)
        return _precompute_spherical.inverse(
            flm, L, spin, kernel, sampling, reality, "jax", nside
        )
    if method == "numpy":
        # The NumPy transforms assign in place, which JAX arrays do not support.
        flm = np.asarray(flm)
    return s2fft.inverse(
        flm,
        L,
        spin=spin,
        nside=nside,
        sampling=sampling,
        method=method,
        reality=reality,
    )


def wigner_forward(
    f,
    L: int,
    N: int,
    sampling: str = "mw",
    method: str = "jax",
    reality: bool = False,
    nside: int = None,
):
    r"""
    Forward Wigner transform, :math:`F^{\ell}_{mn} = \int F\, D^{\ell}_{mn}\,\mathrm{d}\mu` (Eq. 7).

    Args:
        f: Samples on the rotation group with shape ``[2N - 1, ...]``, the first axis
            indexing the third Euler angle :math:`\gamma_j = 2\pi j/(2N-1)`.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        method (str, optional): Backend, one of :data:`METHODS`. Defaults to "jax".
        reality (bool, optional): Whether the signal is real. Defaults to False.
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        Wigner coefficients ``flmn`` with shape ``[2N - 1, L, 2L - 1]``.

    """
    _check_method(method, N=N)
    if method == "jax":
        return s2fft.wigner.forward_jax(f, L, N, nside, sampling, reality)
    if method == "precompute":
        if N == 1:
            # s2fft's precomputed Wigner transforms drop the axis of a single third
            # Euler angle under MW and MWSS sampling; use the scalar transform instead.
            flm = forward(f[0], L, 0, sampling, method, reality, nside)
            return (flm * _fibre_weight(L)[:, None])[None]
        kernel = wigner_kernel(L, N, True, sampling, reality, nside)
        return _precompute_wigner.forward(
            f, L, N, kernel, sampling, reality, "jax", nside
        )
    if method == "numpy":
        f = np.asarray(f)
    return s2fft.wigner.forward(
        f, L, N, nside=nside, sampling=sampling, method=method, reality=reality
    )


def wigner_inverse(
    flmn,
    L: int,
    N: int,
    sampling: str = "mw",
    method: str = "jax",
    reality: bool = False,
    nside: int = None,
):
    r"""
    Inverse Wigner transform, :math:`F = \sum_\ell \frac{2\ell+1}{8\pi^2} \sum_{m,n} F^{\ell}_{mn} D^{\ell *}_{mn}` (Eq. 7).

    Args:
        flmn: Wigner coefficients with shape ``[2N - 1, L, 2L - 1]``.
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        method (str, optional): Backend, one of :data:`METHODS`. Defaults to "jax".
        reality (bool, optional): Whether the signal is real. Defaults to False.
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        Samples on the rotation group with shape ``[2N - 1, ...]``.

    """
    _check_method(method, N=N)
    if method == "jax":
        return s2fft.wigner.inverse_jax(flmn, L, N, nside, sampling, reality)
    if method == "precompute":
        if N == 1:
            flm = flmn[0] / _fibre_weight(L)[:, None]
            return inverse(flm, L, 0, sampling, method, reality, nside)[None]
        kernel = wigner_kernel(L, N, False, sampling, reality, nside)
        return _precompute_wigner.inverse(
            flmn, L, N, kernel, sampling, reality, "jax", nside
        )
    if method == "numpy":
        flmn = np.asarray(flmn)
    return s2fft.wigner.inverse(
        flmn, L, N, nside=nside, sampling=sampling, method=method, reality=reality
    )


def sphere_angles(L: int, sampling: str = "mw"):
    """
    Colatitudes and longitudes of an equiangular sampling of the sphere.

    Args:
        L (int): Harmonic band-limit.
        sampling (str, optional): Equiangular sampling scheme, one of {"mw", "mwss",
            "dh", "gl"}. Defaults to "mw".

    Returns:
        tuple: Arrays of colatitudes and longitudes.

    """
    if sampling.lower() == "healpix":
        raise ValueError("sphere_angles supports equiangular samplings only.")
    return s2_samples.thetas(L, sampling), s2_samples.phis_equiang(L, sampling)
