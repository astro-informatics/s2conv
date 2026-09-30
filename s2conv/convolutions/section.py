r"""
Evaluation of lifted convolutions on a section of the bundle (Sec. VII A).

Anisotropic constructions that keep the output of a directional kernel on the sphere
evaluate a lifted convolution on a section :math:`\sigma(x) = s(x) R_z(\chi_\sigma(x))`.  For a
kernel whose column :math:`k = -s` is supported on :math:`|n| < N`, this is a sum of spin
components in the frame of the section (Proposition VII.1, Eq. 63):

.. math::

    E_\sigma[\Psi]\, {}_sf = \sum_{|s'| < N} \frac{e^{-\mathrm{i}s'\chi_\sigma}}{\sqrt{2\pi}}\,
    \mathcal{C}_{s\to s'}[\Psi]\, {}_sf .

The output is not a band-limited field on the sphere in general; the functions below
return its exact values on a sampling grid.  Such operators are equivariant under polar
rotations for the coordinate section, but not under all rotations in general.
"""

import jax.numpy as jnp
import numpy as np

from s2conv import kernels as _kernels
from s2conv import transforms
from s2conv.convolutions.lifted import spin_to_spin
from s2conv.utils.indexing import conjugate, m_axis, n_axis
from s2conv.utils.rotation import delta_matrices


def section_components(flm, kernel, s_in: int, N: int) -> jnp.ndarray:
    r"""
    Spin components :math:`(2\pi)^{-1/2}\mathcal{C}_{s\to s'}[\Psi]\, {}_sf`, :math:`|s'| < N`, of a section evaluation.

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        kernel: The kernel, ``[2N_k - 1, L, 2L - 1]``.
        s_in (int): Input spin :math:`s`.
        N (int): Bound on the output spins, :math:`|s'| < N`.

    Returns:
        jnp.ndarray: Harmonic coefficients ``[..., 2N - 1, L, 2L - 1]``, indexed
        ``[..., N - 1 + s', el, L - 1 + m]``.

    """
    return jnp.stack(
        [spin_to_spin(flm, kernel, s_in, int(sp)) for sp in n_axis(N)], axis=-3
    ) / np.sqrt(2 * np.pi)


def _chi_on_grid(chi, L: int, sampling: str):
    if chi is None or np.isscalar(chi) or not callable(chi):
        return chi
    thetas, phis = transforms.sphere_angles(L, sampling)
    theta, phi = np.meshgrid(thetas, phis, indexing="ij")
    return chi(theta, phi)


def section_evaluation(
    flm,
    kernel,
    s_in: int,
    N: int,
    chi=None,
    sampling: str = "mw",
    method: str = "jax",
) -> jnp.ndarray:
    r"""
    Evaluation of :math:`\Psi \star \Lambda_s f` on a section, :math:`E_\sigma[\Psi]\, {}_sf` (Proposition VII.1).

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[L, 2L - 1]``.
        kernel: The kernel, ``[2N_k - 1, L, 2L - 1]``.
        s_in (int): Input spin :math:`s`.
        N (int): Bound on the output spins, :math:`|s'| < N`; the kernel's column
            :math:`k = -s` must vanish for :math:`|n| \ge N` for the evaluation to be exact.
        chi (optional): The section twist :math:`\chi_\sigma`: None for the coordinate
            section, a constant, an array on the sampling grid, or a function
            ``chi(theta, phi)`` evaluated on an equiangular grid.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        method (str, optional): Transform backend. Defaults to "jax".

    Returns:
        jnp.ndarray: Samples of the (scalar, coordinate-frame) output on the sampling grid.

    """
    L = flm.shape[-2]
    chi = _chi_on_grid(chi, L, sampling)
    components = section_components(flm, kernel, s_in, N)
    total = 0.0
    for sp in n_axis(N):
        values = transforms.inverse(
            components[N - 1 + sp], L, int(sp), sampling, method
        )
        total = total + (values if chi is None else jnp.exp(-1j * sp * chi) * values)
    return total


def standard_correlation(
    flm, psi_lm, N: int, sampling: str = "mw", method: str = "jax"
):
    r"""
    Standard correlation :math:`(f \circledast \psi)(R_\omega) = \langle f, \mathcal{R}_{(\varphi, \theta, 0)}\psi\rangle` (Definition III.7).

    The directional convolution evaluated on the coordinate section,
    :math:`E_{\sigma_0}[\Lambda_0\psi]` (Proposition VII.1).  It is equivariant under polar rotations,
    but not under all rotations unless the kernel is axisymmetric.

    Args:
        flm: Scalar harmonic coefficients, ``[L, 2L - 1]``.
        psi_lm: Scalar harmonic coefficients of the directional kernel, ``[L, 2L - 1]``.
        N (int): Azimuthal band-limit of the kernel, :math:`\psi_{\ell m} = 0` for
            :math:`|m| \ge N`; the output spins are :math:`|s'| < N`.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        method (str, optional): Transform backend. Defaults to "jax".

    Returns:
        jnp.ndarray: Samples of the output on the sampling grid.

    """
    return section_evaluation(
        flm, _kernels.lifted(psi_lm, 0), 0, N, None, sampling, method
    )


def commutative_anisotropic_convolution(
    flm, h_lm, N: int, sampling: str = "mw", method: str = "jax"
):
    r"""
    Commutative anisotropic convolution of Sadeghi et al., :math:`(h \odot_{\mathrm{S}} f)(\theta, \varphi) = \int (\mathcal{R}_{(\varphi,\theta,\pi-\varphi)} h)\, f\, \mathrm{d}\mu` (Definition III.8).

    A bilinear pairing evaluated on the section :math:`\chi_\sigma = \pi - \varphi`, i.e.
    :math:`E_\sigma[\Lambda_0 h^*]\, f` (Proposition VII.1).  It is commutative, but not
    equivariant under all rotations in general; a polar rotation :math:`R_z(a)` is a symmetry
    exactly when :math:`e^{\mathrm{i}ta} = 1` for every active output spin :math:`t`.

    Args:
        flm: Scalar harmonic coefficients, ``[L, 2L - 1]``.
        h_lm: Scalar harmonic coefficients of the kernel, ``[L, 2L - 1]``.
        N (int): Azimuthal band-limit of the kernel, :math:`h_{\ell m} = 0` for :math:`|m| \ge N`.
        sampling (str, optional): Equiangular sampling scheme. Defaults to "mw".
        method (str, optional): Transform backend. Defaults to "jax".

    Returns:
        jnp.ndarray: Samples of the output on the sampling grid.

    """
    kernel = _kernels.lifted(conjugate(h_lm, 0), 0)

    def chi(theta, phi):
        return np.pi - phi + 0.0 * theta

    return section_evaluation(flm, kernel, 0, N, chi, sampling, method)


def standard_correlation_fourier(flm, psi_lm) -> jnp.ndarray:
    r"""
    Standard correlation as a finite Fourier series, the :math:`\mathcal{O}(L^3)` algorithm of App. B 8.

    With :math:`\Delta^\ell = d^\ell(\pi/2)`, :math:`G^{\ell}_k = \sum_n \mathrm{i}^{-n}
    \Delta^{\ell}_{kn} \psi^*_{\ell n}` and :math:`H_{mk} = \sum_\ell \mathrm{i}^m \Delta^{\ell}_{km}
    f_{\ell m} G^{\ell}_k` (Eq. B26), the standard correlation is
    :math:`\sum_{m,k} H_{mk}\, e^{\mathrm{i}m\varphi - \mathrm{i}k\theta}` (Eq. B27), which
    :func:`s2conv.utils.evaluation.evaluate_fourier` evaluates at any points.

    Args:
        flm: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.
        psi_lm: Scalar harmonic coefficients of the kernel, ``[L, 2L - 1]``.

    Returns:
        jnp.ndarray: :math:`H_{mk}`, ``[..., 2L - 1, 2L - 1]``.

    """
    L = flm.shape[-2]
    delta = delta_matrices(L)
    ipow = (1j) ** m_axis(L)
    g = jnp.einsum("n,lkn,ln->lk", np.conj(ipow), delta, jnp.conj(psi_lm))
    return jnp.einsum("m,lkm,...lm,lk->...mk", ipow, delta, flm, g)
