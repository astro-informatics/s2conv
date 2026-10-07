r"""
The rotation-equivariant spherical convolutions of the literature (Secs. III and VI).

Each is a generalised lifted convolution with a particular kernel (Table II), and is
written here in its closed harmonic form, with the constants of the article:

.. list-table::
   :header-rows: 1

   * - Convolution
     - Lifted form
     - Harmonic form
   * - axisymmetric (Defs III.1, III.6)
     - :math:`(2\pi)^{-1/2}\mathcal{C}_{s\to0}[\Lambda_s\psi]`
     - :math:`\sqrt{4\pi/(2\ell+1)}\, {}_sf_{\ell m}\, {}_s\psi^*_{\ell0}`
   * - left (Def. III.2)
     - :math:`\mathcal{C}_{0\to0}[\Psi_h]`
     - :math:`2\pi\sqrt{4\pi/(2\ell+1)}\, f_{\ell m} h_{\ell0}`
   * - directional (Defs III.3, III.5)
     - :math:`\mathcal{C}_{s\to G}[\Lambda_s\psi]`
     - :math:`\frac{8\pi^2}{2\ell+1}\, {}_sf_{\ell m}\, {}_s\psi^*_{\ell n}`
   * - synthesis (Def. III.10)
     - :math:`\mathcal{C}_{G\to s}[\widetilde{\Lambda_s\psi}]`
     - :math:`\sum_n W^{\ell}_{mn}\, {}_s\psi_{\ell n}`
   * - spin-changing (Prop. VI.5)
     - :math:`\mathcal{C}_{s\to s'}[\Psi_T]`
     - :math:`c_\ell\, {}_sf_{\ell m}`

The group convolution (Definition III.4) is :func:`s2conv.convolutions.lifted.group_convolution`.
All functions act on harmonic coefficients in the s2fft layout, broadcast over leading
axes, and are differentiable and JIT compilable.
"""

import jax.numpy as jnp
import numpy as np

from s2conv.convolutions.lifted import (
    group_to_spin_weights,
    spin_multiplier,
    spin_to_group_weights,
)
from s2conv.utils.indexing import azimuthal_bandlimit, lm_mask


def _axisymmetric_weight(L: int) -> np.ndarray:
    return np.sqrt(4 * np.pi / (2 * np.arange(L) + 1))


def axisymmetric_convolution(flm, psi_lm, spin: int = 0) -> jnp.ndarray:
    r"""
    Axisymmetric convolution :math:`(f \odot \psi)(\omega) = \langle f, \mathcal{R}_\omega \psi \rangle`.

    For a scalar field (Definition III.1) or a spin-:math:`s` field (Definition III.6) with an
    axisymmetric kernel, the output is a scalar field with
    :math:`(f \odot \psi)_{\ell m} = \sqrt{4\pi/(2\ell+1)}\, {}_sf_{\ell m}\, {}_s\psi^*_{\ell 0}`
    (Eqs. 14, 25).  It is :math:`(2\pi)^{-1/2}\mathcal{C}_{s\to0}[\Lambda_s\psi]`, the factor being the
    difference between evaluation on the section and the adjoint projection
    (Proposition VI.3).  Only the axisymmetric part :math:`\psi_{\ell0}` of the kernel enters.

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        psi_lm: Spin-:math:`s` harmonic coefficients of the kernel, ``[L, 2L - 1]``.
        spin (int, optional): Spin :math:`s`. Defaults to 0.

    Returns:
        jnp.ndarray: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    L = flm.shape[-2]
    c = _axisymmetric_weight(L) * jnp.conj(psi_lm[..., :, L - 1])
    return spin_multiplier(flm, c, spin, 0)


def left_convolution(flm, h_lm) -> jnp.ndarray:
    r"""
    Left convolution of Driscoll & Healy, :math:`(f \star_{\mathrm{L}} h)(\omega) = \int f(\rho\eta)\, h(R_\rho^{-1}\omega)\, \mathrm{d}\mu(\rho)`.

    Its harmonic form is :math:`2\pi\sqrt{4\pi/(2\ell+1)}\, f_{\ell m} h_{\ell 0}` (Definition III.2,
    Eq. 16): only the zonal part of the kernel enters, and the :math:`2\pi` is the volume of
    the fibre.  It is :math:`\mathcal{C}_{0\to0}[\Psi_h]` with :math:`\Psi_h(g) = h^*(g^{-1}\eta)`,
    exactly (Proposition VI.4).

    Args:
        flm: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.
        h_lm: Scalar harmonic coefficients of the kernel, ``[L, 2L - 1]``.

    Returns:
        jnp.ndarray: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    L = flm.shape[-2]
    c = 2 * np.pi * _axisymmetric_weight(L) * h_lm[..., :, L - 1]
    return spin_multiplier(flm, c, 0, 0)


def directional_convolution(flm, psi_lm, spin: int = 0, N: int = None) -> jnp.ndarray:
    r"""
    Directional convolution :math:`(f \circledast \psi)(\rho) = \langle f, \mathcal{R}_\rho\psi\rangle`, a function on the rotation group.

    For scalar (Definition III.3) or spin-:math:`s` fields (Definition III.5) its Wigner
    coefficients are :math:`\frac{8\pi^2}{2\ell+1}\, {}_sf_{\ell m}\, {}_s\psi^*_{\ell n}`
    (Eqs. 18, 23).  It is :math:`\mathcal{C}_{s\to G}[\Lambda_s\psi]`, exactly, with no residual
    constant (Proposition VI.1).  For a steerable kernel, azimuthally band-limited at
    :math:`N`, the output is the tuple of :math:`2N - 1` spin-changing convolutions of
    Proposition VI.6 (see :func:`s2conv.lifting.spin_components`).

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        psi_lm: Spin-:math:`s` harmonic coefficients of the kernel, ``[L, 2L - 1]``.
        spin (int, optional): Spin :math:`s`. Defaults to 0.
        N (int, optional): Azimuthal band-limit of the output, retaining the kernel's
            orders :math:`|n| < N`.  Defaults to :math:`L`, which is exact for any kernel.

    Returns:
        jnp.ndarray: Wigner coefficients, ``[..., 2N - 1, L, 2L - 1]``.

    """
    L = flm.shape[-2]
    N = L if N is None else N
    weight = 8 * np.pi**2 / (2 * np.arange(L) + 1)
    u = (
        jnp.conj(
            psi_lm[..., :, L - N : L + N - 1] * lm_mask(L, spin)[:, L - N : L + N - 1]
        )
        * weight[:, None]
    )
    return spin_to_group_weights(flm, jnp.swapaxes(u, -1, -2), spin)


def directional_synthesis(flmn, psi_lm, spin: int = 0) -> jnp.ndarray:
    r"""
    Haar synthesis :math:`(W \circledast^\dagger \psi)(\omega) = \int W(\rho)\, (\mathcal{R}_\rho\psi)(\omega)\, \mathrm{d}\mu(\rho)`.

    Its harmonic form is :math:`\sum_n W^{\ell}_{mn}\, {}_s\psi_{\ell n}` (Definition III.10,
    Eq. 31).  It is :math:`\mathcal{C}_{G\to s}[\widetilde{\Lambda_s\psi}]`, the adjoint of the
    directional convolution with the same kernel (Proposition VI.2).

    Args:
        flmn: Wigner coefficients, ``[..., 2N - 1, L, 2L - 1]``.
        psi_lm: Spin-:math:`s` harmonic coefficients of the kernel, ``[L, 2L - 1]``.
        spin (int, optional): Spin :math:`s` of the output. Defaults to 0.

    Returns:
        jnp.ndarray: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    N, L = azimuthal_bandlimit(flmn), flmn.shape[-2]
    v = (psi_lm * lm_mask(L, spin))[..., :, L - N : L + N - 1]
    return group_to_spin_weights(flmn, jnp.swapaxes(v, -1, -2), spin)


def axisymmetric_synthesis(flm, psi_lm, spin: int = 0) -> jnp.ndarray:
    r"""
    Axisymmetric synthesis :math:`\int W(\omega')\, (\mathcal{R}_{\omega'}\psi)(\omega)\, \mathrm{d}\mu(\omega')`.

    The adjoint of :func:`axisymmetric_convolution`, mapping a scalar field to a spin-:math:`s`
    field, :math:`\sqrt{4\pi/(2\ell+1)}\, W_{\ell m}\, {}_s\psi_{\ell 0}`: the first term of the
    spin wavelet synthesis (Eq. 26).

    Args:
        flm: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.
        psi_lm: Spin-:math:`s` harmonic coefficients of the kernel, ``[L, 2L - 1]``.
        spin (int, optional): Spin :math:`s` of the output. Defaults to 0.

    Returns:
        jnp.ndarray: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    L = flm.shape[-2]
    c = _axisymmetric_weight(L) * psi_lm[..., :, L - 1]
    return spin_multiplier(flm, c, 0, spin)


spin_changing_convolution = spin_multiplier
r"""Spin-changing convolution with multiplier :math:`c_\ell` (Proposition VI.5); see :func:`s2conv.convolutions.lifted.spin_multiplier`."""
