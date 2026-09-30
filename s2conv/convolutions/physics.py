r"""
Spin-changing operators of mathematical physics (Proposition VI.8).

Each is a spin-changing multiplier, i.e. a generalised lifted convolution
:math:`\mathcal{C}_{s\to s'}[\Psi_T]` with a (generally distributional) canonical kernel:

- the spin raising and lowering operators :math:`\eth` and :math:`\bar\eth` (Eqs. 3, 5), with
  :math:`c_\ell = \sqrt{(\ell-s)(\ell+s+1)}` and :math:`-\sqrt{(\ell+s)(\ell-s+1)}`;
- the :math:`E`/:math:`B` decomposition of a spin :math:`\pm2` polarisation field,
  :math:`a_E = -\tfrac12(a_2 + a_{-2})` and :math:`a_B = \tfrac{\mathrm{i}}{2}(a_2 - a_{-2})`, each a
  sum of a :math:`\mathcal{C}_{2\to0}` and a :math:`\mathcal{C}_{-2\to0}` operator;
- the spherical Kaiser–Squires relation of weak gravitational lensing, the
  :math:`\mathcal{C}_{0\to2}` multiplier :math:`\mathcal{D}_\ell = -\sqrt{(\ell+2)(\ell-1)/(\ell(\ell+1))}`
  from convergence to shear.
"""

import jax.numpy as jnp
import numpy as np

from s2conv.convolutions.lifted import spin_multiplier
from s2conv.utils.indexing import conjugate


def eth_multiplier(L: int, spin: int) -> np.ndarray:
    r"""
    Multiplier of the spin raising operator :math:`\eth` from spin :math:`s` to :math:`s + 1`.

    Args:
        L (int): Harmonic band-limit.
        spin (int): Input spin :math:`s`.

    Returns:
        np.ndarray: :math:`c_\ell = \sqrt{(\ell-s)(\ell+s+1)}`, ``[L]``.

    """
    el = np.arange(L)
    return np.sqrt(np.clip((el - spin) * (el + spin + 1), 0, None))


def ethbar_multiplier(L: int, spin: int) -> np.ndarray:
    r"""
    Multiplier of the spin lowering operator :math:`\bar\eth` from spin :math:`s` to :math:`s - 1`.

    Args:
        L (int): Harmonic band-limit.
        spin (int): Input spin :math:`s`.

    Returns:
        np.ndarray: :math:`c_\ell = -\sqrt{(\ell+s)(\ell-s+1)}`, ``[L]``.

    """
    el = np.arange(L)
    return -np.sqrt(np.clip((el + spin) * (el - spin + 1), 0, None))


def eth(flm, spin: int) -> jnp.ndarray:
    r"""
    The spin raising operator :math:`\eth`, from spin :math:`s` to :math:`s + 1` (Proposition VI.8(a)).

    With the conventions of Eq. (3), :math:`\eth = -\sin^s\theta\, (\partial_\theta +
    \mathrm{i}\sin^{-1}\theta\, \partial_\varphi)\, \sin^{-s}\theta`.

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        spin (int): Input spin :math:`s`.

    Returns:
        jnp.ndarray: Spin-:math:`(s+1)` harmonic coefficients.

    """
    return spin_multiplier(flm, eth_multiplier(flm.shape[-2], spin), spin, spin + 1)


def ethbar(flm, spin: int) -> jnp.ndarray:
    r"""
    The spin lowering operator :math:`\bar\eth`, from spin :math:`s` to :math:`s - 1` (Proposition VI.8(a)).

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        spin (int): Input spin :math:`s`.

    Returns:
        jnp.ndarray: Spin-:math:`(s-1)` harmonic coefficients.

    """
    return spin_multiplier(flm, ethbar_multiplier(flm.shape[-2], spin), spin, spin - 1)


def eb_decomposition(p_lm, p_conj_lm=None) -> tuple:
    r"""
    :math:`E` and :math:`B` components of a polarisation field (Proposition VI.8(b)).

    :math:`a_E = -\tfrac12(a_2 + a_{-2})` and :math:`a_B = \tfrac{\mathrm{i}}{2}(a_2 - a_{-2})`, each the
    sum of a :math:`\mathcal{C}_{2\to0}` and a :math:`\mathcal{C}_{-2\to0}` map.

    Args:
        p_lm: Spin-2 coefficients :math:`a_2` of :math:`Q + \mathrm{i}U`, ``[..., L, 2L - 1]``.
        p_conj_lm (optional): Spin :math:`-2` coefficients :math:`a_{-2}` of :math:`Q - \mathrm{i}U`.
            Defaults to the conjugate of the spin-2 field, i.e. real :math:`Q` and :math:`U`.

    Returns:
        tuple: Scalar harmonic coefficients of :math:`E` and :math:`B`.

    """
    L = p_lm.shape[-2]
    p_conj_lm = conjugate(p_lm, 2) if p_conj_lm is None else p_conj_lm
    half = 0.5 * np.ones(L)
    e = spin_multiplier(p_lm, -half, 2, 0) + spin_multiplier(p_conj_lm, -half, -2, 0)
    b = spin_multiplier(p_lm, 1j * half, 2, 0) + spin_multiplier(
        p_conj_lm, -1j * half, -2, 0
    )
    return e, b


def eb_to_polarisation(e_lm, b_lm) -> tuple:
    r"""
    Spin :math:`\pm2` polarisation from :math:`E` and :math:`B`, the inverse of :func:`eb_decomposition`.

    Args:
        e_lm: Scalar harmonic coefficients of :math:`E`, ``[..., L, 2L - 1]``.
        b_lm: Scalar harmonic coefficients of :math:`B`.

    Returns:
        tuple: Coefficients :math:`a_{\pm2} = -(a_E \pm \mathrm{i}\, a_B)` of the spin :math:`\pm2` fields,
        for :math:`\ell \ge 2`.

    """
    L = e_lm.shape[-2]
    one = np.ones(L)
    p = spin_multiplier(e_lm, -one, 0, 2) + spin_multiplier(b_lm, -1j * one, 0, 2)
    p_conj = spin_multiplier(e_lm, -one, 0, -2) + spin_multiplier(b_lm, 1j * one, 0, -2)
    return p, p_conj


def kaiser_squires_multiplier(L: int) -> np.ndarray:
    r"""
    Multiplier of the spherical Kaiser–Squires relation from convergence to shear.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: :math:`\mathcal{D}_\ell = -\sqrt{(\ell+2)(\ell-1)/(\ell(\ell+1))}` for
        :math:`\ell \ge 2`, and zero below, ``[L]``.

    """
    el = np.arange(L).astype(float)
    d = np.zeros(L)
    d[2:] = -np.sqrt((el[2:] + 2) * (el[2:] - 1) / (el[2:] * (el[2:] + 1)))
    return d


def kaiser_squires(kappa_lm) -> jnp.ndarray:
    r"""
    Shear from convergence, the :math:`\mathcal{C}_{0\to2}` multiplier :math:`\mathcal{D}_\ell` (Proposition VI.8(c)).

    Args:
        kappa_lm: Scalar harmonic coefficients of the convergence, ``[..., L, 2L - 1]``.

    Returns:
        jnp.ndarray: Spin-2 harmonic coefficients of the shear.

    """
    return spin_multiplier(
        kappa_lm, kaiser_squires_multiplier(kappa_lm.shape[-2]), 0, 2
    )


def kaiser_squires_inverse(gamma_lm) -> jnp.ndarray:
    r"""
    Convergence from shear, the :math:`\mathcal{C}_{2\to0}` multiplier :math:`\mathcal{D}_\ell^{-1}` (Proposition VI.8(c)).

    The monopole and dipole of the convergence are not recoverable from the shear and are
    returned as zero.

    Args:
        gamma_lm: Spin-2 harmonic coefficients of the shear, ``[..., L, 2L - 1]``.

    Returns:
        jnp.ndarray: Scalar harmonic coefficients of the convergence.

    """
    d = kaiser_squires_multiplier(gamma_lm.shape[-2])
    inverse = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    return spin_multiplier(gamma_lm, inverse, 2, 0)
