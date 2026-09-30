r"""
Evaluation of band-limited fields at arbitrary points of the sphere and rotation group.

Through the factorisation of Eq. (B25), a spin field band-limited at :math:`L` is a
finite two-dimensional Fourier series,
:math:`{}_sf(\theta, \varphi) = \sum_{m,k} H_{mk}\, e^{\mathrm{i}m\varphi}
e^{-\mathrm{i}k\theta}`, which can be evaluated exactly at any point, including points
that do not lie on a sampling grid.  This is what the real-space checks of rotations and
equivariance require.
"""

import jax.numpy as jnp
import numpy as np

from s2conv.utils.indexing import lm_mask, m_axis, parity
from s2conv.utils.rotation import delta_matrices, wigner_d


def spin_fourier(flm, spin: int = 0) -> jnp.ndarray:
    r"""
    Fourier coefficients of a band-limited spin field on the torus.

    Uses :math:`{}_sY_{\ell m}(\theta, \varphi) = (-1)^s \sqrt{(2\ell+1)/(4\pi)}\,
    d^{\ell}_{m,-s}(\theta)\, e^{\mathrm{i}m\varphi}` (Eq. A7) with the factorisation
    of Eq. (B25).

    Args:
        flm: Spin-:math:`s` harmonic coefficients with shape ``[..., L, 2L - 1]``.
        spin (int, optional): Spin. Defaults to 0.

    Returns:
        jnp.ndarray: Coefficients :math:`H_{mk}` with shape ``[..., 2L - 1, 2L - 1]``,
        indexed ``[..., L - 1 + m, L - 1 + k]``.

    """
    L = flm.shape[-2]
    delta = delta_matrices(L)
    el, m = np.arange(L), m_axis(L)
    weight = parity(spin) * np.sqrt((2 * el + 1) / (4 * np.pi))
    a = flm * lm_mask(L, spin) * weight[:, None]
    h = jnp.einsum("...lm,lkm,lk->...mk", a, delta, delta[:, :, L - 1 - spin])
    return h * ((1j) ** (m + spin))[:, None]


def evaluate_fourier(h, theta, phi) -> jnp.ndarray:
    r"""
    Evaluate :math:`\sum_{m,k} H_{mk}\, e^{\mathrm{i}m\varphi} e^{-\mathrm{i}k\theta}`.

    Args:
        h: Coefficients with shape ``[..., 2L - 1, 2L - 1]``.
        theta: Colatitudes (any shape broadcastable with ``phi``).
        phi: Longitudes.

    Returns:
        jnp.ndarray: Values with shape ``[..., *points]``.

    """
    L = h.shape[-1] // 2 + 1
    m = m_axis(L)
    theta, phi = jnp.broadcast_arrays(
        jnp.asarray(theta, float), jnp.asarray(phi, float)
    )
    shape = theta.shape
    e_phi = jnp.exp(1j * jnp.outer(phi.ravel(), m))
    e_theta = jnp.exp(-1j * jnp.outer(theta.ravel(), m))
    values = jnp.einsum("pm,...mk,pk->...p", e_phi, h, e_theta)
    return values.reshape(values.shape[:-1] + shape)


def evaluate(flm, spin: int, theta, phi) -> jnp.ndarray:
    """
    Evaluate a band-limited spin field at arbitrary points.

    Args:
        flm: Spin-:math:`s` harmonic coefficients with shape ``[..., L, 2L - 1]``.
        spin (int): Spin.
        theta: Colatitudes.
        phi: Longitudes.

    Returns:
        jnp.ndarray: Field values with shape ``[..., *points]``.

    """
    return evaluate_fourier(spin_fourier(flm, spin), theta, phi)


def evaluate_so3(flmn, alpha, beta, gamma) -> jnp.ndarray:
    r"""
    Evaluate a band-limited function on the rotation group at one rotation.

    Computes :math:`F(\rho) = \sum_\ell \frac{2\ell+1}{8\pi^2} \sum_{m,n}
    F^{\ell}_{mn} D^{\ell *}_{mn}(\rho)` (Eq. 7).

    Args:
        flmn: Wigner coefficients with shape ``[..., 2N - 1, L, 2L - 1]``.
        alpha: First Euler angle.
        beta: Second Euler angle.
        gamma: Third Euler angle.

    Returns:
        jnp.ndarray: The value(s) :math:`F(\alpha, \beta, \gamma)`, one per leading index.

    """
    N, L = (flmn.shape[-3] + 1) // 2, flmn.shape[-2]
    m, n = m_axis(L), np.arange(-(N - 1), N)
    d = wigner_d(beta, L)[:, :, L - N : L + N - 1]
    d_conj = (
        jnp.exp(1j * m * alpha)[None, :, None]
        * d
        * jnp.exp(1j * n * gamma)[None, None, :]
    )
    weight = (2 * np.arange(L) + 1) / (8 * np.pi**2)
    return jnp.einsum("l,...nlm,lmn->...", weight, flmn, d_conj)
