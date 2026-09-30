r"""
Wigner functions and rotations, in the conventions of s2fft and of the article.

Rotations are parameterised by :math:`zyz` Euler angles,
:math:`R(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)` (Eq. 1), and the
Wigner functions are :math:`D^{\ell}_{mn}(\alpha, \beta, \gamma) = e^{-\mathrm{i}m\alpha}
d^{\ell}_{mn}(\beta) e^{-\mathrm{i}n\gamma}` (Eq. 6).  The reduced functions are computed
by the factorisation :math:`d^{\ell}_{mn}(\beta) = \mathrm{i}^{m-n} \sum_k
\Delta^{\ell}_{km} \Delta^{\ell}_{kn} e^{-\mathrm{i}k\beta}` with
:math:`\Delta^{\ell} = d^{\ell}(\pi/2)` (Eq. B25), so that every function here is
differentiable in the Euler angles.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from s2fft.recursions import risbo

from s2conv.utils.indexing import m_axis


@functools.cache
def delta_matrices(L: int) -> jnp.ndarray:
    r"""
    Wigner functions at :math:`\beta = \pi/2`, computed once by the Risbo recursion.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: :math:`\Delta^{\ell}_{km} = d^{\ell}_{km}(\pi/2)` with shape
        ``[L, 2L - 1, 2L - 1]``, indexed ``[el, L - 1 + k, L - 1 + m]``.

    """
    delta = np.zeros((L, 2 * L - 1, 2 * L - 1))
    plane = np.zeros((2 * L - 1, 2 * L - 1))
    for el in range(L):
        plane = risbo.compute_full(plane, np.pi / 2, L, el)
        delta[el] = plane
    # Create a concrete array even when first called while tracing, so that the cache
    # never holds a tracer.
    with jax.ensure_compile_time_eval():
        return jnp.asarray(delta)


def wigner_d(beta, L: int) -> jnp.ndarray:
    r"""
    Reduced Wigner functions :math:`d^{\ell}_{mn}(\beta)` for all :math:`\ell < L`.

    Args:
        beta: Second Euler angle (a scalar).
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Array of shape ``[L, 2L - 1, 2L - 1]`` indexed
        ``[el, L - 1 + m, L - 1 + n]``.

    """
    delta = delta_matrices(L)
    k = m_axis(L)
    phase = (1j) ** (k[:, None] - k[None, :])
    d = jnp.einsum("lkm,k,lkn->lmn", delta, jnp.exp(-1j * k * beta), delta)
    return (d * phase).real


def wigner_D(alpha, beta, gamma, L: int) -> jnp.ndarray:  # noqa: N802
    r"""
    Wigner functions :math:`D^{\ell}_{mn}(\alpha, \beta, \gamma)` (Eq. 6).

    Args:
        alpha: First Euler angle.
        beta: Second Euler angle.
        gamma: Third Euler angle.
        L (int): Harmonic band-limit.

    Returns:
        jnp.ndarray: Array of shape ``[L, 2L - 1, 2L - 1]`` indexed
        ``[el, L - 1 + m, L - 1 + n]``.

    """
    m = m_axis(L)
    return (
        jnp.exp(-1j * m * alpha)[None, :, None]
        * wigner_d(beta, L)
        * jnp.exp(-1j * m * gamma)[None, None, :]
    )


def rotate_flm(flm, rotation) -> jnp.ndarray:
    r"""
    Rotate the harmonic coefficients of a field of any spin (Eq. 10).

    Computes :math:`(\mathcal{R}_\rho f)_{\ell m} = \sum_n D^{\ell}_{mn}(\rho)\, f_{\ell n}`,
    which holds with the same Wigner matrices for every spin, in
    :math:`\mathcal{O}(L^3)` operations through the factorisation of rotations.
    Leading axes are broadcast, so Wigner coefficients ``flmn[..., 2N - 1, L, 2L - 1]``
    may be passed to obtain the left translation :math:`F(\rho^{-1}\,\cdot\,)`, which acts
    on the first Wigner index.

    Args:
        flm: Harmonic coefficients with shape ``[..., L, 2L - 1]``.
        rotation: Euler angles :math:`(\alpha, \beta, \gamma)` of the rotation.

    Returns:
        jnp.ndarray: Coefficients of the rotated field, with the shape of ``flm``.

    """
    alpha, beta, gamma = rotation
    L = flm.shape[-2]
    delta = delta_matrices(L)
    m = m_axis(L)
    x = flm * ((1j) ** (-m) * jnp.exp(-1j * m * gamma))
    g = jnp.einsum("lkn,...ln->...lk", delta, x) * jnp.exp(-1j * m * beta)
    out = jnp.einsum("lkm,...lk->...lm", delta, g)
    return out * ((1j) ** m * jnp.exp(-1j * m * alpha))


rotate_flmn = rotate_flm


def rotation_matrix(alpha, beta, gamma) -> jnp.ndarray:
    r"""
    Rotation matrices :math:`R_z(\alpha) R_y(\beta) R_z(\gamma)` (Eq. 1).

    Args:
        alpha: First Euler angle(s).
        beta: Second Euler angle(s).
        gamma: Third Euler angle(s).

    Returns:
        jnp.ndarray: Matrices of shape ``[..., 3, 3]``, broadcast over the angles.

    """
    alpha, beta, gamma = jnp.broadcast_arrays(
        jnp.asarray(alpha, float), jnp.asarray(beta, float), jnp.asarray(gamma, float)
    )
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    cg, sg = jnp.cos(gamma), jnp.sin(gamma)
    rows = [
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ]
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)


def euler_angles(R) -> tuple:
    r"""
    :math:`zyz` Euler angles of rotation matrices.

    Args:
        R: Rotation matrices with shape ``[..., 3, 3]``.

    Returns:
        tuple: The angles :math:`(\alpha, \beta, \gamma)` with
        :math:`R = R_z(\alpha) R_y(\beta) R_z(\gamma)`, :math:`\beta \in [0, \pi]`.
        The first and third angles are ill-defined at :math:`\beta \in \{0, \pi\}`.

    """
    alpha = jnp.arctan2(R[..., 1, 2], R[..., 0, 2])
    beta = jnp.arccos(jnp.clip(R[..., 2, 2], -1.0, 1.0))
    gamma = jnp.arctan2(R[..., 2, 1], -R[..., 2, 0])
    return alpha, beta, gamma


def section(theta, phi) -> jnp.ndarray:
    r"""
    The coordinate section :math:`s(\theta, \varphi) = R(\varphi, \theta, 0)` (Eq. 33).

    Args:
        theta: Colatitudes.
        phi: Longitudes.

    Returns:
        jnp.ndarray: Rotation matrices of shape ``[..., 3, 3]``.

    """
    return rotation_matrix(phi, theta, 0.0)


def random_rotation(key) -> tuple:
    r"""
    Euler angles of a rotation drawn from the Haar measure on the rotation group.

    Args:
        key: A JAX random key.

    Returns:
        tuple: Euler angles :math:`(\alpha, \beta, \gamma)`.

    """
    ka, kb, kg = jax.random.split(key, 3)
    alpha = jax.random.uniform(ka, minval=0.0, maxval=2 * jnp.pi)
    beta = jnp.arccos(jax.random.uniform(kb, minval=-1.0, maxval=1.0))
    gamma = jax.random.uniform(kg, minval=0.0, maxval=2 * jnp.pi)
    return alpha, beta, gamma
