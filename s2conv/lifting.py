r"""
The lift of spin fields to the rotation group and its adjoint projection (Sec. IV).

For spin :math:`s` the isometric lift :math:`\Lambda_s` and its adjoint, the projection
:math:`\Lambda_s^\dagger`, are (Definition IV.6)

.. math::

    (\Lambda_s f)(\alpha, \beta, \gamma) = \frac{1}{\sqrt{2\pi}} e^{-\mathrm{i}s\gamma} f(\beta, \alpha),
    \qquad
    (\Lambda_s^\dagger F)(\beta, \alpha) = \frac{1}{\sqrt{2\pi}} \int_0^{2\pi}
    e^{\mathrm{i}s\gamma} F(\alpha, \beta, \gamma)\, \mathrm{d}\gamma .

In harmonic space the lift fills the single Wigner column :math:`n = -s` and the
projection reads it (Lemma IV.7(c), Eq. 39):

.. math::

    (\Lambda_s f)^{\ell}_{mn} = (-1)^s \sqrt{\tfrac{8\pi^2}{2\ell+1}}\, {}_sf_{\ell m}\, \delta_{n,-s},
    \qquad
    (\Lambda_s^\dagger F)_{\ell m} = (-1)^s \sqrt{\tfrac{2\ell+1}{8\pi^2}}\, F^{\ell}_{m,-s} .

Evaluation on a section, which differs from the adjoint by :math:`\sqrt{2\pi}`, is a
separate operator (:mod:`s2conv.convolutions.section`).
"""

import jax.numpy as jnp
import numpy as np

from s2conv.utils.indexing import (
    azimuthal_bandlimit,
    check_spin,
    lm_mask,
    n_axis,
    parity,
)


def _lift_weight(L: int, spin: int) -> np.ndarray:
    return parity(spin) * np.sqrt(8 * np.pi**2 / (2 * np.arange(L) + 1))


def lift(flm, spin: int, N: int = None) -> jnp.ndarray:
    r"""
    The lift :math:`\Lambda_s` in harmonic space (Lemma IV.7(c)).

    Args:
        flm: Spin-:math:`s` harmonic coefficients with shape ``[..., L, 2L - 1]``.
        spin (int): Spin :math:`s`.
        N (int, optional): Azimuthal band-limit of the output, at least :math:`|s| + 1`.
            Defaults to :math:`|s| + 1`.

    Returns:
        jnp.ndarray: Wigner coefficients with shape ``[..., 2N - 1, L, 2L - 1]``, supported
        on the column :math:`n = -s`.

    """
    L = flm.shape[-2]
    N = abs(spin) + 1 if N is None else N
    check_spin(L, spin, N)
    column = flm * lm_mask(L, spin) * _lift_weight(L, spin)[:, None]
    out = jnp.zeros(flm.shape[:-2] + (2 * N - 1,) + flm.shape[-2:], dtype=column.dtype)
    return out.at[..., N - 1 - spin, :, :].set(column)


def project(flmn, spin: int) -> jnp.ndarray:
    r"""
    The adjoint projection :math:`\Lambda_s^\dagger` in harmonic space (Lemma IV.7(c)).

    Args:
        flmn: Wigner coefficients with shape ``[..., 2N - 1, L, 2L - 1]``.
        spin (int): Spin :math:`s` of the output.

    Returns:
        jnp.ndarray: Spin-:math:`s` harmonic coefficients with shape ``[..., L, 2L - 1]``,
        zero when :math:`|s| \ge N`.

    """
    N, L = azimuthal_bandlimit(flmn), flmn.shape[-2]
    if abs(spin) >= N:
        return jnp.zeros(flmn.shape[:-3] + flmn.shape[-2:], dtype=flmn.dtype)
    weight = parity(spin) * np.sqrt((2 * np.arange(L) + 1) / (8 * np.pi**2))
    return flmn[..., N - 1 - spin, :, :] * lm_mask(L, spin) * weight[:, None]


def spin_components(flmn) -> jnp.ndarray:
    r"""
    Decompose a function on the rotation group into spin fields (Corollary IV.12).

    Every :math:`F` azimuthally band-limited at :math:`N` is
    :math:`F = \sum_{|s| < N} \Lambda_s \Lambda_s^\dagger F`, a tuple of
    :math:`2N - 1` spin fields.

    Args:
        flmn: Wigner coefficients with shape ``[..., 2N - 1, L, 2L - 1]``.

    Returns:
        jnp.ndarray: Spin harmonic coefficients :math:`\Lambda_s^\dagger F` with shape
        ``[..., 2N - 1, L, 2L - 1]``, indexed ``[..., N - 1 + s, el, L - 1 + m]``.

    """
    N = azimuthal_bandlimit(flmn)
    return jnp.stack([project(flmn, s) for s in n_axis(N)], axis=-3)


def from_spin_components(components) -> jnp.ndarray:
    r"""
    Assemble a function on the rotation group from its spin fields (Corollary IV.12).

    Args:
        components: Spin harmonic coefficients with shape ``[..., 2N - 1, L, 2L - 1]``,
            indexed ``[..., N - 1 + s, el, L - 1 + m]``.

    Returns:
        jnp.ndarray: Wigner coefficients :math:`\sum_{|s|<N} \Lambda_s f_s` with shape
        ``[..., 2N - 1, L, 2L - 1]``.

    """
    N = azimuthal_bandlimit(components)
    return sum(lift(components[..., N - 1 + s, :, :], s, N) for s in n_axis(N))


def gamma_samples(N: int) -> np.ndarray:
    r"""
    Samples of the third Euler angle used by s2fft, :math:`\gamma_j = 2\pi j/(2N-1)`.

    Args:
        N (int): Azimuthal band-limit.

    Returns:
        np.ndarray: The :math:`2N - 1` angles.

    """
    return 2 * np.pi * np.arange(2 * N - 1) / (2 * N - 1)


def lift_samples(f, spin: int, N: int = None) -> jnp.ndarray:
    r"""
    The lift :math:`\Lambda_s` on samples (Eq. 38).

    Args:
        f: Samples of a spin-:math:`s` field on the sphere, of any sampling scheme.
        spin (int): Spin :math:`s`.
        N (int, optional): Azimuthal band-limit of the output samples. Defaults to
            :math:`|s| + 1`.

    Returns:
        jnp.ndarray: Samples on the rotation group with shape ``[2N - 1, *f.shape]``.

    """
    N = abs(spin) + 1 if N is None else N
    phase = np.exp(-1j * spin * gamma_samples(N)).reshape((-1,) + (1,) * jnp.ndim(f))
    return phase * jnp.asarray(f)[None] / np.sqrt(2 * np.pi)


def project_samples(F, spin: int) -> jnp.ndarray:
    r"""
    The adjoint projection :math:`\Lambda_s^\dagger` on samples (Eq. 38).

    The integral over the fibre is evaluated by the trapezoidal rule, which is exact for
    samples azimuthally band-limited at :math:`N > |s|`.

    Args:
        F: Samples on the rotation group with shape ``[2N - 1, ...]``.
        spin (int): Spin :math:`s` of the output.

    Returns:
        jnp.ndarray: Samples of the spin-:math:`s` field on the sphere.

    """
    N = (F.shape[0] + 1) // 2
    weights = (
        np.exp(1j * spin * gamma_samples(N))
        * (2 * np.pi / (2 * N - 1))
        / np.sqrt(2 * np.pi)
    )
    return jnp.tensordot(weights, F, axes=(0, 0))
