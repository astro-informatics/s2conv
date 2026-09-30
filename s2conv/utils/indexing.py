r"""
Index conventions shared by s2conv, following s2fft.

Harmonic coefficients of a spin field are stored as ``flm[..., el, L - 1 + m]`` with
shape ``[..., L, 2L - 1]``.  Wigner coefficients of a function on the rotation group
are stored in the s2fft layout ``flmn[..., N - 1 + n, el, L - 1 + m]``, the
coefficient :math:`F^{\ell}_{mn}`, with shape ``[..., 2N - 1, L, 2L - 1]`` for
azimuthal band-limit :math:`N`.
"""

import jax.numpy as jnp
import numpy as np


def m_axis(L: int) -> np.ndarray:
    r"""
    Azimuthal orders along the last axis of a coefficient array.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        np.ndarray: The orders :math:`-(L-1), \dots, L-1`.

    """
    return np.arange(-(L - 1), L)


def n_axis(N: int) -> np.ndarray:
    r"""
    Orders along the first axis of Wigner coefficients.

    Args:
        N (int): Azimuthal band-limit.

    Returns:
        np.ndarray: The orders :math:`-(N-1), \dots, N-1`.

    """
    return np.arange(-(N - 1), N)


def parity(s) -> int:
    """
    The sign :math:`(-1)^s` of an integer, valid for negative and NumPy integers.

    Args:
        s: An integer.

    Returns:
        int: :math:`(-1)^s`.

    """
    return 1 - 2 * (abs(int(s)) % 2)


def lm_mask(L: int, spin: int = 0) -> np.ndarray:
    r"""
    Entries carried by a band-limited spin field.

    Args:
        L (int): Harmonic band-limit.
        spin (int, optional): Spin. Defaults to 0.

    Returns:
        np.ndarray: Boolean mask of shape ``[L, 2L - 1]``, true where
        :math:`\ell \ge |s|` and :math:`|m| \le \ell`.

    """
    el = np.arange(L)[:, None]
    m = m_axis(L)[None, :]
    return (el >= abs(spin)) & (np.abs(m) <= el)


def wigner_mask(L: int, N: int) -> np.ndarray:
    r"""
    Entries carried by a function on the rotation group.

    Args:
        L (int): Harmonic band-limit.
        N (int): Azimuthal band-limit.

    Returns:
        np.ndarray: Boolean mask of shape ``[2N - 1, L, 2L - 1]``, true where
        :math:`|n| \le \ell` and :math:`|m| \le \ell`.

    """
    n = n_axis(N)[:, None, None]
    el = np.arange(L)[None, :, None]
    m = m_axis(L)[None, None, :]
    return (np.abs(n) <= el) & (np.abs(m) <= el)


def conjugate(flm, spin: int = 0):
    r"""
    Harmonic coefficients of the complex conjugate of a spin-:math:`s` field (Eq. A9).

    The conjugate is a spin :math:`-s` field with coefficients
    :math:`(\,{}_sf^*)_{\ell m} = (-1)^{s+m}\, {}_sf^*_{\ell,-m}`.

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        spin (int, optional): Spin :math:`s`. Defaults to 0.

    Returns:
        Spin :math:`-s` harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    m = m_axis(flm.shape[-2])
    return ((-1.0) ** (spin + m)) * jnp.conj(flm[..., ::-1])


def bandlimit(flm) -> int:
    """
    Harmonic band-limit of spherical harmonic coefficients ``flm[..., L, 2L - 1]``.

    Args:
        flm: Harmonic coefficients.

    Returns:
        int: The band-limit :math:`L`.

    """
    return flm.shape[-2]


def azimuthal_bandlimit(flmn) -> int:
    """
    Azimuthal band-limit of Wigner coefficients ``flmn[..., 2N - 1, L, 2L - 1]``.

    Args:
        flmn: Wigner coefficients.

    Returns:
        int: The azimuthal band-limit :math:`N`.

    """
    return (flmn.shape[-3] + 1) // 2


def check_spin(L: int, spin: int, N: int = None) -> None:
    r"""
    Check that a spin is representable at the given band-limits.

    Args:
        L (int): Harmonic band-limit.
        spin (int): Spin.
        N (int, optional): Azimuthal band-limit, if the spin indexes a Wigner column.

    Raises:
        ValueError: If :math:`|s| \ge L`, or :math:`|s| \ge N` when N is given.

    """
    if abs(spin) >= L:
        raise ValueError(f"Spin {spin} is not representable at band-limit L = {L}.")
    if N is not None and abs(spin) >= N:
        raise ValueError(
            f"Spin {spin} is not representable at azimuthal band-limit N = {N}; "
            f"require N > |spin|."
        )
