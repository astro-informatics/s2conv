r"""
Kernels of the generalised lifted convolution and their compact weights.

A kernel :math:`\Psi` is a function on the rotation group, stored by its Wigner
coefficients in the s2fft layout, ``kernel[..., N - 1 + k, el, L - 1 + n]`` holding
:math:`\Psi^{\ell}_{nk}`, with shape ``[..., 2N - 1, L, 2L - 1]``.  Its azimuthal band-limit
:math:`N` bounds the second index :math:`k`, the one along which a lifted spin-:math:`s`
input is supported (:math:`k = -s`).

Each operator reads only part of the kernel block (Theorem V.2, Fig. 5): the entry
:math:`(-s', -s)` for :math:`\mathcal{C}_{s\to s'}`, the column :math:`k = -s` for
:math:`\mathcal{C}_{s\to G}`, the row :math:`n = -s'` for :math:`\mathcal{C}_{G\to s'}` and the
whole block for :math:`\mathcal{C}_{G\to G}`.  The corresponding compact weights of
Theorem V.5 are, with :math:`a_\ell = (2\ell+1)/(8\pi^2)`,

.. math::

    c_\ell = (-1)^{s+s'} \Psi^{\ell *}_{-s',-s}, \quad
    u^{\ell}_n = (-1)^s a_\ell^{-1/2} \Psi^{\ell *}_{n,-s}, \quad
    v^{\ell}_k = (-1)^{s'} a_\ell^{1/2} \Psi^{\ell *}_{-s',k}, \quad
    B^{\ell} = \Psi^{\ell\dagger},

stored as ``c[..., el]``, ``u[..., N - 1 + n, el]``, ``v[..., N - 1 + k, el]`` and
``B[..., el, N_in - 1 + k, N_out - 1 + n]``.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from s2conv import lifting
from s2conv.utils.indexing import (
    azimuthal_bandlimit,
    check_spin,
    lm_mask,
    parity,
    wigner_mask,
)


def _a(L: int) -> np.ndarray:
    return (2 * np.arange(L) + 1) / (8 * np.pi**2)


def _second_index(kernel, N: int) -> jnp.ndarray:
    """The kernel with its second index restricted or zero-padded to |k| < N."""
    N_k = azimuthal_bandlimit(kernel)
    if N <= N_k:
        return kernel[..., N_k - N : N_k + N - 1, :, :]
    pad = [(0, 0)] * (kernel.ndim - 3) + [(N - N_k, N - N_k), (0, 0), (0, 0)]
    return jnp.pad(kernel, pad)


# ----------------------------------------------------------------------------- constructors


def lifted(psi_lm, spin: int, N: int = None) -> jnp.ndarray:
    r"""
    The lifted spherical kernel :math:`\Lambda_s \psi` (Lemma IV.7(c)).

    A spin-:math:`s` kernel on the sphere lifted like a signal occupies the column
    :math:`k = -s`, :math:`(\Lambda_s\psi)^{\ell}_{n,-s} = (-1)^s \sqrt{8\pi^2/(2\ell+1)}\,
    {}_s\psi_{\ell n}`.  With :math:`\mathcal{C}_{s\to G}` it gives the directional
    convolutions (Proposition VI.1) and, for axisymmetric :math:`\psi`, the axisymmetric
    convolutions (Proposition VI.3).

    Args:
        psi_lm: Spin-:math:`s` harmonic coefficients of the kernel, ``[..., L, 2L - 1]``.
        spin (int): Spin :math:`s`.
        N (int, optional): Azimuthal band-limit of the kernel. Defaults to :math:`|s| + 1`.

    Returns:
        jnp.ndarray: The kernel, ``[..., 2N - 1, L, 2L - 1]``.

    """
    return lifting.lift(psi_lm, spin, N)


def flipped(kernel, N: int = None) -> jnp.ndarray:
    r"""
    The flipped kernel :math:`\widetilde\Psi(g) = \Psi^*(g^{-1})`, with blocks :math:`\Psi^{\ell\dagger}`.

    It appears in the adjoints of Theorem V.2; for instance Haar synthesis is
    :math:`\mathcal{C}_{G\to s}[\widetilde{\Lambda_s\psi}]` (Proposition VI.2).

    Args:
        kernel: The kernel :math:`\Psi`, ``[..., 2N_k - 1, L, 2L - 1]``.
        N (int, optional): Azimuthal band-limit of the flipped kernel, which holds the
            first index of :math:`\Psi`.  Defaults to :math:`L`, which is exact.

    Returns:
        jnp.ndarray: The kernel :math:`\widetilde\Psi`, ``[..., 2N - 1, L, 2L - 1]``.

    """
    N_k, L = azimuthal_bandlimit(kernel), kernel.shape[-2]
    N = L if N is None else N
    sub = jnp.swapaxes(kernel[..., :, :, L - N : L + N - 1], -3, -1)
    out = jnp.zeros(kernel.shape[:-3] + (2 * N - 1, L, 2 * L - 1), dtype=kernel.dtype)
    return jnp.conj(out.at[..., :, :, L - N_k : L + N_k - 1].set(sub))


def left(h_lm, N: int = None) -> jnp.ndarray:
    r"""
    The kernel :math:`\Psi_h(g) = h^*(g^{-1}\eta)` of the left convolution (Proposition VI.4).

    It is invariant under left multiplication by the stabiliser, so its block has the
    single row :math:`n = 0`, :math:`(\Psi_h)^{\ell}_{0k} = 2\pi \sqrt{4\pi/(2\ell+1)}\, h^*_{\ell k}`.

    Args:
        h_lm: Harmonic coefficients of the scalar kernel :math:`h`, ``[..., L, 2L - 1]``.
        N (int, optional): Azimuthal band-limit of the kernel.  Defaults to :math:`L`,
            which is exact; :math:`\mathcal{C}_{0\to0}` reads only :math:`k = 0`.

    Returns:
        jnp.ndarray: The kernel, ``[..., 2N - 1, L, 2L - 1]``.

    """
    L = h_lm.shape[-2]
    N = L if N is None else N
    weight = 2 * np.pi * np.sqrt(4 * np.pi / (2 * np.arange(L) + 1))
    row = jnp.conj(h_lm * lm_mask(L))[..., :, L - N : L + N - 1] * weight[:, None]
    out = jnp.zeros(h_lm.shape[:-2] + (2 * N - 1, L, 2 * L - 1), dtype=row.dtype)
    return out.at[..., :, :, L - 1].set(jnp.swapaxes(row, -1, -2))


def multiplier(c, s_in: int, s_out: int, N: int = None) -> jnp.ndarray:
    r"""
    The canonical kernel of a spin-changing multiplier (Theorem V.5(a)).

    Every bounded rotation-equivariant map from spin :math:`s` to spin :math:`s'` multiplies
    by :math:`c_\ell` and is :math:`\mathcal{C}_{s\to s'}[\Psi_T]` with the single entry
    :math:`(\Psi_T)^{\ell}_{-s',-s} = (-1)^{s+s'} c^*_\ell` (Proposition VI.5).

    Args:
        c: Multiplier :math:`c_\ell`, ``[..., L]``.
        s_in (int): Input spin :math:`s`.
        s_out (int): Output spin :math:`s'`.
        N (int, optional): Azimuthal band-limit of the kernel. Defaults to :math:`|s| + 1`.

    Returns:
        jnp.ndarray: The kernel, ``[..., 2N - 1, L, 2L - 1]``.

    """
    c = jnp.asarray(c)
    L = c.shape[-1]
    N = abs(s_in) + 1 if N is None else N
    check_spin(L, s_in, N)
    check_spin(L, s_out)
    entry = (
        parity(s_in + s_out)
        * jnp.conj(c)
        * (np.arange(L) >= max(abs(s_in), abs(s_out)))
    )
    dtype = jnp.result_type(entry, jnp.complex64)
    out = jnp.zeros(c.shape[:-1] + (2 * N - 1, L, 2 * L - 1), dtype=dtype)
    return out.at[..., N - 1 - s_in, :, L - 1 - s_out].set(entry)


def identity(L: int, spin: int, N: int = None) -> jnp.ndarray:
    r"""
    The single entry :math:`\Psi^{\ell}_{-s,-s} = 1`, the kernel of the lift and projection.

    It gives :math:`\mathcal{C}_{s\to G} = \Lambda_s`, :math:`\mathcal{C}_{G\to s} = \Lambda_s^\dagger`
    and :math:`\mathcal{C}_{s\to s} = \mathrm{Id}` (Corollary V.6).  It is a distribution,
    not a square-integrable kernel (Theorem V.5(b)).

    Args:
        L (int): Harmonic band-limit.
        spin (int): Spin :math:`s`.
        N (int, optional): Azimuthal band-limit of the kernel. Defaults to :math:`|s| + 1`.

    Returns:
        jnp.ndarray: The kernel, ``[2N - 1, L, 2L - 1]``.

    """
    return multiplier(np.ones(L), spin, spin, N)


def steerable(multipliers: Mapping, s_in: int, N: int = None) -> jnp.ndarray:
    r"""
    A kernel for a tuple of spin-changing convolutions (Proposition VI.6).

    With :math:`\mathcal{C}_{s\to G}` the column :math:`k = -s` supported on :math:`|n| < N`
    produces the :math:`2N - 1` spin components :math:`\mathcal{C}_{s\to s'}`,
    :math:`|s'| < N`; this constructs that column from their multipliers.

    Args:
        multipliers (Mapping): Map from output spin :math:`s'` to multiplier :math:`c_\ell`.
        s_in (int): Input spin :math:`s`.
        N (int, optional): Azimuthal band-limit of the kernel. Defaults to :math:`|s| + 1`.

    Returns:
        jnp.ndarray: The kernel, ``[2N - 1, L, 2L - 1]``.

    """
    return sum(multiplier(c, s_in, s_out, N) for s_out, c in multipliers.items())


# ------------------------------------------------------------------------------ conversions


def to_multiplier(kernel, s_in: int, s_out: int) -> jnp.ndarray:
    r"""
    The multiplier :math:`c_\ell = (-1)^{s+s'} \Psi^{\ell *}_{-s',-s}` of :math:`\mathcal{C}_{s\to s'}[\Psi]` (Eq. 43).

    Args:
        kernel: The kernel, ``[..., 2N - 1, L, 2L - 1]``.
        s_in (int): Input spin :math:`s`.
        s_out (int): Output spin :math:`s'`.

    Returns:
        jnp.ndarray: :math:`c_\ell`, ``[..., L]``, zero for :math:`\ell < \max(|s|, |s'|)`.

    """
    N, L = azimuthal_bandlimit(kernel), kernel.shape[-2]
    if abs(s_in) >= N:
        return jnp.zeros(kernel.shape[:-3] + (L,), dtype=kernel.dtype)
    c = parity(s_in + s_out) * jnp.conj(kernel[..., N - 1 - s_in, :, L - 1 - s_out])
    return c * (np.arange(L) >= max(abs(s_in), abs(s_out)))


def to_column(kernel, s_in: int, N_out: int) -> jnp.ndarray:
    r"""
    The weights :math:`u^{\ell}_n = (-1)^s a_\ell^{-1/2} \Psi^{\ell *}_{n,-s}` of :math:`\mathcal{C}_{s\to G}[\Psi]` (Eq. 53).

    Args:
        kernel: The kernel, ``[..., 2N - 1, L, 2L - 1]``.
        s_in (int): Input spin :math:`s`.
        N_out (int): Azimuthal band-limit of the output.

    Returns:
        jnp.ndarray: :math:`u`, ``[..., 2N_out - 1, L]``.

    """
    L = kernel.shape[-2]
    column = _second_index(kernel, abs(s_in) + 1)[
        ..., abs(s_in) - s_in, :, L - N_out : L + N_out - 1
    ]
    weight = parity(s_in) / np.sqrt(_a(L))
    return jnp.swapaxes(jnp.conj(column) * weight[:, None], -1, -2)


def to_row(kernel, s_out: int, N_in: int) -> jnp.ndarray:
    r"""
    The weights :math:`v^{\ell}_k = (-1)^{s'} a_\ell^{1/2} \Psi^{\ell *}_{-s',k}` of :math:`\mathcal{C}_{G\to s'}[\Psi]` (Eq. 53).

    Args:
        kernel: The kernel, ``[..., 2N - 1, L, 2L - 1]``.
        s_out (int): Output spin :math:`s'`.
        N_in (int): Azimuthal band-limit of the input.

    Returns:
        jnp.ndarray: :math:`v`, ``[..., 2N_in - 1, L]``.

    """
    L = kernel.shape[-2]
    row = _second_index(kernel, N_in)[..., :, :, L - 1 - s_out]
    return parity(s_out) * jnp.conj(row) * np.sqrt(_a(L))


def to_block(kernel, N_in: int, N_out: int) -> jnp.ndarray:
    r"""
    The weights :math:`B^{\ell} = \Psi^{\ell\dagger}` of :math:`\mathcal{C}_{G\to G}[\Psi]` (Eq. 53).

    Args:
        kernel: The kernel, ``[..., 2N - 1, L, 2L - 1]``.
        N_in (int): Azimuthal band-limit of the input.
        N_out (int): Azimuthal band-limit of the output.

    Returns:
        jnp.ndarray: :math:`B`, ``[..., L, 2N_in - 1, 2N_out - 1]``, with
        ``B[..., el, N_in - 1 + k, N_out - 1 + n]`` :math:`= \Psi^{\ell *}_{nk}`.

    """
    L = kernel.shape[-2]
    block = _second_index(kernel, N_in)[..., :, :, L - N_out : L + N_out - 1]
    return jnp.conj(jnp.moveaxis(block, -2, -3))


def from_multiplier(c, s_in: int, s_out: int, N: int = None) -> jnp.ndarray:
    """Alias of :func:`multiplier`: the canonical kernel of a multiplier."""
    return multiplier(c, s_in, s_out, N)


def from_column(u, s_in: int, N: int = None) -> jnp.ndarray:
    """
    The canonical kernel realising column weights :math:`u` (Theorem V.5(c)).

    Args:
        u: Weights ``[..., 2N_out - 1, L]``.
        s_in (int): Input spin :math:`s`.
        N (int, optional): Azimuthal band-limit of the kernel. Defaults to :math:`|s| + 1`.

    Returns:
        jnp.ndarray: A kernel whose only nonzero column is :math:`k = -s`.

    """
    u = jnp.asarray(u)
    N_out, L = (u.shape[-2] + 1) // 2, u.shape[-1]
    N = abs(s_in) + 1 if N is None else N
    check_spin(L, s_in, N)
    column = jnp.conj(
        jnp.swapaxes(u, -1, -2) * (parity(s_in) * np.sqrt(_a(L)))[:, None]
    )
    dtype = jnp.result_type(column, jnp.complex64)
    out = jnp.zeros(u.shape[:-2] + (2 * N - 1, L, 2 * L - 1), dtype=dtype)
    return out.at[..., N - 1 - s_in, :, L - N_out : L + N_out - 1].set(column)


def from_row(v, s_out: int) -> jnp.ndarray:
    """
    The canonical kernel realising row weights :math:`v` (Theorem V.5(c)).

    Args:
        v: Weights ``[..., 2N_in - 1, L]``.
        s_out (int): Output spin :math:`s'`.

    Returns:
        jnp.ndarray: A kernel ``[..., 2N_in - 1, L, 2L - 1]`` whose only nonzero row is
        :math:`n = -s'`.

    """
    v = jnp.asarray(v)
    L = v.shape[-1]
    row = jnp.conj(v * (parity(s_out) / np.sqrt(_a(L))))
    dtype = jnp.result_type(row, jnp.complex64)
    out = jnp.zeros(v.shape + (2 * L - 1,), dtype=dtype)
    return out.at[..., :, :, L - 1 - s_out].set(row)


def from_block(B) -> jnp.ndarray:
    r"""
    The kernel :math:`\Psi^{\ell} = B^{\ell\dagger}` realising block weights (Theorem V.5(c)).

    Args:
        B: Weights ``[..., L, 2N_in - 1, 2N_out - 1]``.

    Returns:
        jnp.ndarray: The kernel ``[..., 2N_in - 1, L, 2L - 1]``.

    """
    B = jnp.asarray(B)
    L, N_in, N_out = B.shape[-3], (B.shape[-2] + 1) // 2, (B.shape[-1] + 1) // 2
    block = jnp.conj(jnp.moveaxis(B, -3, -2))
    dtype = jnp.result_type(block, jnp.complex64)
    out = jnp.zeros(B.shape[:-3] + (2 * N_in - 1, L, 2 * L - 1), dtype=dtype)
    out = out.at[..., :, :, L - N_out : L + N_out - 1].set(block)
    return out * wigner_mask(L, N_in)


def compose(kernel2, kernel1) -> jnp.ndarray:
    r"""
    The kernel of a composition, :math:`\Psi_3^{\ell} = \Psi_2^{\ell} \Psi_1^{\ell}` (Proposition V.3).

    :math:`\mathcal{C}_{G\to G}[\Psi_2]\, \mathcal{C}_{G\to G}[\Psi_1] = \mathcal{C}_{G\to G}[\Psi_3]`,
    and likewise with a lift or projection at either end.  The product runs over the
    intermediate index, restricted to the azimuthal band-limit of ``kernel2``.

    Args:
        kernel2: The kernel applied second, ``[..., 2N_2 - 1, L, 2L - 1]``.
        kernel1: The kernel applied first, ``[..., 2N_1 - 1, L, 2L - 1]``.

    Returns:
        jnp.ndarray: :math:`\Psi_3`, ``[..., 2N_1 - 1, L, 2L - 1]``.

    """
    N2, L = azimuthal_bandlimit(kernel2), kernel2.shape[-2]
    return jnp.einsum(
        "...jln,...klj->...kln", kernel2, kernel1[..., L - N2 : L + N2 - 1]
    )
