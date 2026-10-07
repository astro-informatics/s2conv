r"""
The generalised lifted spherical convolution (Sec. V).

For a kernel :math:`\Psi` on the rotation group and spins :math:`s, s'`, the four operators
of Definition V.1 are

.. math::

    \mathcal{C}_{s\to s'}[\Psi] = \Lambda_{s'}^\dagger (\Psi \star \Lambda_s\,\cdot\,), \quad
    \mathcal{C}_{s\to G}[\Psi] = \Psi \star \Lambda_s\,\cdot\,, \quad
    \mathcal{C}_{G\to s'}[\Psi] = \Lambda_{s'}^\dagger (\Psi \star\,\cdot\,), \quad
    \mathcal{C}_{G\to G}[\Psi] = \Psi \star\,\cdot\,,

with :math:`\star` the group convolution in correlation form,
:math:`(\Psi \star F)(\rho) = \int \Psi^*(\rho^{-1}\rho') F(\rho')\, \mathrm{d}\mu(\rho')`
(Definition III.4).  In harmonic space they read a single entry, column, row or the
whole block of the kernel (Theorem V.2, Eqs. 42–45):

.. math::

    (\mathcal{C}_{s\to G}[\Psi]\, {}_sf)^{\ell}_{mn} &= (-1)^s \sqrt{\tfrac{8\pi^2}{2\ell+1}}\, {}_sf_{\ell m}\, \Psi^{\ell *}_{n,-s}, \\
    (\mathcal{C}_{s\to s'}[\Psi]\, {}_sf)_{\ell m} &= (-1)^{s+s'}\, {}_sf_{\ell m}\, \Psi^{\ell *}_{-s',-s}, \\
    (\mathcal{C}_{G\to s'}[\Psi]\, F)_{\ell m} &= (-1)^{s'} \sqrt{\tfrac{2\ell+1}{8\pi^2}} \sum_{m'} F^{\ell}_{mm'}\, \Psi^{\ell *}_{-s',m'}, \\
    (\mathcal{C}_{G\to G}[\Psi]\, F)^{\ell}_{mn} &= \sum_{m'} F^{\ell}_{mm'}\, \Psi^{\ell *}_{nm'} .

Every bounded rotation-equivariant map of the four types is of this form, and acts
through compact weights (Theorem V.5): a multiplier :math:`c_\ell`, a column
:math:`u^{\ell}_n`, a row :math:`v^{\ell}_k` or a block :math:`B^{\ell}`.  This module applies
the operators in both parameterisations; :mod:`s2conv.kernels` converts between them.
All functions act on harmonic coefficients in the s2fft layout, broadcast over leading
axes, and are differentiable and JIT compilable (with spins and band-limits static).
"""

import jax.numpy as jnp

from s2conv import kernels
from s2conv.utils.indexing import azimuthal_bandlimit, lm_mask, wigner_mask

# ------------------------------------------------------------------------------ weight form


def spin_multiplier(flm, c, s_in: int, s_out: int) -> jnp.ndarray:
    r"""
    A spin-changing multiplier, :math:`({}_{s'}g)_{\ell m} = c_\ell\, {}_sf_{\ell m}` (Theorem V.5).

    This is the general bounded rotation-equivariant map from spin :math:`s` to spin
    :math:`s'`; degrees :math:`\ell < \max(|s|, |s'|)` are absent from the target and are
    annihilated.

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        c: Multiplier :math:`c_\ell`, ``[..., L]``.
        s_in (int): Input spin :math:`s`.
        s_out (int): Output spin :math:`s'`.

    Returns:
        jnp.ndarray: Spin-:math:`s'` harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    L = flm.shape[-2]
    return jnp.asarray(c)[..., :, None] * flm * lm_mask(L, max(abs(s_in), abs(s_out)))


def spin_to_group_weights(flm, u, s_in: int) -> jnp.ndarray:
    r"""
    An equivariant map from spin :math:`s` to the rotation group, :math:`F^{\ell}_{mn} = u^{\ell}_n\, {}_sf_{\ell m}` (Theorem V.5(c)).

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        u: Column weights ``[..., 2N - 1, L]``.
        s_in (int): Input spin :math:`s`.

    Returns:
        jnp.ndarray: Wigner coefficients, ``[..., 2N - 1, L, 2L - 1]``.

    """
    L, N = flm.shape[-2], (u.shape[-2] + 1) // 2
    return (
        u[..., :, :, None]
        * (flm * lm_mask(L, s_in))[..., None, :, :]
        * wigner_mask(L, N)
    )


def group_to_spin_weights(flmn, v, s_out: int) -> jnp.ndarray:
    r"""
    An equivariant map from the rotation group to spin :math:`s'`, :math:`g_{\ell m} = \sum_k v^{\ell}_k F^{\ell}_{mk}` (Theorem V.5(c)).

    Args:
        flmn: Wigner coefficients, ``[..., 2N - 1, L, 2L - 1]``.
        v: Row weights ``[..., 2N - 1, L]``.
        s_out (int): Output spin :math:`s'`.

    Returns:
        jnp.ndarray: Spin-:math:`s'` harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    L = flmn.shape[-2]
    return jnp.einsum("...kl,...klm->...lm", v, flmn) * lm_mask(L, s_out)


def group_to_group_weights(flmn, B) -> jnp.ndarray:
    r"""
    An equivariant map on the rotation group, :math:`G^{\ell} = F^{\ell} B^{\ell}` (Theorem V.5(c)).

    Args:
        flmn: Wigner coefficients, ``[..., 2N_in - 1, L, 2L - 1]``.
        B: Block weights ``[..., L, 2N_in - 1, 2N_out - 1]``.

    Returns:
        jnp.ndarray: Wigner coefficients, ``[..., 2N_out - 1, L, 2L - 1]``.

    """
    L, N_out = flmn.shape[-2], (B.shape[-1] + 1) // 2
    return jnp.einsum("...klm,...lkn->...nlm", flmn, B) * wigner_mask(L, N_out)


# ------------------------------------------------------------------------------ kernel form


def spin_to_spin(flm, kernel, s_in: int, s_out: int) -> jnp.ndarray:
    r"""
    :math:`\mathcal{C}_{s\to s'}[\Psi]`, which reads the single entry :math:`\Psi^{\ell}_{-s',-s}` (Eq. 43).

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        kernel: The kernel :math:`\Psi`, ``[2N - 1, L, 2L - 1]``.
        s_in (int): Input spin :math:`s`.
        s_out (int): Output spin :math:`s'`.

    Returns:
        jnp.ndarray: Spin-:math:`s'` harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    return spin_multiplier(flm, kernels.to_multiplier(kernel, s_in, s_out), s_in, s_out)


def spin_to_group(flm, kernel, s_in: int, N_out: int = None) -> jnp.ndarray:
    r"""
    :math:`\mathcal{C}_{s\to G}[\Psi]`, which reads the column :math:`k = -s` (Eq. 42).

    Args:
        flm: Spin-:math:`s` harmonic coefficients, ``[..., L, 2L - 1]``.
        kernel: The kernel :math:`\Psi`, ``[2N - 1, L, 2L - 1]``.
        s_in (int): Input spin :math:`s`.
        N_out (int, optional): Azimuthal band-limit of the output.  Defaults to :math:`L`.

    Returns:
        jnp.ndarray: Wigner coefficients, ``[..., 2N_out - 1, L, 2L - 1]``.

    """
    N_out = flm.shape[-2] if N_out is None else N_out
    return spin_to_group_weights(flm, kernels.to_column(kernel, s_in, N_out), s_in)


def group_to_spin(flmn, kernel, s_out: int) -> jnp.ndarray:
    r"""
    :math:`\mathcal{C}_{G\to s'}[\Psi]`, which reads the row :math:`n = -s'` (Eq. 44).

    Args:
        flmn: Wigner coefficients, ``[..., 2N - 1, L, 2L - 1]``.
        kernel: The kernel :math:`\Psi`, ``[2N_k - 1, L, 2L - 1]``.
        s_out (int): Output spin :math:`s'`.

    Returns:
        jnp.ndarray: Spin-:math:`s'` harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    v = kernels.to_row(kernel, s_out, azimuthal_bandlimit(flmn))
    return group_to_spin_weights(flmn, v, s_out)


def group_to_group(flmn, kernel, N_out: int = None) -> jnp.ndarray:
    r"""
    :math:`\mathcal{C}_{G\to G}[\Psi] = \Psi \star\,\cdot\,`, the group convolution (Eqs. 20, 45).

    Args:
        flmn: Wigner coefficients, ``[..., 2N_in - 1, L, 2L - 1]``.
        kernel: The kernel :math:`\Psi`, ``[2N_k - 1, L, 2L - 1]``.
        N_out (int, optional): Azimuthal band-limit of the output.  Defaults to :math:`L`.

    Returns:
        jnp.ndarray: Wigner coefficients, ``[..., 2N_out - 1, L, 2L - 1]``.

    """
    N_out = flmn.shape[-2] if N_out is None else N_out
    B = kernels.to_block(kernel, azimuthal_bandlimit(flmn), N_out)
    return group_to_group_weights(flmn, B)


group_convolution = group_to_group


def lifted_convolution(x, kernel, s_in, s_out, N_out: int = None) -> jnp.ndarray:
    r"""
    The generalised lifted convolution :math:`\mathcal{C}_{s\to s'}[\Psi]` of Definition V.1.

    Args:
        x: Spin-:math:`s` harmonic coefficients ``[..., L, 2L - 1]`` or, if
            ``s_in == "G"``, Wigner coefficients ``[..., 2N - 1, L, 2L - 1]``.
        kernel: The kernel :math:`\Psi`, ``[2N_k - 1, L, 2L - 1]``.
        s_in (int or "G"): Input spin, or ``"G"`` for an input on the rotation group.
        s_out (int or "G"): Output spin, or ``"G"`` for an output on the rotation group.
        N_out (int, optional): Azimuthal band-limit of an output on the rotation group.
            Defaults to :math:`L`.

    Returns:
        jnp.ndarray: Harmonic or Wigner coefficients of the output.

    """
    if s_in == "G":
        return (
            group_to_group(x, kernel, N_out)
            if s_out == "G"
            else group_to_spin(x, kernel, s_out)
        )
    return (
        spin_to_group(x, kernel, s_in, N_out)
        if s_out == "G"
        else spin_to_spin(x, kernel, s_in, s_out)
    )
