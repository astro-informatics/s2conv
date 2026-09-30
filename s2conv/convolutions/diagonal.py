r"""
Operators diagonal in the spherical harmonic basis (Sec. VII B).

Harmonic multiplication of Kennedy et al. and the sifting convolution of Roddy & McEwen
multiply each coefficient by :math:`g_{\ell m}` or :math:`g^*_{\ell m}` (Definition III.9).
They commute with every polar rotation, and are equivariant under all rotations exactly
when :math:`g_{\ell m}` does not depend on :math:`m`, in which case they are the lifted
convolution :math:`\mathcal{C}_{0\to0}[\Psi_T]` with :math:`c_\ell = g_{\ell0}` (Proposition VII.2).
"""

import jax.numpy as jnp


def harmonic_multiplication(flm, glm) -> jnp.ndarray:
    r"""
    Harmonic multiplication :math:`(g \odot_{\mathrm{K}} f)_{\ell m} = g_{\ell m} f_{\ell m}` (Definition III.9).

    Args:
        flm: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.
        glm: Scalar harmonic coefficients of the filter, ``[..., L, 2L - 1]``.

    Returns:
        jnp.ndarray: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    return glm * flm


def sifting_convolution(flm, glm) -> jnp.ndarray:
    r"""
    Sifting convolution :math:`(f \circledcirc g)_{\ell m} = f_{\ell m} g^*_{\ell m}` (Definition III.9).

    Args:
        flm: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.
        glm: Scalar harmonic coefficients of the filter, ``[..., L, 2L - 1]``.

    Returns:
        jnp.ndarray: Scalar harmonic coefficients, ``[..., L, 2L - 1]``.

    """
    return flm * jnp.conj(glm)
