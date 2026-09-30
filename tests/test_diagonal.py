"""
Operators diagonal in the spherical harmonic basis (Sec. VII B, Proposition VII.2).

Harmonic multiplication and the sifting convolution multiply each coefficient by g_lm or
g^*_lm (Definition III.9).  They commute with polar rotations, and are lifted convolutions
C_{0->0}[Psi_T] exactly when g_lm does not depend on m.
"""

import numpy as np
import reference as ref
from s2fft.utils.signal_generator import generate_flm

from s2conv import kernels
from s2conv.convolutions import diagonal, lifted

L = 12


def test_definitions(rng):
    flm, glm = generate_flm(rng, L), generate_flm(rng, L)
    assert ref.rel_err(diagonal.harmonic_multiplication(flm, glm), glm * flm) < ref.TOL
    assert (
        ref.rel_err(diagonal.sifting_convolution(flm, glm), flm * np.conj(glm))
        < ref.TOL
    )


def test_sifting_real_space_identity(rng):
    """(f (o) g)(theta, phi) = [(g (o) f)(theta, -phi)]^* (Definition III.9)."""
    flm, glm = generate_flm(rng, L), generate_flm(rng, L)
    reverse = (-np.arange(2 * L - 1)) % (2 * L - 1)
    fg = ref.isht(diagonal.sifting_convolution(flm, glm), L)
    gf = ref.isht(diagonal.sifting_convolution(glm, flm), L)
    assert ref.rel_err(fg, np.conj(gf[:, reverse])) < ref.TOL


def test_m_independent_multiplier_is_a_lifted_convolution(rng):
    """g_lm = g_l0 for every m gives C_{0->0}[Psi_T] with c_l = g_l0 (Proposition VII.2)."""
    flm, glm = generate_flm(rng, L), generate_flm(rng, L)
    g0 = glm[:, L - 1]
    isotropic = np.repeat(g0[:, None], 2 * L - 1, axis=1) * ref.lm_mask(L)
    out = diagonal.harmonic_multiplication(flm, isotropic)
    assert (
        ref.rel_err(out, lifted.spin_to_spin(flm, kernels.multiplier(g0, 0, 0), 0, 0))
        < ref.TOL
    )
    sifted = diagonal.sifting_convolution(flm, isotropic)
    assert (
        ref.rel_err(
            sifted,
            lifted.spin_to_spin(flm, kernels.multiplier(np.conj(g0), 0, 0), 0, 0),
        )
        < ref.TOL
    )


def test_diagonal_operators_commute_with_polar_rotations(rng):
    """A polar rotation multiplies f_lm by e^{-im a}, which commutes with any diagonal operator."""
    flm, glm = generate_flm(rng, L), generate_flm(rng, L)
    polar = (1.1, 0.0, 0.0)
    for operator in (diagonal.harmonic_multiplication, diagonal.sifting_convolution):
        lhs = operator(ref.rotate_flm(flm, polar), glm)
        assert (
            ref.rel_err(lhs, ref.rotate_flm(np.asarray(operator(flm, glm)), polar))
            < ref.TOL
        )
