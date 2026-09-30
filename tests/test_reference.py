"""
Checks of the independent references themselves, against s2fft alone.

The other test modules trust :mod:`reference`; these tests make sure that its evaluation
of spin fields and of functions on the rotation group, its rotations and its quadratures
agree with s2fft's transforms, so that a failure elsewhere points at s2conv.
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm, generate_flmn

L = 10


@pytest.mark.parametrize("s", [0, 1, -2, 3])
def test_eval_spin_matches_s2fft_on_the_grid(rng, s):
    """Eq. (A7) with Risbo Wigner functions reproduces s2fft's inverse transform."""
    flm = generate_flm(rng, L, spin=s)
    thetas, phis = ref.sphere_grid(L)
    T, P = np.meshgrid(thetas, phis, indexing="ij")
    assert ref.rel_err(ref.eval_spin(flm, s, T, P), ref.isht(flm, L, s)) < ref.TOL


def test_eval_so3_matches_s2fft_on_the_grid(rng):
    """Eq. (7) with Risbo Wigner functions reproduces s2fft's inverse Wigner transform."""
    N = 3
    flmn = generate_flmn(rng, L, N)
    samples = ref.wigner_inv(flmn, L, N)
    thetas, phis = ref.sphere_grid(L)
    J, T, P = np.meshgrid(
        np.arange(2 * N - 1),
        np.arange(len(thetas)),
        np.arange(len(phis)),
        indexing="ij",
    )
    values = ref.eval_so3(flmn, phis[P], thetas[T], ref.gamma_grid(N)[J])
    assert ref.rel_err(values, samples) < ref.TOL


@pytest.mark.parametrize("s", [0, 2, -1])
def test_reference_rotation_harmonic_vs_real_space(rng, s):
    """Eq. (10) on coefficients agrees with Eq. (12) in real space, twist phase included."""
    flm = generate_flm(rng, L, spin=s)
    rotation = ref.random_rotation(rng)
    theta, phi = ref.random_points(rng, 30)
    lhs = ref.eval_spin(ref.rotate_flm(flm, rotation), s, theta, phi)
    assert ref.rel_err(lhs, ref.rotated_values(flm, s, rotation, theta, phi)) < ref.TOL


def test_quadrature_is_exact_for_products(rng):
    """The Gauss-Legendre rule integrates <f, g> of band-limited spin fields exactly."""
    s = 2
    f, g = generate_flm(rng, L, spin=s), generate_flm(rng, L, spin=s)
    T, P, W = ref.gl_quadrature(L)
    by_quadrature = np.sum(
        W * ref.eval_spin(f, s, T, P) * np.conj(ref.eval_spin(g, s, T, P))
    )
    assert abs(by_quadrature - ref.inner_s2(f, g)) < ref.TOL * abs(ref.inner_s2(f, g))


def test_so3_quadrature_is_exact_for_products(rng):
    """The quadrature on SO(3) integrates <F, G> of band-limited functions exactly (Parseval, Eq. 7)."""
    N = 2
    F, G = generate_flmn(rng, L, N), generate_flmn(rng, L, N)
    A, B, C, W = ref.so3_quadrature(L)
    by_quadrature = np.sum(
        W * ref.eval_so3(F, A, B, C) * np.conj(ref.eval_so3(G, A, B, C))
    )
    assert abs(by_quadrature - ref.inner_so3(F, G)) < ref.TOL * abs(ref.inner_so3(F, G))


def test_reference_wigner_round_trip(rng):
    """s2fft's NumPy Wigner transforms, used by the literal checks, invert each other."""
    N = 4
    flmn = generate_flmn(rng, L, N)
    assert ref.rel_err(ref.wigner_fwd(ref.wigner_inv(flmn, L, N), L, N), flmn) < ref.TOL


def test_block_layout_round_trip(rng):
    """The block <-> s2fft layout conversions invert each other."""
    N = 3
    block = ref.random_block(L, rng, N)
    assert (
        ref.rel_err(ref.kernel_to_block(ref.block_to_kernel(block, N)), block) < ref.TOL
    )


def test_needlet_window_partition_of_unity():
    """sum_j b(xi / B^j)^2 = 1 for xi >= 1 (Marinucci et al.)."""
    xi = np.linspace(1.0, 60.0, 500)
    total = sum(ref.needlet_window(xi / 2.0**j) ** 2 for j in range(8))
    assert np.max(np.abs(total - 1)) < 1e-12
