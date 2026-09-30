"""
The lift of spin fields to the rotation group and its adjoint projection (Sec. IV).

The harmonic forms of Lemma IV.7(c) (Eq. 39) are checked against the literal operators
on samples (Eq. 38) followed by s2fft's transforms; the projection inverts the lift, is
its adjoint, and the fibre decomposition of Corollary IV.12 reassembles any function on
the rotation group from its spin components.
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm, generate_flmn

from s2conv import lifting

L = 12
N = 4


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_lift_is_the_sampled_lift(rng, spin):
    """(Lambda_s f)(alpha, beta, gamma) = (2 pi)^{-1/2} e^{-is gamma} f(beta, alpha) on the SO(3) grid,
    Wigner transformed, is the single column n = -s of Eq. (39)."""
    flm = generate_flm(rng, L, spin=spin)
    sampled = ref.lift_grid(ref.isht(flm, L, spin), spin, N)
    assert (
        ref.rel_err(lifting.lift(flm, spin, N), ref.wigner_fwd(sampled, L, N)) < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_project_is_the_fibre_integral(rng, spin):
    """(Lambda_s^dagger F)(beta, alpha) = (2 pi)^{-1/2} int e^{is gamma} F d gamma, spin transformed,
    reads the column n = -s of Eq. (39)."""
    flmn = generate_flmn(rng, L, N)
    projected = ref.project_grid(ref.wigner_inv(flmn, L, N), spin)
    assert (
        ref.rel_err(lifting.project(flmn, spin), ref.sht(projected, L, spin)) < ref.TOL
    )


@pytest.mark.parametrize(("spin", "n_out"), [(0, None), (1, None), (-2, 5), (3, 4)])
def test_lift_matches_closed_form(rng, spin, n_out):
    flm = generate_flm(rng, L, spin=spin)
    lifted = lifting.lift(flm, spin, n_out)
    n_expected = abs(spin) + 1 if n_out is None else n_out
    assert lifted.shape == (2 * n_expected - 1, L, 2 * L - 1)
    assert ref.rel_err(lifted, ref.lift(flm, spin, n_expected)) < ref.TOL


@pytest.mark.parametrize("spin", [0, 3, -1])
def test_project_matches_closed_form(rng, spin):
    flmn = generate_flmn(rng, L, N)
    assert ref.rel_err(lifting.project(flmn, spin), ref.project(flmn, spin)) < ref.TOL


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_lift_samples_and_project_samples(rng, spin):
    f = ref.isht(generate_flm(rng, L, spin=spin), L, spin)
    assert (
        ref.rel_err(lifting.lift_samples(f, spin, N), ref.lift_grid(f, spin, N))
        < ref.TOL
    )
    F = ref.wigner_inv(generate_flmn(rng, L, N), L, N)
    assert (
        ref.rel_err(lifting.project_samples(F, spin), ref.project_grid(F, spin))
        < ref.TOL
    )


def test_lift_samples_default_bandlimit(rng):
    f = ref.isht(generate_flm(rng, L, spin=2), L, 2)
    assert lifting.lift_samples(f, 2).shape == (5,) + f.shape


def test_gamma_samples():
    np.testing.assert_allclose(
        lifting.gamma_samples(N), 2 * np.pi * np.arange(2 * N - 1) / (2 * N - 1)
    )


@pytest.mark.parametrize(("spin", "n_out"), [(0, 1), (2, 3), (-3, 6)])
def test_projection_inverts_the_lift(rng, spin, n_out):
    """Lambda_s^dagger Lambda_s = Id: the lift is an isometry (Lemma IV.7(a))."""
    flm = generate_flm(rng, L, spin=spin)
    assert (
        ref.rel_err(lifting.project(lifting.lift(flm, spin, n_out), spin), flm)
        < ref.TOL
    )


def test_projection_inverts_the_lift_on_samples(rng):
    spin = -2
    f = ref.isht(generate_flm(rng, L, spin=spin), L, spin)
    assert (
        ref.rel_err(lifting.project_samples(lifting.lift_samples(f, spin, N), spin), f)
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_projection_is_the_adjoint_of_the_lift(rng, spin):
    """<Lambda_s f, F>_SO(3) = <f, Lambda_s^dagger F>_S2, with the inner products of Eq. (7)."""
    flm = generate_flm(rng, L, spin=spin)
    flmn = generate_flmn(rng, L, N)
    lhs = ref.inner_so3(lifting.lift(flm, spin, N), flmn)
    rhs = ref.inner_s2(flm, lifting.project(flmn, spin))
    assert abs(lhs - rhs) < ref.TOL * abs(lhs)


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_lift_and_projection_intertwine_rotations(rng, spin):
    """Lambda_s pi_s(g) = pi_G(g) Lambda_s and Lambda_s^dagger pi_G(g) = pi_s(g) Lambda_s^dagger: rotating a
    spin field (Eq. 10, with the twist phase of Eq. 12) is left translation of its lift (Prop. IV.10)."""
    flm = generate_flm(rng, L, spin=spin)
    flmn = generate_flmn(rng, L, N)
    g = ref.random_rotation(rng)
    lhs = lifting.lift(ref.rotate_flm(flm, g), spin, N)
    assert (
        ref.rel_err(lhs, ref.rotate_flmn(np.asarray(lifting.lift(flm, spin, N)), g))
        < ref.TOL
    )
    lhs = lifting.project(ref.rotate_flmn(flmn, g), spin)
    assert (
        ref.rel_err(lhs, ref.rotate_flm(np.asarray(lifting.project(flmn, spin)), g))
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_lift_is_an_isometry(rng, spin):
    """||Lambda_s f||_SO(3) = ||f||_S2 (Lemma IV.7(a)), with the norm of Eq. (7) on the group."""
    flm = generate_flm(rng, L, spin=spin)
    lifted_norm = ref.inner_so3(lifting.lift(flm, spin, N), lifting.lift(flm, spin, N))
    assert abs(lifted_norm - ref.inner_s2(flm, flm)) < ref.TOL * abs(
        ref.inner_s2(flm, flm)
    )


@pytest.mark.parametrize("spin", [N, -N, N + 2])
def test_projection_vanishes_beyond_the_azimuthal_bandlimit(rng, spin):
    flmn = generate_flmn(rng, L, N)
    out = lifting.project(flmn, spin)
    assert out.shape == (L, 2 * L - 1)
    assert np.all(out == 0)


def test_lift_rejects_unrepresentable_spins(rng):
    flm = generate_flm(rng, L)
    with pytest.raises(ValueError):
        lifting.lift(flm, 3, 3)
    with pytest.raises(ValueError):
        lifting.lift(flm, L, L + 1)


def test_spin_components_are_projections(rng):
    """Corollary IV.12: the components are Lambda_s^dagger F, indexed [N - 1 + s]."""
    flmn = generate_flmn(rng, L, N)
    components = lifting.spin_components(flmn)
    assert components.shape == (2 * N - 1, L, 2 * L - 1)
    for s in range(-(N - 1), N):
        assert ref.rel_err(components[N - 1 + s], ref.project(flmn, s)) < ref.TOL


def test_fibre_decomposition_in_harmonic_space(rng):
    """F = sum_{|s| < N} Lambda_s Lambda_s^dagger F (Corollary IV.12)."""
    flmn = generate_flmn(rng, L, N)
    assert (
        ref.rel_err(lifting.from_spin_components(lifting.spin_components(flmn)), flmn)
        < ref.TOL
    )
    assembled = sum(ref.lift(ref.project(flmn, s), s, N) for s in range(-(N - 1), N))
    assert (
        ref.rel_err(
            lifting.from_spin_components(lifting.spin_components(flmn)), assembled
        )
        < ref.TOL
    )


def test_spin_components_of_assembled_fields(rng):
    components = np.stack([generate_flm(rng, L, spin=s) for s in range(-(N - 1), N)])
    assert (
        ref.rel_err(
            lifting.spin_components(lifting.from_spin_components(components)),
            components,
        )
        < ref.TOL
    )


def test_fibre_decomposition_on_samples(rng):
    """F = sum_{|s| < N} Lambda_s Lambda_s^dagger F on the SO(3) grid (Lemma IV.7(a), Cor. IV.12)."""
    F = ref.wigner_inv(generate_flmn(rng, L, N), L, N)
    total = sum(
        lifting.lift_samples(lifting.project_samples(F, s), s, N)
        for s in range(-(N - 1), N)
    )
    assert ref.rel_err(total, F) < ref.TOL
