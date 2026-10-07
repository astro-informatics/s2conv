"""
Index conventions, Wigner functions, rotations and point evaluation of :mod:`s2conv.utils`.

The Wigner functions are checked against s2fft's Risbo recursion, rotations against an
explicit Wigner-matrix product and against real-space rotation with the spin twist phase
(Eq. 12), and point evaluation against s2fft's inverse transforms and the definition of
the spin spherical harmonics (Eq. A7).
"""

import jax
import numpy as np
import pytest
import reference as ref
from s2fft.recursions import risbo
from s2fft.utils.signal_generator import generate_flm, generate_flmn

from s2conv import utils
from s2conv.utils import indexing, rotation

L = 12

# ------------------------------------------------------------------------------ indexing


def test_axes_and_parity():
    np.testing.assert_array_equal(utils.m_axis(3), [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(utils.n_axis(2), [-1, 0, 1])
    assert [indexing.parity(s) for s in (-3, -2, -1, 0, 1, 2, 3)] == [
        -1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
    ]
    assert indexing.parity(np.int64(-5)) == -1


@pytest.mark.parametrize("spin", [0, 1, -3])
def test_lm_mask(spin):
    mask = utils.lm_mask(L, spin)
    np.testing.assert_array_equal(mask, ref.lm_mask(L, spin))
    assert mask.sum() == L**2 - spin**2


def test_wigner_mask():
    N = 3
    mask = utils.wigner_mask(L, N)
    assert mask.shape == (2 * N - 1, L, 2 * L - 1)
    for n in range(-(N - 1), N):
        np.testing.assert_array_equal(mask[N - 1 + n], ref.lm_mask(L, n))


def test_bandlimits(rng):
    assert utils.bandlimit(generate_flm(rng, L)) == L
    assert utils.azimuthal_bandlimit(generate_flmn(rng, L, 4)) == 4


def test_check_spin():
    indexing.check_spin(L, L - 1)
    indexing.check_spin(L, -2, 3)
    with pytest.raises(ValueError, match="band-limit"):
        indexing.check_spin(L, L)
    with pytest.raises(ValueError, match="azimuthal"):
        indexing.check_spin(L, 3, 3)


# ------------------------------------------------------------------------ Wigner functions


@pytest.mark.parametrize("beta", [0.0, 0.3, np.pi / 2, 2.0, np.pi])
def test_wigner_d_matches_risbo(beta):
    """The factorisation d = i^{m-n} sum_k Delta_km Delta_kn e^{-ik beta} (Eq. B25) vs the Risbo recursion."""
    assert ref.rel_err(utils.wigner_d(beta, L), ref.wigner_d(beta, L)) < ref.TOL


def test_delta_matrices_are_wigner_d_at_right_angle():
    delta = np.zeros((L, 2 * L - 1, 2 * L - 1))
    plane = np.zeros((2 * L - 1, 2 * L - 1))
    for el in range(L):
        plane = risbo.compute_full(plane, np.pi / 2, L, el)
        delta[el] = plane
    assert ref.rel_err(utils.delta_matrices(L), delta) < ref.TOL


def test_wigner_D_matches_definition(rng):  # noqa: N802
    """D^l_{mn} = e^{-im alpha} d^l_{mn}(beta) e^{-in gamma} (Eq. 6)."""
    angles = ref.random_rotation(rng)
    assert ref.rel_err(utils.wigner_D(*angles, L), ref.wigner_D(*angles, L)) < ref.TOL


def test_wigner_D_is_a_representation(rng):  # noqa: N802
    """D(rho_1 rho_2) = D(rho_1) D(rho_2), with the product's angles from the matrices."""
    rho1, rho2 = ref.random_rotation(rng), ref.random_rotation(rng)
    product = ref.euler_zyz(ref.rot_matrix(*rho1) @ ref.rot_matrix(*rho2))
    lhs = np.einsum("lmk,lkn->lmn", utils.wigner_D(*rho1, L), utils.wigner_D(*rho2, L))
    assert ref.rel_err(lhs, utils.wigner_D(*product, L)) < ref.TOL


# ------------------------------------------------------------------------------ rotations


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_rotate_flm_matches_wigner_matrix_product(rng, spin):
    """(R f)_{lm} = sum_n D^l_{mn}(rho) f_{ln} (Eq. 10) with D from the Risbo recursion."""
    flm = generate_flm(rng, L, spin=spin)
    angles = ref.random_rotation(rng)
    assert (
        ref.rel_err(utils.rotate_flm(flm, angles), ref.rotate_flm(flm, angles))
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1, 3])
def test_rotate_flm_matches_real_space_rotation(rng, spin):
    """(R f)(omega) = e^{-is vartheta} f(R^{-1} omega) (Eq. 12), vartheta the third angle of R^{-1} s(omega)."""
    flm = generate_flm(rng, L, spin=spin)
    angles = ref.random_rotation(rng)
    theta, phi = ref.random_points(rng, 40)
    rotated = ref.eval_spin(utils.rotate_flm(flm, angles), spin, theta, phi)
    assert (
        ref.rel_err(rotated, ref.rotated_values(flm, spin, angles, theta, phi))
        < ref.TOL
    )


def test_rotate_flmn_is_left_translation(rng):
    """On Wigner coefficients the rotation acts on the first index: (pi_G(g) F)(rho) = F(g^{-1} rho)."""
    N = 3
    flmn = generate_flmn(rng, L, N)
    g = ref.random_rotation(rng)
    rhos = np.array([ref.random_rotation(rng) for _ in range(8)]).T
    moved = ref.euler_zyz(
        np.swapaxes(ref.rot_matrix(*g), -1, -2) @ ref.rot_matrix(*rhos)
    )
    lhs = ref.eval_so3(np.asarray(utils.rotate_flmn(flmn, g)), *rhos)
    assert ref.rel_err(lhs, ref.eval_so3(flmn, *moved)) < ref.TOL


def test_rotations_compose(rng):
    """R(rho_1) R(rho_2) f = R(rho_1 rho_2) f."""
    flm = generate_flm(rng, L, spin=1)
    rho1, rho2 = ref.random_rotation(rng), ref.random_rotation(rng)
    product = ref.euler_zyz(ref.rot_matrix(*rho1) @ ref.rot_matrix(*rho2))
    lhs = utils.rotate_flm(utils.rotate_flm(flm, rho2), rho1)
    assert ref.rel_err(lhs, utils.rotate_flm(flm, product)) < ref.TOL


def test_rotation_is_unitary_and_inverted_by_reversed_angles(rng):
    flm = generate_flm(rng, L, spin=-2)
    alpha, beta, gamma = ref.random_rotation(rng)
    rotated = utils.rotate_flm(flm, (alpha, beta, gamma))
    assert abs(
        np.linalg.norm(rotated) - np.linalg.norm(flm)
    ) < ref.TOL * np.linalg.norm(flm)
    assert (
        ref.rel_err(utils.rotate_flm(rotated, (-gamma, -beta, -alpha)), flm) < ref.TOL
    )


def test_rotation_matrix_is_product_of_elementary_rotations(rng):
    angles = np.array([ref.random_rotation(rng) for _ in range(5)]).T
    R = np.asarray(utils.rotation_matrix(*angles))
    assert R.shape == (5, 3, 3)
    assert ref.rel_err(R, ref.rot_matrix(*angles)) < ref.TOL
    np.testing.assert_allclose(
        R @ np.swapaxes(R, -1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-14
    )
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)


def test_euler_angles_round_trip(rng):
    angles = np.array([ref.random_rotation(rng) for _ in range(20)]).T
    R = ref.rot_matrix(*angles)
    alpha, beta, gamma = (np.asarray(x) for x in utils.euler_angles(R))
    assert ref.rel_err(ref.rot_matrix(alpha, beta, gamma), R) < ref.TOL
    np.testing.assert_allclose(beta, angles[1], atol=1e-12)
    np.testing.assert_allclose(np.exp(1j * alpha), np.exp(1j * angles[0]), atol=1e-12)
    np.testing.assert_allclose(np.exp(1j * gamma), np.exp(1j * angles[2]), atol=1e-12)


def test_section_is_the_coordinate_frame(rng):
    """s(theta, phi) = R(phi, theta, 0) carries the north pole to the point (theta, phi) (Eq. 32)."""
    theta, phi = ref.random_points(rng, 6)
    S = np.asarray(utils.section(theta, phi))
    assert ref.rel_err(S, ref.rot_matrix(phi, theta, 0.0)) < ref.TOL
    np.testing.assert_allclose(S[..., :, 2], ref.unit_vector(theta, phi), atol=1e-14)


def test_random_rotation_is_haar_distributed():
    """alpha and gamma uniform on [0, 2 pi) and cos(beta) uniform on [-1, 1]."""
    keys = jax.random.split(jax.random.PRNGKey(2026), 4000)
    alpha, beta, gamma = (
        np.asarray(x) for x in jax.vmap(rotation.random_rotation)(keys)
    )
    assert np.all(
        (alpha >= 0) & (alpha < 2 * np.pi) & (gamma >= 0) & (gamma < 2 * np.pi)
    )
    assert np.all((beta >= 0) & (beta <= np.pi))
    z = np.cos(beta)
    # uniform on [-1, 1]: mean 0 and variance 1/3, within five standard errors
    assert abs(z.mean()) < 5 * np.sqrt(1 / 3 / z.size)
    assert abs(z.var() - 1 / 3) < 5 * np.sqrt(4 / 45 / z.size)
    assert abs(np.mean(np.exp(1j * alpha))) < 5 / np.sqrt(z.size)


# ---------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize("spin", [0, 1, -2, 3])
def test_evaluate_matches_s2fft_on_the_grid(rng, spin):
    flm = generate_flm(rng, L, spin=spin)
    thetas, phis = ref.sphere_grid(L)
    T, P = np.meshgrid(thetas, phis, indexing="ij")
    assert (
        ref.rel_err(utils.evaluate(flm, spin, T, P), ref.isht(flm, L, spin)) < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, -1, 4])
def test_evaluate_at_arbitrary_points(rng, spin):
    """Off the grid, against the definition of the spin spherical harmonics (Eq. A7)."""
    flm = generate_flm(rng, L, spin=spin)
    theta, phi = rng.uniform(0, np.pi, 50), rng.uniform(-np.pi, 3 * np.pi, 50)
    assert (
        ref.rel_err(
            utils.evaluate(flm, spin, theta, phi), ref.eval_spin(flm, spin, theta, phi)
        )
        < ref.TOL
    )


def test_evaluate_keeps_leading_axes(rng):
    flm = np.stack([generate_flm(rng, L, spin=2) for _ in range(3)])
    theta, phi = ref.random_points(rng, 7)
    values = utils.evaluate(flm, 2, theta, phi)
    assert values.shape == (3, 7)
    assert ref.rel_err(values, ref.eval_spin(flm, 2, theta, phi)) < ref.TOL


def test_evaluate_so3_matches_inverse_wigner_samples(rng):
    N = 4
    flmn = generate_flmn(rng, L, N)
    samples = ref.wigner_inv(flmn, L, N)
    thetas, phis = ref.sphere_grid(L)
    gammas = ref.gamma_grid(N)
    for j, t, p in ((0, 0, 0), (3, 5, 11), (6, 11, 22), (2, 7, 3)):
        value = utils.evaluate_so3(flmn, phis[p], thetas[t], gammas[j])
        assert abs(value - samples[j, t, p]) < ref.TOL * np.max(np.abs(samples))


def test_evaluate_so3_at_arbitrary_rotations(rng):
    N = 3
    flmn = generate_flmn(rng, L, N)
    rhos = [ref.random_rotation(rng) for _ in range(5)]
    values = [utils.evaluate_so3(flmn, *rho) for rho in rhos]
    assert ref.rel_err(values, [ref.eval_so3(flmn, *rho) for rho in rhos]) < ref.TOL


# -------------------------------------------------------------------------------- conjugate


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_conjugate_is_the_conjugate_field(rng, spin):
    """(sf^*)_{lm} = (-1)^{s+m} sf^*_{l,-m} (Eq. A9) are the coefficients of the spin -s field sf^*."""
    flm = generate_flm(rng, L, spin=spin)
    conj_lm = utils.conjugate(flm, spin)
    assert (
        ref.rel_err(ref.isht(conj_lm, L, -spin), np.conj(ref.isht(flm, L, spin)))
        < ref.TOL
    )
    assert ref.rel_err(utils.conjugate(conj_lm, -spin), flm) < ref.TOL


def test_conjugate_of_real_field_is_itself(rng):
    flm = generate_flm(rng, L, reality=True)
    assert ref.rel_err(utils.conjugate(flm, 0), flm) < ref.TOL
