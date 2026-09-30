"""
Spin-changing operators of mathematical physics (Proposition VI.8).

The spin raising and lowering operators are compared with fourth-order finite differences
of the differential operators of Eq. (3) at random points; the E/B decomposition and the
spherical Kaiser-Squires relation are checked through their round trips, reality and
their relations to eth by composition (Proposition V.3).
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm

from s2conv import kernels
from s2conv.convolutions import lifted, physics
from s2conv.utils import conjugate

L = 12
H = 1e-4


def _derivatives(flm, spin, theta, phi):
    """Values and fourth-order central differences in theta and phi of a spin field."""

    def values(t, p):
        return ref.eval_spin(flm, spin, t, p)

    d_theta = (
        -values(theta + 2 * H, phi)
        + 8 * values(theta + H, phi)
        - 8 * values(theta - H, phi)
        + values(theta - 2 * H, phi)
    ) / (12 * H)
    d_phi = (
        -values(theta, phi + 2 * H)
        + 8 * values(theta, phi + H)
        - 8 * values(theta, phi - H)
        + values(theta, phi - 2 * H)
    ) / (12 * H)
    return values(theta, phi), d_theta, d_phi


@pytest.mark.parametrize("spin", [0, 1, -2, 3])
def test_eth_and_ethbar_are_the_differential_operators(rng, spin):
    """eth sf = -(d_theta + i csc(theta) d_phi - s cot(theta)) sf and
    ethbar sf = -(d_theta - i csc(theta) d_phi + s cot(theta)) sf (Eq. 3, Proposition VI.8(a))."""
    flm = generate_flm(rng, L, spin=spin)
    theta, phi = ref.random_points(rng, 6, 0.9)
    f, d_theta, d_phi = _derivatives(flm, spin, theta, phi)
    eth_fd = -(d_theta + 1j / np.sin(theta) * d_phi - spin / np.tan(theta) * f)
    ethbar_fd = -(d_theta - 1j / np.sin(theta) * d_phi + spin / np.tan(theta) * f)
    eth = ref.eval_spin(physics.eth(flm, spin), spin + 1, theta, phi)
    ethbar = ref.eval_spin(physics.ethbar(flm, spin), spin - 1, theta, phi)
    assert ref.rel_err(eth, eth_fd) < 1e-9
    assert ref.rel_err(ethbar, ethbar_fd) < 1e-9


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_eth_is_a_lifted_convolution(rng, spin):
    """eth and ethbar are C_{s->s+-1}[Psi_T] with the canonical kernels of their multipliers."""
    flm = generate_flm(rng, L, spin=spin)
    for operator, multiplier, step in (
        (physics.eth, physics.eth_multiplier, 1),
        (physics.ethbar, physics.ethbar_multiplier, -1),
    ):
        kernel = kernels.multiplier(multiplier(L, spin), spin, spin + step)
        assert (
            ref.rel_err(
                operator(flm, spin), lifted.spin_to_spin(flm, kernel, spin, spin + step)
            )
            < ref.TOL
        )


@pytest.mark.parametrize("spin", [-2, 0, 1])
def test_eth_multipliers(spin):
    el = np.arange(L)
    valid = el >= max(abs(spin), abs(spin + 1))
    expected = np.sqrt((el - spin) * (el + spin + 1.0), where=valid, out=np.zeros(L))
    assert (
        ref.rel_err(physics.eth_multiplier(L, spin)[valid], expected[valid]) < ref.TOL
    )
    valid = el >= max(abs(spin), abs(spin - 1))
    expected = -np.sqrt((el + spin) * (el - spin + 1.0), where=valid, out=np.zeros(L))
    assert (
        ref.rel_err(physics.ethbar_multiplier(L, spin)[valid], expected[valid])
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_ethbar_eth_eigenvalues(rng, spin):
    """ethbar eth sf = -(l - s)(l + s + 1) sf, by composition of multipliers (Proposition V.3)."""
    flm = generate_flm(rng, L, spin=spin)
    el = np.arange(L)
    expected = (
        -((el - spin) * (el + spin + 1.0))[:, None]
        * flm
        * ref.lm_mask(L, max(abs(spin), abs(spin + 1)))
    )
    assert (
        ref.rel_err(physics.ethbar(physics.eth(flm, spin), spin + 1), expected)
        < ref.TOL
    )


def test_second_derivatives_of_spin_two():
    """ethbar^2 on spin 2 and eth^2 on spin -2 have c_l = sqrt((l+2)!/(l-2)!) (Proposition VI.8(b))."""
    el = np.arange(2, L)
    expected = np.sqrt((el - 1.0) * el * (el + 1) * (el + 2))
    down = physics.ethbar_multiplier(L, 2) * physics.ethbar_multiplier(L, 1)
    up = physics.eth_multiplier(L, -2) * physics.eth_multiplier(L, -1)
    assert ref.rel_err(down[2:], expected) < ref.TOL
    assert ref.rel_err(up[2:], expected) < ref.TOL


def _polarisation(rng):
    """Real E and B fields (l >= 2) and their spin +-2 coefficients a_{+-2} = -(a_E +- i a_B)."""
    e_lm = generate_flm(rng, L, L_lower=2, reality=True)
    b_lm = generate_flm(rng, L, L_lower=2, reality=True)
    return e_lm, b_lm, -(e_lm + 1j * b_lm), -(e_lm - 1j * b_lm)


def test_eb_decomposition(rng):
    """a_E = -(a_2 + a_{-2})/2 and a_B = i (a_2 - a_{-2})/2 (Proposition VI.8(b))."""
    e_lm, b_lm, p, p_conj = _polarisation(rng)
    # Q - iU is the conjugate of Q + iU for real E and B, so it may be left out
    assert ref.rel_err(ref.isht(p_conj, L, -2), np.conj(ref.isht(p, L, 2))) < ref.TOL
    for e, b in (physics.eb_decomposition(p, p_conj), physics.eb_decomposition(p)):
        assert ref.rel_err(e, e_lm) < ref.TOL
        assert ref.rel_err(b, b_lm) < ref.TOL
        assert np.max(np.abs(np.imag(ref.isht(e, L)))) < 1e-12
        assert np.max(np.abs(np.imag(ref.isht(b, L)))) < 1e-12


def test_eb_decomposition_from_samples(rng):
    """E and B from sampled Q and U fields, through the spin +-2 transforms."""
    e_lm, b_lm, p, _ = _polarisation(rng)
    samples = ref.isht(p, L, 2)
    q, u = samples.real, samples.imag
    e, b = physics.eb_decomposition(
        ref.sht(q + 1j * u, L, 2), ref.sht(q - 1j * u, L, -2)
    )
    assert ref.rel_err(e, e_lm) < ref.TOL
    assert ref.rel_err(b, b_lm) < ref.TOL


def test_eb_round_trip(rng):
    e_lm, b_lm, p, p_conj = _polarisation(rng)
    p_out, p_conj_out = physics.eb_to_polarisation(e_lm, b_lm)
    assert ref.rel_err(p_out, p) < ref.TOL
    assert ref.rel_err(p_conj_out, p_conj) < ref.TOL
    assert ref.rel_err(conjugate(p_out, 2), p_conj_out) < ref.TOL
    e, b = physics.eb_decomposition(p_out, p_conj_out)
    assert ref.rel_err(e, e_lm) < ref.TOL
    assert ref.rel_err(b, b_lm) < ref.TOL


def test_eb_of_a_complex_pair(rng):
    """With independent spin +-2 fields (complex E and B), the decomposition is still linear and exact."""
    p, p_conj = generate_flm(rng, L, spin=2), generate_flm(rng, L, spin=-2)
    e, b = physics.eb_decomposition(p, p_conj)
    assert ref.rel_err(e, -(p + p_conj) / 2) < ref.TOL
    assert ref.rel_err(b, 1j * (p - p_conj) / 2) < ref.TOL


def test_kaiser_squires_multiplier():
    """D_l = -sqrt((l+2)(l-1)/(l(l+1))) for l >= 2, the ratio of the multipliers of eth^2/2 and
    ethbar eth/2 (Propositions VI.8(c), V.3)."""
    el = np.arange(2, L).astype(float)
    d = physics.kaiser_squires_multiplier(L)
    assert np.all(d[:2] == 0)
    assert ref.rel_err(d[2:], -np.sqrt((el + 2) * (el - 1) / (el * (el + 1)))) < ref.TOL
    half_eth_squared = physics.eth_multiplier(L, 0) * physics.eth_multiplier(L, 1) / 2
    half_laplacian = physics.eth_multiplier(L, 0) * physics.ethbar_multiplier(L, 1) / 2
    assert ref.rel_err(d[2:], half_eth_squared[2:] / half_laplacian[2:]) < ref.TOL


def test_kaiser_squires_from_a_lensing_potential(rng):
    """With kappa = ethbar eth psi / 2 and gamma = eth eth psi / 2 for a lensing potential psi,
    gamma = C_{0->2}[D] kappa and kappa = C_{2->0}[D^{-1}] gamma for l >= 2."""
    psi = generate_flm(rng, L, reality=True)
    kappa = physics.ethbar(physics.eth(psi, 0), 1) / 2
    gamma = physics.eth(physics.eth(psi, 0), 1) / 2
    assert ref.rel_err(physics.kaiser_squires(kappa), gamma) < ref.TOL
    high = (np.arange(L) >= 2)[:, None]
    assert ref.rel_err(physics.kaiser_squires_inverse(gamma), kappa * high) < ref.TOL


def test_kaiser_squires_round_trip(rng):
    kappa = generate_flm(rng, L, reality=True)
    recovered = physics.kaiser_squires_inverse(physics.kaiser_squires(kappa))
    assert ref.rel_err(recovered, kappa * (np.arange(L) >= 2)[:, None]) < ref.TOL
    # the monopole and dipole are not recoverable and are returned as zero
    assert np.all(np.asarray(recovered)[:2] == 0)
