"""
Spin-changing convolutions (Proposition VI.5, Theorem V.5, Corollaries V.7 and Prop. VI.6).

A spin s -> s' multiplier applied with its canonical kernel is compared with exact
quadrature of the radial two-point kernel with its frame phases (Eq. 55 / A10), in the
canonical angles and in the literal angles of Tessore et al.; the realisations of the
spin-weighted convolutions of Esteves et al. (Eqs. 13, 15) and of the spin needlets are
checked, and steerability: the directional output is the sum of lifted spin components.
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm

from s2conv import kernels, lifting
from s2conv.convolutions import classical, lifted

L = 12

SPIN_PAIRS = [(2, 1), (0, 2), (-1, 2), (1, -1), (0, 0), (-3, 0)]


def _multiplier_output(rng, s_in, s_out):
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    flm = generate_flm(rng, L, spin=s_in)
    return (
        c,
        flm,
        lifted.spin_to_spin(flm, kernels.multiplier(c, s_in, s_out), s_in, s_out),
    )


@pytest.mark.parametrize(("s_in", "s_out"), SPIN_PAIRS)
def test_multiplier_is_the_radial_two_point_kernel(rng, s_in, s_out):
    """(T f)(x) = int e^{is'a} K(b) e^{isc} f(y) dy, s(x)^{-1} s(y) = R_z(a) R_y(b) R_z(c),
    K(b) = sum_l (2l+1)/(4 pi) c_l d^l_{s's}(b) (Corollary V.7, Eq. A10)."""
    c, flm, out = _multiplier_output(rng, s_in, s_out)
    assert ref.rel_err(out, ref.apply_multiplier(c, flm, s_in, s_out)) < ref.TOL
    theta, phi = ref.random_points(rng, 3, 0.9)
    values = ref.eval_spin(out, s_out, theta, phi)
    assert (
        ref.rel_err(values, ref.two_point_quadrature(c, s_in, s_out, flm, theta, phi))
        < ref.TOL
    )


@pytest.mark.parametrize(("s_in", "s_out"), [(2, 1), (0, 2), (-1, -3)])
def test_tessore_literal_angles_give_a_sign(rng, s_in, s_out):
    """The printed angle a + pi of Tessore et al. gives (-1)^{s'} times the canonical result (Prop. VI.5(c))."""
    c, flm, out = _multiplier_output(rng, s_in, s_out)
    theta, phi = ref.random_points(rng, 3, 0.9)
    literal = ref.two_point_quadrature(
        c, s_in, s_out, flm, theta, phi, literal_angles=True
    )
    assert (
        ref.rel_err(literal, ref.parity(s_out) * ref.eval_spin(out, s_out, theta, phi))
        < ref.TOL
    )


@pytest.mark.parametrize(("s_in", "s_out"), SPIN_PAIRS)
def test_esteves_equation_13(rng, s_in, s_out):
    """g_{lm} = k_l f_{lm} (Esteves et al., Eq. 13) as the canonical kernel and as a lifted spherical
    kernel with spsi_{l,-s'} = (-1)^{s'} sqrt((2l+1)/(8 pi^2)) k^*_l (Prop. VI.5(a))."""
    k = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    flm = generate_flm(rng, L, spin=s_in)
    expected = ref.apply_multiplier(k, flm, s_in, s_out)
    assert (
        ref.rel_err(
            lifted.spin_to_spin(flm, kernels.multiplier(k, s_in, s_out), s_in, s_out),
            expected,
        )
        < ref.TOL
    )
    psi = np.zeros((L, 2 * L - 1), np.complex128)
    psi[:, L - 1 - s_out] = (
        ref.parity(s_out) * np.sqrt((2 * np.arange(L) + 1) / ref.EIGHT_PI2) * np.conj(k)
    )
    psi *= ref.lm_mask(L, s_in)
    assert (
        ref.rel_err(
            lifted.spin_to_spin(flm, kernels.lifted(psi, s_in), s_in, s_out), expected
        )
        < ref.TOL
    )


@pytest.mark.parametrize(("s_in", "s_out"), [(2, 1), (0, 2), (-1, 0), (1, -1)])
def test_esteves_equation_15(rng, s_in, s_out):
    """g^{s'}_{lm} = f^s_{lm} k^{s*}_{ls'} (Esteves et al., Eq. 15) through the reparameterised kernel
    k'_{ln} = (-1)^n sqrt((2l+1)/(8 pi^2)) k_{l,-n}; lifting the unmodified kernel gives another operator."""
    k = generate_flm(rng, L, spin=s_in)
    flm = generate_flm(rng, L, spin=s_in)
    expected = ref.apply_multiplier(np.conj(k[:, L - 1 + s_out]), flm, s_in, s_out)
    signs = np.array([ref.parity(n) for n in ref.m_axis(L)])
    k_prime = (
        signs[None, :]
        * np.sqrt((2 * np.arange(L) + 1) / ref.EIGHT_PI2)[:, None]
        * k[:, ::-1]
    )
    k_prime *= ref.lm_mask(L, s_in)
    assert (
        ref.rel_err(
            lifted.spin_to_spin(flm, kernels.lifted(k_prime, s_in), s_in, s_out),
            expected,
        )
        < ref.TOL
    )
    assert (
        ref.rel_err(
            lifted.spin_to_spin(flm, kernels.lifted(k, s_in), s_in, s_out), expected
        )
        > 1e-2
    )


def _needlet_multipliers(spin):
    """c_{jl} = b(sqrt(e_ls) / B^j), e_ls = (l - |s|)(l + |s| + 1), with sum_j c_{jl}^2 = 1 for l > |s|."""
    el = np.arange(L)
    eig = np.sqrt(np.clip((el - abs(spin)) * (el + abs(spin) + 1.0), 0, None))
    return [ref.needlet_window(eig / 2.0**j) for j in range(6)]


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_spin_needlets_reconstruct(rng, spin):
    """sum_j C_{s->s}[b_j] C_{s->s}[b_j] = Id and sum_j C_{0->s}[b_j] C_{s->0}[b_j] = Id for l > |s| (Prop. VI.5(b))."""
    flm = generate_flm(rng, L, spin=spin)
    expected = flm * (np.arange(L) > abs(spin))[:, None]
    needlets = _needlet_multipliers(spin)
    same = sum(
        lifted.spin_to_spin(
            lifted.spin_to_spin(flm, kernels.multiplier(b, spin, spin), spin, spin),
            kernels.multiplier(b, spin, spin),
            spin,
            spin,
        )
        for b in needlets
    )
    assert ref.rel_err(same, expected) < ref.TOL
    mixed = sum(
        lifted.spin_to_spin(
            lifted.spin_to_spin(flm, kernels.multiplier(b, spin, 0), spin, 0),
            kernels.multiplier(b, 0, spin),
            0,
            spin,
        )
        for b in needlets
    )
    assert ref.rel_err(mixed, expected) < ref.TOL
    # the same by composing kernels (Proposition V.3)
    composed = sum(
        kernels.compose(
            kernels.multiplier(b, spin, spin), kernels.multiplier(b, spin, spin)
        )
        for b in needlets
    )
    c = ref.multiplier_of(ref.kernel_to_block(composed), spin, spin)
    assert ref.rel_err(c, 1.0 * (np.arange(L) > abs(spin))) < ref.TOL


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_steerability(rng, spin):
    """For a kernel with orders |n| < N, C_{s->G}[Lambda_s psi] = sum_{|s'| < N} Lambda_{s'} C_{s->s'}[Lambda_s psi]
    (Proposition VI.6), in harmonic space and on the samples of SO(3)."""
    N = 4
    psi = generate_flm(rng, L, spin=spin) * (np.abs(ref.m_axis(L)) < N)[None]
    flm = generate_flm(rng, L, spin=spin)
    kernel = kernels.lifted(psi, spin)
    directional = classical.directional_convolution(flm, psi, spin, N)
    components = {
        sp: lifted.spin_to_spin(flm, kernel, spin, sp) for sp in range(-(N - 1), N)
    }
    assembled = sum(ref.lift(np.asarray(c), sp, N) for sp, c in components.items())
    assert ref.rel_err(directional, assembled) < ref.TOL
    assert (
        ref.rel_err(
            lifting.spin_components(directional), np.stack(list(components.values()))
        )
        < ref.TOL
    )
    steered = np.array(
        [
            sum(
                np.exp(-1j * sp * gamma) * ref.isht(c, L, sp)
                for sp, c in components.items()
            )
            / np.sqrt(2 * np.pi)
            for gamma in ref.gamma_grid(N)
        ]
    )
    assert ref.rel_err(steered, ref.wigner_inv(directional, L, N)) < ref.TOL
