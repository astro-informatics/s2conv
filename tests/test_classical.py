"""
The spherical convolutions of the literature (Secs. III and VI, Tables I and II).

Each convolution of :mod:`s2conv.convolutions.classical` is checked three ways: against
its harmonic form, against exact real-space quadrature of its definition, with the kernel
rotated in real space and the spin phase of Eq. (12), and against the generalised lifted
convolution with the kernel and constant of Table II ((2 pi)^{-1/2} for the axisymmetric
convolutions, none for the directional and left convolutions).
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm, generate_flmn

from s2conv import kernels
from s2conv.convolutions import classical, lifted

L = 12


def _zonal(flm):
    """The axisymmetric (zonal) part of a field, its orders m = 0."""
    return flm * (ref.m_axis(flm.shape[-2]) == 0)[None]


# ------------------------------------------------------------------ axisymmetric convolution


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_axisymmetric_harmonic_form(rng, spin):
    """(f (.) psi)_{lm} = sqrt(4 pi/(2l+1)) sf_{lm} spsi^*_{l0} (Eqs. 14, 25)."""
    flm, psi = generate_flm(rng, L, spin=spin), generate_flm(rng, L, spin=spin)
    weight = np.sqrt(ref.FOUR_PI / (2 * np.arange(L) + 1)) * np.conj(psi[:, L - 1])
    expected = weight[:, None] * flm * ref.lm_mask(L, spin)
    assert (
        ref.rel_err(classical.axisymmetric_convolution(flm, psi, spin), expected)
        < ref.TOL
    )
    # only the axisymmetric part of the kernel enters
    assert (
        ref.rel_err(
            classical.axisymmetric_convolution(flm, _zonal(psi), spin), expected
        )
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_axisymmetric_quadrature(rng, spin):
    """(f (.) psi)(omega) = <f, R_omega psi> with R_omega = R(phi, theta, 0) (Defs III.1, III.6)."""
    flm, psi = generate_flm(rng, L, spin=spin), _zonal(generate_flm(rng, L, spin=spin))
    theta, phi = ref.random_points(rng, 3, 0.9)
    out = ref.eval_spin(
        classical.axisymmetric_convolution(flm, psi, spin), 0, theta, phi
    )
    expected = [
        ref.inner_with_rotated(flm, psi, spin, (p, t, 0.0))
        for t, p in zip(theta, phi, strict=True)
    ]
    assert ref.rel_err(out, expected) < ref.TOL


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_axisymmetric_is_a_lifted_convolution(rng, spin):
    """f (.) psi = (2 pi)^{-1/2} C_{s->0}[Lambda_s psi] f (Proposition VI.3)."""
    flm, psi = generate_flm(rng, L, spin=spin), _zonal(generate_flm(rng, L, spin=spin))
    lifted_form = lifted.spin_to_spin(
        flm, kernels.lifted(psi, spin), spin, 0
    ) / np.sqrt(2 * np.pi)
    assert (
        ref.rel_err(classical.axisymmetric_convolution(flm, psi, spin), lifted_form)
        < ref.TOL
    )


# -------------------------------------------------------------------------- left convolution


def test_left_harmonic_form(rng):
    """(f *_L h)_{lm} = 2 pi sqrt(4 pi/(2l+1)) f_{lm} h_{l0} (Eq. 16): only the zonal part of h enters."""
    flm, h = generate_flm(rng, L), generate_flm(rng, L)
    weight = 2 * np.pi * np.sqrt(ref.FOUR_PI / (2 * np.arange(L) + 1)) * h[:, L - 1]
    assert (
        ref.rel_err(classical.left_convolution(flm, h), weight[:, None] * flm) < ref.TOL
    )
    assert (
        ref.rel_err(classical.left_convolution(flm, _zonal(h)), weight[:, None] * flm)
        < ref.TOL
    )


def test_left_quadrature(rng):
    """(f *_L h)(omega) = int_SO(3) f(rho eta) h(R_rho^{-1} omega) d mu(rho) (Definition III.2)."""
    flm, h = generate_flm(rng, L), generate_flm(rng, L)
    theta, phi = ref.random_points(rng, 3, 0.9)
    out = ref.eval_spin(classical.left_convolution(flm, h), 0, theta, phi)
    assert (
        ref.rel_err(out, ref.left_convolution_quadrature(flm, h, theta, phi)) < ref.TOL
    )


def test_left_is_a_lifted_convolution(rng):
    """f *_L h = C_{0->0}[Psi_h] f, with no residual constant (Proposition VI.4)."""
    flm, h = generate_flm(rng, L), generate_flm(rng, L)
    lifted_form = lifted.spin_to_spin(flm, kernels.left(h), 0, 0)
    assert ref.rel_err(classical.left_convolution(flm, h), lifted_form) < ref.TOL


# ------------------------------------------------------------------- directional convolution


def _directional_expected(flm, psi, spin, N):
    """8 pi^2/(2l+1) sf_{lm} spsi^*_{ln} for |n| < N (Eqs. 18, 23)."""
    weight = ref.EIGHT_PI2 / (2 * np.arange(L) + 1)
    out = np.zeros((2 * N - 1, L, 2 * L - 1), np.complex128)
    for n in range(-(N - 1), N):
        column = weight * np.conj(psi[:, L - 1 + n])
        out[N - 1 + n] = column[:, None] * flm * ref.lm_mask(L, max(abs(spin), abs(n)))
    return out


@pytest.mark.parametrize("spin", [0, 2, -1])
@pytest.mark.parametrize("N", [None, 3])
def test_directional_harmonic_form(rng, spin, N):
    flm, psi = generate_flm(rng, L, spin=spin), generate_flm(rng, L, spin=spin)
    out = classical.directional_convolution(flm, psi, spin, N)
    assert (
        ref.rel_err(out, _directional_expected(flm, psi, spin, L if N is None else N))
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_directional_quadrature(rng, spin):
    """(f (*) psi)(rho) = <f, R_rho psi> at random rotations (Defs III.3, III.5)."""
    flm, psi = generate_flm(rng, L, spin=spin), generate_flm(rng, L, spin=spin)
    out = np.asarray(classical.directional_convolution(flm, psi, spin))
    rotations = [ref.random_rotation(rng) for _ in range(3)]
    values = [ref.eval_so3(out, *rho) for rho in rotations]
    assert (
        ref.rel_err(
            values, [ref.inner_with_rotated(flm, psi, spin, rho) for rho in rotations]
        )
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_directional_is_a_lifted_convolution(rng, spin):
    """f (*) psi = C_{s->G}[Lambda_s psi] f exactly, with no residual constant (Proposition VI.1)."""
    flm, psi = generate_flm(rng, L, spin=spin), generate_flm(rng, L, spin=spin)
    lifted_form = lifted.spin_to_group(flm, kernels.lifted(psi, spin), spin)
    assert (
        ref.rel_err(classical.directional_convolution(flm, psi, spin), lifted_form)
        < ref.TOL
    )


# -------------------------------------------------------------------------------- synthesis


@pytest.mark.parametrize("spin", [0, 2])
def test_directional_synthesis_harmonic_form(rng, spin):
    """(W (*)^dagger psi)_{lm} = sum_n W^l_{mn} spsi_{ln} (Eq. 31)."""
    N = 5
    flmn, psi = generate_flmn(rng, L, N), generate_flm(rng, L, spin=spin)
    expected = sum(
        flmn[N - 1 + n] * psi[:, L - 1 + n][:, None] for n in range(-(N - 1), N)
    )
    out = classical.directional_synthesis(flmn, psi, spin)
    assert ref.rel_err(out, expected * ref.lm_mask(L, spin)) < ref.TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_directional_synthesis_is_the_adjoint(rng, spin):
    """<f (*) psi, W>_SO(3) = <f, W (*)^dagger psi> and synthesis is C_{G->s}[flipped Lambda_s psi] (Prop. VI.2)."""
    flm, psi, flmn = (
        generate_flm(rng, L, spin=spin),
        generate_flm(rng, L, spin=spin),
        generate_flmn(rng, L, L),
    )
    synthesis = classical.directional_synthesis(flmn, psi, spin)
    lhs = ref.inner_so3(classical.directional_convolution(flm, psi, spin), flmn)
    assert abs(lhs - ref.inner_s2(flm, synthesis)) < ref.TOL * abs(lhs)
    lifted_form = lifted.group_to_spin(
        flmn, kernels.flipped(kernels.lifted(psi, spin)), spin
    )
    assert ref.rel_err(synthesis, lifted_form) < ref.TOL


@pytest.mark.parametrize("spin", [0, 1])
def test_directional_synthesis_quadrature(rng, spin):
    """(W (*)^dagger psi)(omega) = int_SO(3) W(rho) (R_rho psi)(omega) d mu(rho) (Definition III.10)."""
    L_small, N = 8, 3
    flmn, psi = generate_flmn(rng, L_small, N), generate_flm(rng, L_small, spin=spin)
    theta, phi = ref.random_points(rng, 2, 0.9)
    out = ref.eval_spin(
        classical.directional_synthesis(flmn, psi, spin), spin, theta, phi
    )
    assert (
        ref.rel_err(out, ref.synthesis_quadrature(flmn, psi, spin, theta, phi))
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_axisymmetric_synthesis(rng, spin):
    """sqrt(4 pi/(2l+1)) W_{lm} spsi_{l0}: the adjoint of the axisymmetric convolution, and
    (2 pi)^{-1/2} C_{0->s}[flipped Lambda_s psi] (Eq. 26, Props VI.2, VI.3)."""
    wlm, psi, flm = (
        generate_flm(rng, L),
        _zonal(generate_flm(rng, L, spin=spin)),
        generate_flm(rng, L, spin=spin),
    )
    weight = np.sqrt(ref.FOUR_PI / (2 * np.arange(L) + 1)) * psi[:, L - 1]
    out = classical.axisymmetric_synthesis(wlm, psi, spin)
    assert ref.rel_err(out, weight[:, None] * wlm * ref.lm_mask(L, spin)) < ref.TOL
    lhs = ref.inner_s2(classical.axisymmetric_convolution(flm, psi, spin), wlm)
    assert abs(lhs - ref.inner_s2(flm, out)) < ref.TOL * abs(lhs)
    lifted_form = lifted.spin_to_spin(
        wlm, kernels.flipped(kernels.lifted(psi, spin)), 0, spin
    ) / np.sqrt(2 * np.pi)
    assert ref.rel_err(out, lifted_form) < ref.TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_axisymmetric_synthesis_quadrature(rng, spin):
    """int_S2 W(omega') (R_{omega'} psi)(omega) d omega' (the scaling term of Eq. 26)."""
    wlm, psi = generate_flm(rng, L), _zonal(generate_flm(rng, L, spin=spin))
    theta, phi = ref.random_points(rng, 3, 0.9)
    out = ref.eval_spin(
        classical.axisymmetric_synthesis(wlm, psi, spin), spin, theta, phi
    )
    assert (
        ref.rel_err(
            out, ref.axisymmetric_synthesis_quadrature(wlm, psi, spin, theta, phi)
        )
        < ref.TOL
    )


def _directional_wavelets(s, ell0=2.0, n_scales=4):
    """A spin-s scaling function and steerable wavelets with orders n = 0, +-2 from telescoping
    Gaussian windows kappa^2 + sum_j zeta_j^2 = 1, admissible for l >= |s| (Eq. 27)."""
    el = np.arange(L)
    windows = [np.exp(-((el / (ell0 * 2.0**j)) ** 2)) for j in range(n_scales)] + [
        np.ones(L)
    ]
    scaling = np.zeros((L, 2 * L - 1), np.complex128)
    scaling[:, L - 1] = np.sqrt((2 * el + 1) / ref.FOUR_PI) * np.sqrt(windows[0])
    eta = {-2: 1 / np.sqrt(6), 0: 2 / np.sqrt(6), 2: 1 / np.sqrt(6)}
    wavelets = []
    for j in range(n_scales):
        zeta = np.sqrt(np.clip(windows[j + 1] - windows[j], 0, None))
        psi = np.zeros((L, 2 * L - 1), np.complex128)
        for n, weight in eta.items():
            psi[:, L - 1 + n] = np.sqrt((2 * el + 1) / ref.EIGHT_PI2) * zeta * weight
        wavelets.append(psi * ref.lm_mask(L, s))
    return scaling * ref.lm_mask(L, s), wavelets


def test_directional_spin_wavelet_frame(rng):
    """Scaling and directional wavelet coefficients (Eqs. 23, 25) of a spin-2 field, synthesised by
    Eq. (26), reconstruct it: the admissibility condition (27) makes a Parseval frame."""
    s = 2
    scaling, wavelets = _directional_wavelets(s)
    el = np.arange(L)
    admissibility = ref.FOUR_PI / (2 * el + 1) * np.abs(
        scaling[:, L - 1]
    ) ** 2 + ref.EIGHT_PI2 / (2 * el + 1) * sum(
        np.sum(np.abs(psi) ** 2, axis=1) for psi in wavelets
    )
    assert ref.rel_err(admissibility[s:], np.ones(L - s)) < ref.TOL
    flm = generate_flm(rng, L, spin=s)
    w_scaling = classical.axisymmetric_convolution(flm, scaling, s)
    w_wavelets = [
        classical.directional_convolution(flm, psi, s, N=3) for psi in wavelets
    ]
    reconstruction = classical.axisymmetric_synthesis(w_scaling, scaling, s) + sum(
        classical.directional_synthesis(w, psi, s)
        for w, psi in zip(w_wavelets, wavelets, strict=True)
    )
    assert ref.rel_err(reconstruction, flm) < ref.TOL


def test_spin_changing_convolution_is_a_multiplier(rng):
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    flm = generate_flm(rng, L, spin=3)
    out = classical.spin_changing_convolution(flm, c, 3, -1)
    assert ref.rel_err(out, ref.apply_multiplier(c, flm, 3, -1)) < ref.TOL
