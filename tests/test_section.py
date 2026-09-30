"""
Evaluation of lifted convolutions on a section (Sec. VII A, Proposition VII.1).

The standard correlation is checked against the directional convolution at gamma = 0,
the expansion of Wiaux et al., the O(L^3) Fourier algorithm of Eqs. (B26)-(B27) and
quadrature of Definition III.7; the commutative anisotropic convolution against
quadrature of Definition III.8 and for commutativity; and section evaluation with a
constant twist against slices of the directional convolution.
"""

import numpy as np
import pytest
import reference as ref
from s2fft.sampling import s2_samples
from s2fft.utils.signal_generator import generate_flm

from s2conv import kernels, utils
from s2conv.convolutions import classical, lifted, section

L = 12
N = 4


def _low_orders(flm, n=N):
    """The orders |m| < n of a field: a kernel azimuthally band-limited at n."""
    return flm * (np.abs(ref.m_axis(flm.shape[-2])) < n)[None]


def _grid_points(rng, L_grid, n):
    """Random (t, p) indices of the MW grid, away from the poles."""
    return rng.integers(1, L_grid - 1, n), rng.integers(0, 2 * L_grid - 1, n)


# ------------------------------------------------------------------ standard correlation


def test_standard_correlation_is_the_directional_convolution_at_gamma_zero(rng):
    """(f (*) psi)(R_omega) = (f (*) psi)(phi, theta, 0) (Definition III.7, Proposition VII.1)."""
    flm, psi = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    directional = classical.directional_convolution(flm, psi, 0, N)
    expected = ref.wigner_inv(directional, L, N)[0]
    assert ref.rel_err(section.standard_correlation(flm, psi, N), expected) < ref.TOL


def test_standard_correlation_matches_wiaux(rng):
    """sum_n sum_lm (-1)^n sqrt(4 pi/(2l+1)) psi^*_{l,-n} f_lm nY_lm (Wiaux et al., Sec. VII A)."""
    flm, psi = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    weight = np.sqrt(ref.FOUR_PI / (2 * np.arange(L) + 1))
    expected = sum(
        ref.isht(
            (ref.parity(n) * weight * np.conj(psi[:, L - 1 - n]))[:, None]
            * flm
            * ref.lm_mask(L, n),
            L,
            n,
        )
        for n in range(-(N - 1), N)
    )
    assert ref.rel_err(section.standard_correlation(flm, psi, N), expected) < ref.TOL


def test_standard_correlation_fourier_on_the_grid(rng):
    """The O(L^3) algorithm of Eqs. (B26), (B27) evaluated on the grid."""
    flm, psi = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    h = section.standard_correlation_fourier(flm, psi)
    assert h.shape == (2 * L - 1, 2 * L - 1)
    thetas, phis = ref.sphere_grid(L)
    T, P = np.meshgrid(thetas, phis, indexing="ij")
    assert (
        ref.rel_err(
            utils.evaluate_fourier(h, T, P), section.standard_correlation(flm, psi, N)
        )
        < ref.TOL
    )


def test_standard_correlation_quadrature(rng):
    """<f, R_{(phi, theta, 0)} psi> by quadrature (Definition III.7), on the grid and, for a kernel
    with every order, through the Fourier algorithm at arbitrary points."""
    flm, psi = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    out = np.asarray(section.standard_correlation(flm, psi, N))
    thetas, phis = ref.sphere_grid(L)
    t, p = _grid_points(rng, L, 3)
    expected = [
        ref.inner_with_rotated(flm, psi, 0, (phis[j], thetas[i], 0.0))
        for i, j in zip(t, p, strict=True)
    ]
    assert ref.rel_err(out[t, p], expected) < ref.TOL
    psi_full = generate_flm(rng, L)
    theta, phi = ref.random_points(rng, 3, 0.9)
    values = utils.evaluate_fourier(
        section.standard_correlation_fourier(flm, psi_full), theta, phi
    )
    expected = [
        ref.inner_with_rotated(flm, psi_full, 0, (b, a, 0.0))
        for a, b in zip(theta, phi, strict=True)
    ]
    assert ref.rel_err(values, expected) < ref.TOL


def test_standard_correlation_of_a_zonal_kernel_is_the_axisymmetric_convolution(rng):
    flm, psi = generate_flm(rng, L), generate_flm(rng, L) * (ref.m_axis(L) == 0)[None]
    expected = ref.isht(classical.axisymmetric_convolution(flm, psi), L)
    assert ref.rel_err(section.standard_correlation(flm, psi, 1), expected) < ref.TOL


def test_standard_correlation_commutes_with_grid_polar_rotations(rng):
    """A polar rotation by k samples of longitude shifts the output by k samples (Sec. VII A)."""
    flm, psi = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    k = 5
    rotated = ref.rotate_flm(flm, (2 * np.pi * k / (2 * L - 1), 0.0, 0.0))
    out = np.asarray(section.standard_correlation(flm, psi, N))
    assert (
        ref.rel_err(
            section.standard_correlation(rotated, psi, N), np.roll(out, k, axis=1)
        )
        < ref.TOL
    )


# --------------------------------------------------------------------- section evaluation


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_constant_twist_is_a_slice_of_the_directional_convolution(rng, spin):
    """On the section s(x) R_z(chi), E_sigma[Lambda_s psi] f = (f (*) psi)(phi, theta, chi) (Eq. 63)."""
    flm, psi = (
        generate_flm(rng, L, spin=spin),
        _low_orders(generate_flm(rng, L, spin=spin)),
    )
    samples = ref.wigner_inv(classical.directional_convolution(flm, psi, spin, N), L, N)
    kernel = kernels.lifted(psi, spin)
    for j in (0, 2, 5):
        chi = ref.gamma_grid(N)[j]
        out = section.section_evaluation(flm, kernel, spin, N, chi)
        assert ref.rel_err(out, samples[j]) < ref.TOL


def test_twist_as_function_array_or_constant(rng):
    flm, psi = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    kernel = kernels.lifted(psi, 0)
    thetas, phis = ref.sphere_grid(L)
    T, P = np.meshgrid(thetas, phis, indexing="ij")

    def twist(theta, phi):
        return 0.7 * np.cos(theta) + 2 * phi

    from_function = section.section_evaluation(flm, kernel, 0, N, twist)
    from_array = section.section_evaluation(flm, kernel, 0, N, twist(T, P))
    assert ref.rel_err(from_array, from_function) < ref.TOL
    assert (
        ref.rel_err(
            section.section_evaluation(flm, kernel, 0, N, 0.0),
            section.section_evaluation(flm, kernel, 0, N),
        )
        < ref.TOL
    )
    # the evaluation is sum_{s'} e^{-is' chi} (2 pi)^{-1/2} C_{0->s'} f (Eq. 63)
    expected = sum(
        np.exp(-1j * sp * twist(T, P))
        * ref.isht(lifted.spin_to_spin(flm, kernel, 0, sp), L, sp)
        for sp in range(-(N - 1), N)
    ) / np.sqrt(2 * np.pi)
    assert ref.rel_err(from_function, expected) < ref.TOL


@pytest.mark.parametrize("sampling", ["mwss", "gl"])
def test_section_evaluation_in_other_samplings(rng, sampling):
    """The twist, given as a function, is evaluated on the grid of the chosen sampling."""
    flm, h = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    out = np.asarray(
        section.commutative_anisotropic_convolution(flm, h, N, sampling=sampling)
    )
    thetas, phis = s2_samples.thetas(L, sampling), s2_samples.phis_equiang(L, sampling)
    assert out.shape == (len(thetas), len(phis))
    t, p = rng.integers(1, len(thetas) - 1, 3), rng.integers(0, len(phis), 3)
    expected = [
        ref.inner_with_rotated(
            flm, h, 0, (phis[j], thetas[i], np.pi - phis[j]), bilinear=True
        )
        for i, j in zip(t, p, strict=True)
    ]
    assert ref.rel_err(out[t, p], expected) < ref.TOL


def test_section_components(rng):
    spin = 1
    flm = generate_flm(rng, L, spin=spin)
    kernel = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    components = section.section_components(flm, kernel, spin, N)
    assert components.shape == (2 * N - 1, L, 2 * L - 1)
    for sp in range(-(N - 1), N):
        expected = lifted.spin_to_spin(flm, kernel, spin, sp) / np.sqrt(2 * np.pi)
        assert ref.rel_err(components[N - 1 + sp], expected) < ref.TOL


@pytest.mark.parametrize("method", ["jax", "precompute"])
def test_section_evaluation_backends(rng, method):
    flm, psi = generate_flm(rng, L, spin=2), _low_orders(generate_flm(rng, L, spin=2))
    out = section.section_evaluation(
        flm, kernels.lifted(psi, 2), 2, N, 1.3, method=method
    )
    reference = section.section_evaluation(
        flm, kernels.lifted(psi, 2), 2, N, 1.3, method="numpy"
    )
    assert ref.rel_err(out, reference) < ref.TOL


# ------------------------------------------------------------ commutative anisotropic


def test_commutative_anisotropic_quadrature(rng):
    """(h (.)_S f)(theta, phi) = int (R_{(phi, theta, pi - phi)} h) f (Definition III.8), the bilinear
    pairing on the section chi = pi - phi (Proposition VII.1)."""
    flm, h = generate_flm(rng, L), _low_orders(generate_flm(rng, L))
    out = np.asarray(section.commutative_anisotropic_convolution(flm, h, N))
    thetas, phis = ref.sphere_grid(L)
    t, p = _grid_points(rng, L, 3)
    expected = [
        ref.inner_with_rotated(
            flm, h, 0, (phis[j], thetas[i], np.pi - phis[j]), bilinear=True
        )
        for i, j in zip(t, p, strict=True)
    ]
    assert ref.rel_err(out[t, p], expected) < ref.TOL


def test_commutative_anisotropic_is_commutative(rng):
    """h (.)_S f = f (.)_S h (Sadeghi et al.), for general fields at L = 8 with N = L."""
    L_small = 8
    flm, h = generate_flm(rng, L_small), generate_flm(rng, L_small)
    fh = section.commutative_anisotropic_convolution(flm, h, L_small)
    hf = section.commutative_anisotropic_convolution(h, flm, L_small)
    assert ref.rel_err(fh, hf) < ref.TOL
    thetas, phis = ref.sphere_grid(L_small)
    t, p = _grid_points(rng, L_small, 2)
    expected = [
        ref.inner_with_rotated(
            flm, h, 0, (phis[j], thetas[i], np.pi - phis[j]), bilinear=True
        )
        for i, j in zip(t, p, strict=True)
    ]
    assert ref.rel_err(np.asarray(fh)[t, p], expected) < ref.TOL
