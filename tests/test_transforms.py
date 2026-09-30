"""
Spin spherical harmonic and Wigner transforms of :mod:`s2conv.transforms`.

Round trips for every backend, agreement between backends and with s2fft's NumPy
transforms, the accuracy of the ``"precompute"`` backend at spins and azimuthal
band-limits beyond the recursion limit (against SSHT), and the guard that stops the
recursive backends there.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import reference as ref
import s2fft
from s2fft.sampling import s2_samples
from s2fft.utils.signal_generator import generate_flm, generate_flmn

from s2conv import transforms

L = 12

METHODS = [
    "jax",
    "precompute",
    "numpy",
    pytest.param("jax_ssht", marks=pytest.mark.ssht),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("spin", [0, 2, -3])
def test_spin_round_trip(rng, method, spin):
    flm = generate_flm(rng, L, spin=spin)
    f = transforms.inverse(flm, L, spin, method=method)
    assert ref.rel_err(transforms.forward(f, L, spin, method=method), flm) < ref.TOL


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("N", [1, 3])
def test_wigner_round_trip(rng, method, N):
    flmn = generate_flmn(rng, L, N)
    F = transforms.wigner_inverse(flmn, L, N, method=method)
    assert F.shape == (2 * N - 1, L, 2 * L - 1)
    assert (
        ref.rel_err(transforms.wigner_forward(F, L, N, method=method), flmn) < ref.TOL
    )


@pytest.mark.parametrize("method", METHODS)
def test_spin_backends_agree_with_s2fft_numpy(rng, method):
    """Every backend computes the same transforms as s2fft's NumPy reference."""
    spin = 1
    flm = generate_flm(rng, L, spin=spin)
    f = ref.isht(flm, L, spin)
    assert ref.rel_err(transforms.inverse(flm, L, spin, method=method), f) < ref.TOL
    assert ref.rel_err(transforms.forward(f, L, spin, method=method), flm) < ref.TOL


@pytest.mark.parametrize("method", METHODS)
def test_wigner_backends_agree_with_s2fft_numpy(rng, method):
    N = 4
    flmn = generate_flmn(rng, L, N)
    F = ref.wigner_inv(flmn, L, N)
    assert (
        ref.rel_err(transforms.wigner_inverse(flmn, L, N, method=method), F) < ref.TOL
    )
    assert (
        ref.rel_err(transforms.wigner_forward(F, L, N, method=method), flmn) < ref.TOL
    )


@pytest.mark.ssht
@pytest.mark.parametrize("spin", [9, -9])
def test_precompute_at_high_spin_matches_ssht(rng, spin):
    """The precompute backend is accurate beyond the recursion limit of the "jax" backend."""
    flm = generate_flm(rng, L, spin=spin)
    f_ssht = transforms.inverse(flm, L, spin, method="jax_ssht")
    assert (
        ref.rel_err(transforms.inverse(flm, L, spin, method="precompute"), f_ssht)
        < ref.TOL
    )
    assert (
        ref.rel_err(transforms.forward(f_ssht, L, spin, method="precompute"), flm)
        < ref.TOL
    )


@pytest.mark.ssht
def test_precompute_at_high_azimuthal_bandlimit_matches_ssht(rng):
    N = 9
    flmn = generate_flmn(rng, L, N)
    F_ssht = transforms.wigner_inverse(flmn, L, N, method="jax_ssht")
    assert (
        ref.rel_err(transforms.wigner_inverse(flmn, L, N, method="precompute"), F_ssht)
        < ref.TOL
    )
    assert (
        ref.rel_err(transforms.wigner_forward(F_ssht, L, N, method="precompute"), flmn)
        < ref.TOL
    )


@pytest.mark.parametrize("method", ["jax", "numpy"])
@pytest.mark.parametrize(
    "spin", [transforms.RECURSION_LIMIT, -transforms.RECURSION_LIMIT, 11]
)
def test_recursive_backends_refuse_high_spin(method, spin):
    flm = np.zeros((L, 2 * L - 1), np.complex128)
    with pytest.raises(ValueError, match="precompute"):
        transforms.inverse(flm, L, spin, method=method)
    with pytest.raises(ValueError, match="precompute"):
        transforms.forward(
            np.zeros((L, 2 * L - 1), np.complex128), L, spin, method=method
        )


@pytest.mark.parametrize("method", ["jax", "numpy"])
def test_recursive_backends_refuse_high_azimuthal_bandlimit(method):
    N = transforms.RECURSION_LIMIT
    with pytest.raises(ValueError, match="precompute"):
        transforms.wigner_inverse(
            np.zeros((2 * N - 1, L, 2 * L - 1), np.complex128), L, N, method=method
        )
    with pytest.raises(ValueError, match="precompute"):
        transforms.wigner_forward(
            np.zeros((2 * N - 1, L, 2 * L - 1), np.complex128), L, N, method=method
        )


def test_recursive_backend_accepts_spin_below_the_limit(rng):
    """The guard sits exactly at the limit: spin 7 and N = 7 are still accepted."""
    spin = N = transforms.RECURSION_LIMIT - 1
    flm = generate_flm(rng, L, spin=spin)
    assert (
        ref.rel_err(transforms.inverse(flm, L, spin), ref.isht(flm, L, spin)) < ref.TOL
    )
    flmn = generate_flmn(rng, L, N)
    assert (
        ref.rel_err(transforms.wigner_inverse(flmn, L, N), ref.wigner_inv(flmn, L, N))
        < ref.TOL
    )


def test_unknown_method_is_rejected():
    flm = np.zeros((L, 2 * L - 1), np.complex128)
    with pytest.raises(ValueError, match="not recognised"):
        transforms.inverse(flm, L, 0, method="fortran")


@pytest.mark.parametrize("sampling", ["mwss", "dh", "gl"])
@pytest.mark.parametrize("method", ["jax", "precompute"])
def test_round_trip_in_other_samplings(rng, sampling, method):
    spin = 2
    flm = generate_flm(rng, L, spin=spin)
    f = transforms.inverse(flm, L, spin, sampling=sampling, method=method)
    assert f.shape == s2_samples.f_shape(L, sampling)
    assert (
        ref.rel_err(
            transforms.forward(f, L, spin, sampling=sampling, method=method), flm
        )
        < ref.TOL
    )


@pytest.mark.parametrize("sampling", ["mwss", "gl"])
@pytest.mark.parametrize("method", ["jax", "precompute"])
def test_wigner_round_trip_at_unit_bandlimit_in_other_samplings(rng, sampling, method):
    """N = 1, a function of the first two Euler angles only (a lifted scalar field)."""
    flmn = generate_flmn(rng, L, 1)
    F = transforms.wigner_inverse(flmn, L, 1, sampling=sampling, method=method)
    assert (
        ref.rel_err(
            transforms.wigner_forward(F, L, 1, sampling=sampling, method=method), flmn
        )
        < ref.TOL
    )


@pytest.mark.parametrize("N", [1, 3])
def test_numpy_backend_accepts_jax_arrays(rng, N):
    """The NumPy backend takes the JAX arrays that the harmonic operators return."""
    flm = jnp.asarray(generate_flm(rng, L, spin=1))
    f = transforms.inverse(flm, L, 1, method="numpy")
    assert (
        ref.rel_err(transforms.forward(jnp.asarray(f), L, 1, method="numpy"), flm)
        < ref.TOL
    )
    flmn = jnp.asarray(generate_flmn(rng, L, N))
    F = transforms.wigner_inverse(flmn, L, N, method="numpy")
    assert (
        ref.rel_err(
            transforms.wigner_forward(jnp.asarray(F), L, N, method="numpy"), flmn
        )
        < ref.TOL
    )


@pytest.mark.parametrize("method", ["jax", "precompute"])
def test_reality_flag(rng, method):
    """A real scalar field is transformed identically with reality=True."""
    flm = generate_flm(rng, L, reality=True)
    f = transforms.inverse(flm, L, 0, method=method, reality=True)
    assert np.max(np.abs(np.imag(f))) < 1e-12
    assert ref.rel_err(np.real(f), np.real(ref.isht(flm, L))) < ref.TOL
    assert (
        ref.rel_err(
            transforms.forward(np.real(f), L, 0, method=method, reality=True), flm
        )
        < ref.TOL
    )


@pytest.mark.parametrize("sampling", ["mw", "mwss", "dh", "gl"])
def test_sphere_angles(sampling):
    thetas, phis = transforms.sphere_angles(L, sampling)
    np.testing.assert_allclose(thetas, s2_samples.thetas(L, sampling))
    np.testing.assert_allclose(phis, s2_samples.phis_equiang(L, sampling))


def test_sphere_angles_rejects_healpix():
    with pytest.raises(ValueError, match="equiangular"):
        transforms.sphere_angles(L, "healpix")


def test_precompute_kernels_are_cached():
    """The precompute kernels are built once per configuration."""
    assert transforms.spherical_kernel(L, 1) is transforms.spherical_kernel(L, 1)
    assert transforms.wigner_kernel(L, 2) is transforms.wigner_kernel(L, 2)


def test_package_transforms_are_s2fft_transforms(rng):
    """The "jax" backend is s2fft's JAX transform, with s2fft's conventions."""
    flm = generate_flm(rng, L, spin=-1)
    expected = s2fft.inverse(flm, L, spin=-1, sampling="mw", method="jax")
    assert ref.rel_err(transforms.inverse(flm, L, -1), expected) < ref.TOL
