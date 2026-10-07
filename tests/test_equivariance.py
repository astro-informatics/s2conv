"""
Rotation equivariance in real space (Theorem V.2, Sec. VII).

Each operator T is applied to a rotated field and compared with the rotated output,
T(R f)(omega) against e^{-is' vartheta} (T f)(R^{-1} omega) at random points, including the
spin phase of Eq. (12), and outputs on the rotation group through F(g^{-1} rho).  Every
lifted convolution is equivariant under Haar-random rotations; the section evaluations
and diagonal operators of Sec. VII are equivariant under polar rotations only, except
for axisymmetric kernels, and the commutative anisotropic convolution keeps the half turn
for kernels with even orders only.
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm, generate_flmn

from s2conv import kernels
from s2conv.convolutions import classical, diagonal, lifted, physics, section
from s2conv.utils import conjugate

L = 10
EQUIVARIANT = 1e-12  # relative error of an equivariant operator
NOT_EQUIVARIANT = (
    1e-2  # an operator that is not equivariant misses by far more than this
)

POLAR = (1.1, 0.0, 0.0)
HALF_TURN = (np.pi, 0.0, 0.0)


def _on_sphere(operator, s_out):
    """f -> T f as a function of (theta, phi), evaluated with the reference (Eq. A7)."""
    return lambda f: (
        lambda theta, phi: ref.eval_spin(np.asarray(operator(f)), s_out, theta, phi)
    )


def _full_kernel(rng):
    """A random kernel with every Wigner coefficient Psi^l_{nk}, |n|, |k| <= l, populated."""
    return ref.block_to_kernel(ref.random_block(L, rng), L)


def _zonal(flm):
    """The axisymmetric (zonal) part of a field, its orders m = 0."""
    return flm * (ref.m_axis(L) == 0)[None]


# ------------------------------------------------------------- lifted convolutions (Thm V.2)


@pytest.mark.parametrize(("s_in", "s_out"), [(0, 0), (2, 1), (-1, 2), (3, -3)])
def test_spin_to_spin_is_equivariant(rng, s_in, s_out):
    kernel = _full_kernel(rng)
    apply = _on_sphere(lambda f: lifted.spin_to_spin(f, kernel, s_in, s_out), s_out)
    error = ref.equivariance_error(
        apply, generate_flm(rng, L, spin=s_in), s_out, ref.random_rotation(rng), rng
    )
    assert error < EQUIVARIANT


@pytest.mark.parametrize("s_in", [0, 2])
def test_spin_to_group_is_equivariant(rng, s_in):
    kernel = _full_kernel(rng)
    x = generate_flm(rng, L, spin=s_in)
    error = ref.equivariance_error_group(
        lambda f: np.asarray(lifted.spin_to_group(f, kernel, s_in)),
        x,
        ref.random_rotation(rng),
        rng,
    )
    assert error < EQUIVARIANT


def test_group_to_group_is_equivariant(rng):
    kernel = _full_kernel(rng)
    x = generate_flmn(rng, L, L)
    error = ref.equivariance_error_group(
        lambda F: np.asarray(lifted.group_to_group(F, kernel)),
        x,
        ref.random_rotation(rng),
        rng,
        group_input=True,
    )
    assert error < EQUIVARIANT


@pytest.mark.parametrize("s_out", [0, -2])
def test_group_to_spin_is_equivariant(rng, s_out):
    """The input on the group is rotated by left translation, the output compared on the sphere."""
    kernel = _full_kernel(rng)
    flmn = generate_flmn(rng, L, L)
    rotation = ref.random_rotation(rng)
    theta, phi = ref.random_points(rng, 40)
    after = ref.eval_spin(
        np.asarray(
            lifted.group_to_spin(ref.rotate_flmn(flmn, rotation), kernel, s_out)
        ),
        s_out,
        theta,
        phi,
    )
    a, b, c = ref.euler_zyz(
        np.swapaxes(ref.rot_matrix(*rotation), -1, -2) @ ref.section(theta, phi)
    )
    before = np.exp(-1j * s_out * c) * ref.eval_spin(
        np.asarray(lifted.group_to_spin(flmn, kernel, s_out)), s_out, b, a
    )
    assert ref.rel_err(after, before) < EQUIVARIANT


# ------------------------------------------------------------------ classical convolutions


# Each case: (the map, given a random kernel psi of the input's spin, input spin, output spin).
SPHERE_OPERATORS = {
    "axisymmetric": (
        lambda f, psi: classical.axisymmetric_convolution(f, _zonal(psi)),
        0,
        0,
    ),
    "left": (lambda f, psi: classical.left_convolution(f, psi), 0, 0),
    "spin axisymmetric": (
        lambda f, psi: classical.axisymmetric_convolution(f, _zonal(psi), 2),
        2,
        0,
    ),
    "axisymmetric synthesis": (
        lambda w, psi: classical.axisymmetric_synthesis(w, _zonal(psi), 2),
        0,
        2,
    ),
    "eth": (lambda f, psi: physics.eth(f, 2), 2, 3),
    "ethbar": (lambda f, psi: physics.ethbar(f, 2), 2, 1),
    "Kaiser-Squires": (lambda f, psi: physics.kaiser_squires(f), 0, 2),
    "E from Q +- iU": (lambda p, psi: physics.eb_decomposition(p)[0], 2, 0),
    "B from Q +- iU": (lambda p, psi: physics.eb_decomposition(p)[1], 2, 0),
}


@pytest.mark.parametrize("name", SPHERE_OPERATORS)
def test_classical_convolutions_on_the_sphere_are_equivariant(rng, name):
    """The convolutions of Table II and the operators of Proposition VI.8 with output on the sphere."""
    operator, s_in, s_out = SPHERE_OPERATORS[name]
    # the synthesis maps a scalar field to spin 2 with a spin-2 kernel
    psi = generate_flm(rng, L, spin=2 if name == "axisymmetric synthesis" else s_in)
    apply = _on_sphere(lambda f: operator(f, psi), s_out)
    error = ref.equivariance_error(
        apply, generate_flm(rng, L, spin=s_in), s_out, ref.random_rotation(rng), rng
    )
    assert error < EQUIVARIANT


@pytest.mark.parametrize("spin", [0, 2])
def test_directional_convolution_is_equivariant(rng, spin):
    psi = generate_flm(rng, L, spin=spin)
    x = generate_flm(rng, L, spin=spin)
    error = ref.equivariance_error_group(
        lambda f: np.asarray(classical.directional_convolution(f, psi, spin)),
        x,
        ref.random_rotation(rng),
        rng,
    )
    assert error < EQUIVARIANT


def test_haar_synthesis_is_equivariant(rng):
    psi = generate_flm(rng, L, spin=2)
    flmn = generate_flmn(rng, L, L)
    rotation = ref.random_rotation(rng)
    theta, phi = ref.random_points(rng, 40)
    after = ref.eval_spin(
        np.asarray(
            classical.directional_synthesis(ref.rotate_flmn(flmn, rotation), psi, 2)
        ),
        2,
        theta,
        phi,
    )
    a, b, c = ref.euler_zyz(
        np.swapaxes(ref.rot_matrix(*rotation), -1, -2) @ ref.section(theta, phi)
    )
    before = np.exp(-2j * c) * ref.eval_spin(
        np.asarray(classical.directional_synthesis(flmn, psi, 2)), 2, b, a
    )
    assert ref.rel_err(after, before) < EQUIVARIANT


# -------------------------------------------------------------- beyond the framework (VII)


def _section_map(kernel, twist=None):
    """f -> E_sigma[Psi] f at arbitrary points, from the package's spin components (Eq. 62)."""

    def apply(f):
        components = np.asarray(section.section_components(f, kernel, 0, L))

        def values(theta, phi):
            total = 0.0
            for sp in range(-(L - 1), L):
                v = ref.eval_spin(components[L - 1 + sp], sp, theta, phi)
                total = total + (
                    v if twist is None else np.exp(-1j * sp * twist(theta, phi)) * v
                )
            return total

        return values

    return apply


def _sadeghi_twist(theta, phi):
    return np.pi - phi + 0.0 * theta


def test_standard_correlation_equivariance(rng):
    """Equivariant under polar rotations only, unless the kernel is axisymmetric (Prop. VII.1)."""
    f, psi = generate_flm(rng, L), generate_flm(rng, L)
    standard = _section_map(kernels.lifted(psi, 0))
    assert (
        ref.equivariance_error(standard, f, 0, ref.random_rotation(rng), rng)
        > NOT_EQUIVARIANT
    )
    assert ref.equivariance_error(standard, f, 0, POLAR, rng) < EQUIVARIANT
    zonal = _section_map(kernels.lifted(_zonal(psi), 0))
    assert (
        ref.equivariance_error(zonal, f, 0, ref.random_rotation(rng), rng) < EQUIVARIANT
    )


def test_commutative_anisotropic_equivariance(rng):
    """Not equivariant under random or generic polar rotations; a polar rotation R_z(a) is a
    symmetry exactly when e^{ita} = 1 for every active output spin t, so a kernel with even orders
    only keeps the half turn (Proposition VII.1)."""
    f, h = generate_flm(rng, L), generate_flm(rng, L)
    sadeghi = _section_map(kernels.lifted(conjugate(h, 0), 0), _sadeghi_twist)
    assert (
        ref.equivariance_error(sadeghi, f, 0, ref.random_rotation(rng), rng)
        > NOT_EQUIVARIANT
    )
    assert ref.equivariance_error(sadeghi, f, 0, POLAR, rng) > NOT_EQUIVARIANT
    even = h * (ref.m_axis(L) % 2 == 0)[None]
    sadeghi_even = _section_map(kernels.lifted(conjugate(even, 0), 0), _sadeghi_twist)
    assert ref.equivariance_error(sadeghi_even, f, 0, HALF_TURN, rng) < EQUIVARIANT
    assert ref.equivariance_error(sadeghi, f, 0, HALF_TURN, rng) > NOT_EQUIVARIANT


def test_diagonal_operators_equivariance(rng):
    """Sifting commutes with polar rotations only; an m-independent multiplier with every rotation
    (Proposition VII.2)."""
    f, g = generate_flm(rng, L), generate_flm(rng, L)
    sifting = _on_sphere(lambda x: diagonal.sifting_convolution(x, g), 0)
    assert (
        ref.equivariance_error(sifting, f, 0, ref.random_rotation(rng), rng)
        > NOT_EQUIVARIANT
    )
    assert ref.equivariance_error(sifting, f, 0, POLAR, rng) < EQUIVARIANT
    isotropic = np.repeat(g[:, L - 1 : L], 2 * L - 1, axis=1) * ref.lm_mask(L)
    multiplication = _on_sphere(
        lambda x: diagonal.harmonic_multiplication(x, isotropic), 0
    )
    assert (
        ref.equivariance_error(multiplication, f, 0, ref.random_rotation(rng), rng)
        < EQUIVARIANT
    )
