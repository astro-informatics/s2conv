"""
Kernels of the generalised lifted convolution (:mod:`s2conv.kernels`).

The constructors of Table II are checked by sampling each kernel as a function on the
rotation group and Wigner transforming it with s2fft (the spherical kernels have orders
|m| < N, so that the sampling at azimuthal band-limit N is exact); the identity kernel
reproduces the lift, the projection and the identity (Cor. V.6); composition multiplies
kernel blocks (Prop. V.3); and the compact weights of Theorem V.5 (Eqs. 44, 54) are
checked against formulas written out from the kernel blocks.
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm, generate_flmn

from s2conv import kernels, lifting
from s2conv.convolutions import lifted

L = 12
N = 5


def _low_orders(flm):
    """The orders |m| < N of a field: its dependence on a third Euler angle is sampled exactly at N."""
    return flm * (np.abs(ref.m_axis(L)) < N)[None]


# -------------------------------------------------------------------------- constructors


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_lifted_kernel_is_the_sampled_lift(rng, spin):
    """Lambda_s psi sampled on SO(3) and Wigner transformed is the column k = -s (Lemma IV.7(c))."""
    psi = generate_flm(rng, L, spin=spin)
    sampled = ref.lift_grid(ref.isht(psi, L, spin), spin, N)
    kernel = kernels.lifted(psi, spin, N)
    assert ref.rel_err(kernel, ref.wigner_fwd(sampled, L, N)) < ref.TOL
    assert (
        ref.rel_err(ref.kernel_to_block(kernel), ref.block_lifted(psi, spin)) < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_flipped_kernel_is_the_sampled_flip(rng, spin):
    """(Lambda_s psi)~(g) = (Lambda_s psi)^*(g^{-1}) = (-1)^s (2 pi)^{-1/2} e^{-is alpha} psi^*(beta, pi - gamma),
    since g^{-1} has Euler angles (pi - gamma, beta, -pi - alpha); its blocks are Psi^{l dagger} (Prop. VI.2)."""
    psi = _low_orders(generate_flm(rng, L, spin=spin))
    thetas, phis = ref.sphere_grid(L)
    values = ref.eval_spin(
        psi, spin, thetas[None, :], (np.pi - ref.gamma_grid(N))[:, None]
    )
    phase = ref.parity(spin) / np.sqrt(2 * np.pi) * np.exp(-1j * spin * phis)
    sampled = phase[None, None, :] * np.conj(values)[:, :, None]
    kernel = kernels.flipped(kernels.lifted(psi, spin), N)
    assert kernel.shape == (2 * N - 1, L, 2 * L - 1)
    assert ref.rel_err(kernel, ref.wigner_fwd(sampled, L, N)) < ref.TOL


def test_left_kernel_is_the_sampled_kernel(rng):
    """Psi_h(g) = h^*(g^{-1} eta) = h^*(beta, pi - gamma), the row (Psi_h)^l_{0k} = 2 pi sqrt(4 pi/(2l+1)) h^*_{lk}
    (Prop. VI.4)."""
    h = _low_orders(generate_flm(rng, L))
    thetas, _ = ref.sphere_grid(L)
    values = ref.eval_spin(h, 0, thetas[None, :], (np.pi - ref.gamma_grid(N))[:, None])
    sampled = np.broadcast_to(np.conj(values)[:, :, None], (2 * N - 1, L, 2 * L - 1))
    kernel = kernels.left(h, N)
    assert ref.rel_err(kernel, ref.wigner_fwd(sampled, L, N)) < ref.TOL
    assert (
        ref.rel_err(ref.kernel_to_block(kernels.left(h)), ref.block_left(h)) < ref.TOL
    )


def test_flipped_is_the_conjugate_transpose(rng):
    block = ref.random_block(L, rng)
    kernel = ref.block_to_kernel(block, L)
    flipped = kernels.flipped(kernel)
    assert ref.rel_err(ref.kernel_to_block(flipped), ref.block_flip(block)) < ref.TOL
    assert ref.rel_err(kernels.flipped(flipped), kernel) < ref.TOL


def test_flipped_keeps_the_first_index_below_its_bandlimit(rng):
    """The flipped kernel's azimuthal band-limit holds the first index of Psi."""
    n_kernel, n_flip = 3, 4
    block = (
        ref.random_block(L, rng, n_kernel + 3)
        * (np.abs(ref.m_axis(L)) < n_kernel)[None, None, :]
    )
    flipped = kernels.flipped(ref.block_to_kernel(block, n_kernel), n_flip)
    expected = ref.block_flip(block) * (np.abs(ref.m_axis(L)) < n_flip)[None, None, :]
    assert flipped.shape == (2 * n_flip - 1, L, 2 * L - 1)
    assert ref.rel_err(ref.kernel_to_block(flipped), expected) < ref.TOL


@pytest.mark.parametrize(("s_in", "s_out"), [(2, 1), (0, 0), (-1, 3), (3, -2)])
def test_multiplier_kernel_is_the_canonical_entry(rng, s_in, s_out):
    """(Psi_T)^l_{-s',-s} = (-1)^{s+s'} c^*_l for l >= max(|s|, |s'|) (Theorem V.5(a))."""
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    kernel = kernels.multiplier(c, s_in, s_out)
    assert kernel.shape == (2 * abs(s_in) + 1, L, 2 * L - 1)
    assert (
        ref.rel_err(ref.kernel_to_block(kernel), ref.block_multiplier(c, s_in, s_out))
        < ref.TOL
    )
    assert (
        ref.rel_err(
            kernels.from_multiplier(c, s_in, s_out, N),
            kernels.multiplier(c, s_in, s_out, N),
        )
        < ref.TOL
    )


@pytest.mark.parametrize("spin", [0, 2, -1])
def test_identity_kernel_gives_lift_projection_and_identity(rng, spin):
    """C_{s->G} = Lambda_s, C_{G->s} = Lambda_s^dagger and C_{s->s} = Id (Corollary V.6)."""
    identity = kernels.identity(L, spin)
    flm = generate_flm(rng, L, spin=spin)
    flmn = generate_flmn(rng, L, N)
    n_out = abs(spin) + 2
    assert (
        ref.rel_err(
            lifted.spin_to_group(flm, identity, spin, n_out), ref.lift(flm, spin, n_out)
        )
        < ref.TOL
    )
    assert (
        ref.rel_err(lifted.group_to_spin(flmn, identity, spin), ref.project(flmn, spin))
        < ref.TOL
    )
    assert ref.rel_err(lifted.spin_to_spin(flm, identity, spin, spin), flm) < ref.TOL


def test_steerable_kernel_produces_the_spin_components(rng):
    """With C_{s->G}, a steerable column yields the spin-changing convolutions as spin components (Prop. VI.6)."""
    s_in, n_out = 1, 4
    multipliers = {
        s: rng.standard_normal(L) + 1j * rng.standard_normal(L) for s in (-3, 0, 2)
    }
    kernel = kernels.steerable(multipliers, s_in)
    expected_block = sum(
        ref.block_multiplier(c, s_in, s) for s, c in multipliers.items()
    )
    assert ref.rel_err(ref.kernel_to_block(kernel), expected_block) < ref.TOL
    flm = generate_flm(rng, L, spin=s_in)
    out = lifted.spin_to_group(flm, kernel, s_in, n_out)
    spins = range(-(n_out - 1), n_out)
    components = np.stack([lifting.project(out, s) for s in spins])
    expected = np.stack(
        [
            ref.apply_multiplier(multipliers[s], flm, s_in, s)
            if s in multipliers
            else 0 * flm
            for s in spins
        ]
    )
    assert ref.rel_err(components, expected) < ref.TOL


# --------------------------------------------------------------------------- composition


def test_compose_multiplies_blocks(rng):
    """Psi_3^l = Psi_2^l Psi_1^l (Proposition V.3)."""
    block1, block2 = ref.random_block(L, rng, N), ref.random_block(L, rng, N)
    composed = kernels.compose(
        ref.block_to_kernel(block2, N), ref.block_to_kernel(block1, N)
    )
    expected = np.einsum("lab,lbc->lac", block2, block1)
    assert ref.rel_err(ref.kernel_to_block(composed), expected) < ref.TOL


def test_composition_of_group_convolutions(rng):
    """C_{G->G}[Psi_2] C_{G->G}[Psi_1] = C_{G->G}[Psi_2 Psi_1] (Proposition V.3, Eq. 47)."""
    kernel1 = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    kernel2 = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    flmn = generate_flmn(rng, L, N)
    twice = lifted.group_to_group(lifted.group_to_group(flmn, kernel1, N), kernel2, N)
    assert (
        ref.rel_err(
            lifted.group_to_group(flmn, kernels.compose(kernel2, kernel1), N), twice
        )
        < ref.TOL
    )


def test_composition_of_directional_and_group_convolutions(rng):
    """C_{G->G}[Psi_2] C_{1->G}[Lambda_1 psi] = C_{1->G}[Psi_2 Lambda_1 psi] (Proposition V.3)."""
    psi = generate_flm(rng, L, spin=1)
    kernel2 = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    flm = generate_flm(rng, L, spin=1)
    twice = lifted.group_to_group(
        lifted.spin_to_group(flm, kernels.lifted(psi, 1), 1, N), kernel2, N
    )
    composed = kernels.compose(kernel2, kernels.lifted(psi, 1))
    assert ref.rel_err(lifted.spin_to_group(flm, composed, 1, N), twice) < ref.TOL


def test_composition_with_lift_and_projection_at_the_ends(rng):
    """C_{G->s'}[Psi_2] C_{s->G}[Psi_1] = C_{s->s'}[Psi_2 Psi_1] (Proposition V.3)."""
    s_in, s_out = -2, 1
    kernel1 = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    kernel2 = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    flm = generate_flm(rng, L, spin=s_in)
    twice = lifted.group_to_spin(
        lifted.spin_to_group(flm, kernel1, s_in, N), kernel2, s_out
    )
    composed = kernels.compose(kernel2, kernel1)
    assert ref.rel_err(lifted.spin_to_spin(flm, composed, s_in, s_out), twice) < ref.TOL


def test_composition_through_a_spin_field_multiplies_multipliers(rng):
    """C_{-2->0}[Psi_2] C_{1->-2}[Psi_1]: only the index 2 of the intermediate survives, and the
    multipliers multiply, c3 = c2 c1 (Proposition V.3)."""
    block1, block2 = ref.random_block(L, rng, N), ref.random_block(L, rng, N)
    flm = generate_flm(rng, L, spin=1)
    intermediate = lifted.spin_to_spin(flm, ref.block_to_kernel(block1, N), 1, -2)
    twice = lifted.spin_to_spin(intermediate, ref.block_to_kernel(block2, N), -2, 0)
    c3 = ref.multiplier_of(block2, -2, 0) * ref.multiplier_of(block1, 1, -2)
    assert ref.rel_err(twice, ref.apply_multiplier(c3, flm, 1, 0)) < ref.TOL
    # the canonical kernels compose to the canonical kernel of the product (degrees l >= 2 survive)
    c1, c2 = rng.standard_normal(L) + 1j, rng.standard_normal(L) - 1j
    composed = kernels.compose(
        kernels.multiplier(c2, -2, 0), kernels.multiplier(c1, 1, -2)
    )
    expected = ref.block_multiplier(c2 * c1 * (np.arange(L) >= 2), 1, 0)
    assert ref.rel_err(ref.kernel_to_block(composed), expected) < ref.TOL


# --------------------------------------------------------------------------- conversions


@pytest.mark.parametrize(("s_in", "s_out"), [(2, 1), (0, 0), (-1, 3), (4, -4)])
def test_to_multiplier(rng, s_in, s_out):
    """c_l = (-1)^{s+s'} Psi^{l*}_{-s',-s}, zero below l = max(|s|, |s'|) (Eq. 44)."""
    block = ref.random_block(L, rng)
    c = kernels.to_multiplier(ref.block_to_kernel(block, N), s_in, s_out)
    expected = ref.parity(s_in + s_out) * np.conj(block[:, L - 1 - s_out, L - 1 - s_in])
    expected = expected * (np.arange(L) >= max(abs(s_in), abs(s_out)))
    assert ref.rel_err(c, expected) < ref.TOL


def test_to_multiplier_of_a_kernel_without_the_column(rng):
    """A kernel azimuthally band-limited at N <= |s| does not see a spin-s input."""
    kernel = ref.block_to_kernel(ref.random_block(L, rng, 2), 2)
    assert np.all(kernels.to_multiplier(kernel, 3, 0) == 0)


@pytest.mark.parametrize(("s_in", "n_out"), [(0, 3), (2, 5), (-3, L)])
def test_to_column(rng, s_in, n_out):
    """u^l_n = (-1)^s a_l^{-1/2} Psi^{l*}_{n,-s}, a_l = (2l+1)/(8 pi^2) (Eq. 54)."""
    block = ref.random_block(L, rng)
    u = kernels.to_column(ref.block_to_kernel(block, L), s_in, n_out)
    a = (2 * np.arange(L) + 1) / ref.EIGHT_PI2
    expected = np.array(
        [
            ref.parity(s_in) / np.sqrt(a) * np.conj(block[:, L - 1 + n, L - 1 - s_in])
            for n in range(-(n_out - 1), n_out)
        ]
    )
    assert u.shape == (2 * n_out - 1, L)
    assert ref.rel_err(u, expected) < ref.TOL


@pytest.mark.parametrize(("s_out", "n_in"), [(0, 3), (-2, 5), (3, 7)])
def test_to_row(rng, s_out, n_in):
    """v^l_k = (-1)^{s'} a_l^{1/2} Psi^{l*}_{-s',k} (Eq. 54), zero-padded beyond the kernel's band-limit."""
    n_kernel = 5
    block = ref.random_block(L, rng, n_kernel)
    v = kernels.to_row(ref.block_to_kernel(block, n_kernel), s_out, n_in)
    a = (2 * np.arange(L) + 1) / ref.EIGHT_PI2
    expected = np.array(
        [
            ref.parity(s_out) * np.sqrt(a) * np.conj(block[:, L - 1 - s_out, L - 1 + k])
            for k in range(-(n_in - 1), n_in)
        ]
    )
    assert v.shape == (2 * n_in - 1, L)
    assert ref.rel_err(v, expected) < ref.TOL


@pytest.mark.parametrize(("n_in", "n_out"), [(3, 3), (2, 5), (6, 4)])
def test_to_block(rng, n_in, n_out):
    """B^l = Psi^{l dagger}, B[l, N_in - 1 + k, N_out - 1 + n] = Psi^{l*}_{nk} (Eq. 54)."""
    block = ref.random_block(L, rng, 5)
    B = kernels.to_block(ref.block_to_kernel(block, 5), n_in, n_out)
    expected = np.conj(np.transpose(block, (0, 2, 1)))[
        :, L - n_in : L + n_in - 1, L - n_out : L + n_out - 1
    ]
    assert B.shape == (L, 2 * n_in - 1, 2 * n_out - 1)
    assert ref.rel_err(B, expected) < ref.TOL


@pytest.mark.parametrize("s_in", [0, 2, -1])
def test_from_column_round_trip(rng, s_in):
    n_out = 4
    u = rng.standard_normal((2 * n_out - 1, L)) + 1j * rng.standard_normal(
        (2 * n_out - 1, L)
    )
    kernel = kernels.from_column(u, s_in)
    assert kernel.shape == (2 * abs(s_in) + 1, L, 2 * L - 1)
    assert ref.rel_err(kernels.to_column(kernel, s_in, n_out), u) < ref.TOL
    # the canonical kernel has the single column k = -s
    block = ref.kernel_to_block(kernel)
    assert np.all(np.delete(block, L - 1 - s_in, axis=2) == 0)


@pytest.mark.parametrize("s_out", [0, -2, 3])
def test_from_row_round_trip(rng, s_out):
    n_in = 4
    v = rng.standard_normal((2 * n_in - 1, L)) + 1j * rng.standard_normal(
        (2 * n_in - 1, L)
    )
    kernel = kernels.from_row(v, s_out)
    assert ref.rel_err(kernels.to_row(kernel, s_out, n_in), v) < ref.TOL
    block = ref.kernel_to_block(kernel)
    assert np.all(np.delete(block, L - 1 - s_out, axis=1) == 0)


def test_from_block_round_trip(rng):
    n_in, n_out = 3, 5
    block = (
        ref.random_block(L, rng, max(n_in, n_out))
        * (np.abs(ref.m_axis(L)) < n_out)[None, :, None]
    )
    B = np.conj(np.transpose(block, (0, 2, 1)))[
        :, L - n_in : L + n_in - 1, L - n_out : L + n_out - 1
    ]
    kernel = kernels.from_block(B)
    assert kernel.shape == (2 * n_in - 1, L, 2 * L - 1)
    assert ref.rel_err(kernels.to_block(kernel, n_in, n_out), B) < ref.TOL
    assert (
        ref.rel_err(
            ref.kernel_to_block(kernel),
            block * (np.abs(ref.m_axis(L)) < n_in)[None, None, :],
        )
        < ref.TOL
    )


@pytest.mark.parametrize(("s_in", "s_out"), [(1, 1), (0, -2)])
def test_kernel_forms_equal_weight_forms(rng, s_in, s_out):
    """Each operator applied to a kernel equals its weight form with the converted weights (Theorem V.5)."""
    kernel = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    flm = generate_flm(rng, L, spin=s_in)
    flmn = generate_flmn(rng, L, N)
    c = kernels.to_multiplier(kernel, s_in, s_out)
    assert (
        ref.rel_err(
            lifted.spin_to_spin(flm, kernel, s_in, s_out),
            ref.apply_multiplier(c, flm, s_in, s_out),
        )
        < ref.TOL
    )
    u = kernels.to_column(kernel, s_in, N)
    assert (
        ref.rel_err(
            lifted.spin_to_group(flm, kernel, s_in, N),
            lifted.spin_to_group_weights(flm, u, s_in),
        )
        < ref.TOL
    )
    assert (
        ref.rel_err(
            lifted.spin_to_group(flm, kernels.from_column(u, s_in), s_in, N),
            lifted.spin_to_group(flm, kernel, s_in, N),
        )
        < ref.TOL
    )
    v = kernels.to_row(kernel, s_out, N)
    assert (
        ref.rel_err(
            lifted.group_to_spin(flmn, kernels.from_row(v, s_out), s_out),
            lifted.group_to_spin(flmn, kernel, s_out),
        )
        < ref.TOL
    )
    B = kernels.to_block(kernel, N, N)
    assert (
        ref.rel_err(
            lifted.group_to_group(flmn, kernels.from_block(B), N),
            lifted.group_to_group(flmn, kernel, N),
        )
        < ref.TOL
    )
