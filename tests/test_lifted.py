"""
The generalised lifted convolution (Sec. V) of :mod:`s2conv.convolutions.lifted`.

Each of the four operators of Definition V.1 is computed literally, on samples (the lift
on the SO(3) grid, s2fft's Wigner transform, the block product of Eq. (20), the inverse
transform and the fibre integral), and compared with the package, which applies the
closed forms of Theorem V.2 (Eqs. 42-45).  The adjoints with the flipped kernel,
Corollary V.4, the weight forms of Theorem V.5 and the sampled wrappers are checked too.
"""

import numpy as np
import pytest
import reference as ref
import s2fft
from s2fft.utils.signal_generator import generate_flm, generate_flmn

import s2conv
from s2conv import kernels, lifting
from s2conv.convolutions import lifted

L = 12
N = 5  # the literal checks need kernels with orders |n|, |k| < N

SPIN_PAIRS = [(0, 0), (2, 1), (-1, 3), (3, -2), (-4, 4), (1, -1)]


def _kernel(rng, n=N):
    """A random kernel with orders |n|, |k| < n, as a block and in the s2fft layout."""
    block = ref.random_block(L, rng, n)
    return block, ref.block_to_kernel(block, n)


# ------------------------------------------------------------ literal evaluation (Thm V.2)


@pytest.mark.parametrize(("s_in", "s_out"), SPIN_PAIRS)
def test_spin_to_spin_is_the_literal_operator(rng, s_in, s_out):
    """C_{s->s'}[Psi]: lift on the grid, Wigner transform, block product, fibre integral (Eq. 43)."""
    block, kernel = _kernel(rng)
    flm = generate_flm(rng, L, spin=s_in)
    literal = ref.sht(
        ref.lifted_conv_literal(block, ref.isht(flm, L, s_in), s_in, s_out, N), L, s_out
    )
    assert ref.rel_err(lifted.spin_to_spin(flm, kernel, s_in, s_out), literal) < ref.TOL


@pytest.mark.parametrize("s_in", [0, 2, -3])
def test_spin_to_group_is_the_literal_operator(rng, s_in):
    """C_{s->G}[Psi] (Eq. 42)."""
    block, kernel = _kernel(rng)
    flm = generate_flm(rng, L, spin=s_in)
    literal = ref.wigner_fwd(
        ref.lifted_conv_literal(block, ref.isht(flm, L, s_in), s_in, "G", N), L, N
    )
    assert ref.rel_err(lifted.spin_to_group(flm, kernel, s_in, N), literal) < ref.TOL


@pytest.mark.parametrize("s_out", [0, 1, -4])
def test_group_to_spin_is_the_literal_operator(rng, s_out):
    """C_{G->s'}[Psi] (Eq. 44)."""
    block, kernel = _kernel(rng)
    flmn = generate_flmn(rng, L, N)
    literal = ref.sht(
        ref.lifted_conv_literal(block, ref.wigner_inv(flmn, L, N), "G", s_out, N),
        L,
        s_out,
    )
    assert ref.rel_err(lifted.group_to_spin(flmn, kernel, s_out), literal) < ref.TOL


def test_group_to_group_is_the_literal_operator(rng):
    """C_{G->G}[Psi], the group convolution (Eqs. 20, 45)."""
    block, kernel = _kernel(rng)
    flmn = generate_flmn(rng, L, N)
    literal = ref.wigner_fwd(
        ref.lifted_conv_literal(block, ref.wigner_inv(flmn, L, N), "G", "G", N), L, N
    )
    assert ref.rel_err(lifted.group_to_group(flmn, kernel, N), literal) < ref.TOL
    assert ref.rel_err(lifted.group_convolution(flmn, kernel, N), literal) < ref.TOL


def test_group_convolution_is_the_correlation_on_the_group(rng):
    """(Psi * F)(rho) = int Psi^*(rho^{-1} rho') F(rho') d mu(rho') = <F, pi_G(rho) Psi> (Definition III.4),
    by Parseval with the kernel left-translated as a function on SO(3)."""
    _, kernel = _kernel(rng)
    flmn = generate_flmn(rng, L, N)
    out = np.asarray(lifted.group_convolution(flmn, kernel, N))
    rotations = [ref.random_rotation(rng) for _ in range(4)]
    values = [ref.eval_so3(out, *rho) for rho in rotations]
    expected = [ref.inner_so3(flmn, ref.rotate_flmn(kernel, rho)) for rho in rotations]
    assert ref.rel_err(values, expected) < ref.TOL


# -------------------------------------------------- closed forms with every kernel order


def _full_kernel(rng):
    """A random kernel with every Wigner coefficient populated, as a block and in the s2fft layout."""
    block = ref.random_block(L, rng)
    return block, ref.block_to_kernel(block, L)


def _assert_closed_form(actual, block, x, s_in, s_out):
    """Against the composition Lambda^dagger (Psi * Lambda) of Definition V.1 and Eqs. (42)-(45)."""
    assert ref.rel_err(actual, ref.theorem_v2(block, x, s_in, s_out)) < ref.TOL
    assert ref.rel_err(actual, ref.lifted_conv(block, x, s_in, s_out)) < ref.TOL


@pytest.mark.parametrize(("s_in", "s_out"), SPIN_PAIRS + [(7, -9), (-11, 0)])
def test_spin_to_spin_closed_form_with_a_full_kernel(rng, s_in, s_out):
    """Eq. (43) with every entry of the kernel populated, at any spins (no transform involved)."""
    block, kernel = _full_kernel(rng)
    flm = generate_flm(rng, L, spin=s_in)
    _assert_closed_form(
        lifted.spin_to_spin(flm, kernel, s_in, s_out), block, flm, s_in, s_out
    )


@pytest.mark.parametrize("spin", [0, 3, -4, 9, -11])
def test_spin_to_group_closed_form_with_a_full_kernel(rng, spin):
    """Eq. (42)."""
    block, kernel = _full_kernel(rng)
    flm = generate_flm(rng, L, spin=spin)
    _assert_closed_form(lifted.spin_to_group(flm, kernel, spin), block, flm, spin, "G")


@pytest.mark.parametrize("spin", [0, 3, -4, 9, -11])
def test_group_to_spin_closed_form_with_a_full_kernel(rng, spin):
    """Eq. (44)."""
    block, kernel = _full_kernel(rng)
    flmn = generate_flmn(rng, L, L)
    _assert_closed_form(
        lifted.group_to_spin(flmn, kernel, spin), block, flmn, "G", spin
    )


def test_group_to_group_closed_form_with_a_full_kernel(rng):
    """Eq. (45)."""
    block, kernel = _full_kernel(rng)
    flmn = generate_flmn(rng, L, L)
    _assert_closed_form(lifted.group_to_group(flmn, kernel), block, flmn, "G", "G")


@pytest.mark.parametrize(("s_in", "s_out"), [(2, 1), (0, "G"), ("G", -3), ("G", "G")])
def test_lifted_convolution_dispatches(rng, s_in, s_out):
    block = ref.random_block(L, rng)
    kernel = ref.block_to_kernel(block, L)
    x = generate_flmn(rng, L, N) if s_in == "G" else generate_flm(rng, L, spin=s_in)
    n_out = 4 if s_out == "G" else None
    out = lifted.lifted_convolution(x, kernel, s_in, s_out, n_out)
    assert ref.rel_err(out, ref.lifted_conv(block, x, s_in, s_out, n_out)) < ref.TOL
    assert (
        ref.rel_err(s2conv.lifted_convolution(x, kernel, s_in, s_out, n_out), out)
        < ref.TOL
    )


@pytest.mark.parametrize("n_out", [1, 3, L])
def test_spin_to_group_output_bandlimit(rng, n_out):
    """The output keeps the orders |n| < N_out; the default is N_out = L."""
    block = ref.random_block(L, rng)
    kernel = ref.block_to_kernel(block, L)
    flm = generate_flm(rng, L, spin=1)
    out = lifted.spin_to_group(flm, kernel, 1, n_out)
    assert out.shape == (2 * n_out - 1, L, 2 * L - 1)
    assert ref.rel_err(out, ref.theorem_v2(block, flm, 1, "G", n_out)) < ref.TOL
    assert lifted.spin_to_group(flm, kernel, 1).shape == (2 * L - 1, L, 2 * L - 1)


@pytest.mark.parametrize("n_out", [1, 2, 4])
def test_group_to_group_output_bandlimit(rng, n_out):
    """The output keeps the orders |n| < N_out of the kernel's first index."""
    block, kernel = _kernel(rng)
    flmn = generate_flmn(rng, L, N)
    out = lifted.group_to_group(flmn, kernel, n_out)
    assert out.shape == (2 * n_out - 1, L, 2 * L - 1)
    assert ref.rel_err(out, ref.theorem_v2(block, flmn, "G", "G", n_out)) < ref.TOL


def test_kernel_narrower_than_the_input(rng):
    """A kernel azimuthally band-limited below the input's N reads zeros beyond its band-limit."""
    block, kernel = _kernel(rng, 3)
    flmn = generate_flmn(rng, L, N)
    assert (
        ref.rel_err(
            lifted.group_to_spin(flmn, kernel, 2), ref.theorem_v2(block, flmn, "G", 2)
        )
        < ref.TOL
    )
    assert (
        ref.rel_err(
            lifted.group_to_group(flmn, kernel, N),
            ref.theorem_v2(block, flmn, "G", "G", N),
        )
        < ref.TOL
    )
    flm = generate_flm(rng, L, spin=4)
    assert np.all(lifted.spin_to_group(flm, kernel, 4, N) == 0)
    assert np.all(lifted.spin_to_spin(flm, kernel, 4, 0) == 0)


def test_lifted_input_reduces_group_to_spin_input(rng):
    """C_{G->G}[Psi] Lambda_s f = C_{s->G}[Psi] f (Sec. VI D)."""
    _, kernel = _kernel(rng)
    flm = generate_flm(rng, L, spin=1)
    lhs = lifted.group_to_group(lifting.lift(flm, 1, N), kernel, N)
    assert ref.rel_err(lhs, lifted.spin_to_group(flm, kernel, 1, N)) < ref.TOL
    lhs = lifted.group_to_spin(lifting.lift(flm, 1, N), kernel, -2)
    assert ref.rel_err(lhs, lifted.spin_to_spin(flm, kernel, 1, -2)) < ref.TOL


# ----------------------------------------------------------------------------- adjoints


@pytest.mark.parametrize(("s_in", "s_out"), SPIN_PAIRS)
def test_spin_to_spin_adjoint(rng, s_in, s_out):
    """<C_{s->s'}[Psi] f, g> = <f, C_{s'->s}[Psi~] g> (Theorem V.2)."""
    kernel = ref.block_to_kernel(ref.random_block(L, rng), L)
    flm, glm = generate_flm(rng, L, spin=s_in), generate_flm(rng, L, spin=s_out)
    lhs = ref.inner_s2(lifted.spin_to_spin(flm, kernel, s_in, s_out), glm)
    rhs = ref.inner_s2(
        flm, lifted.spin_to_spin(glm, kernels.flipped(kernel), s_out, s_in)
    )
    assert abs(lhs - rhs) < ref.TOL * abs(lhs)


@pytest.mark.parametrize("spin", [0, 2, -3])
def test_spin_to_group_adjoint(rng, spin):
    """<C_{s->G}[Psi] f, G>_SO(3) = <f, C_{G->s}[Psi~] G> (Theorem V.2)."""
    kernel = ref.block_to_kernel(ref.random_block(L, rng), L)
    flm, flmn = generate_flm(rng, L, spin=spin), generate_flmn(rng, L, L)
    lhs = ref.inner_so3(lifted.spin_to_group(flm, kernel, spin), flmn)
    rhs = ref.inner_s2(flm, lifted.group_to_spin(flmn, kernels.flipped(kernel), spin))
    assert abs(lhs - rhs) < ref.TOL * abs(lhs)


def test_group_to_group_adjoint(rng):
    """<C_{G->G}[Psi] F, G> = <F, C_{G->G}[Psi~] G> (Theorem V.2)."""
    kernel = ref.block_to_kernel(ref.random_block(L, rng), L)
    F, G = generate_flmn(rng, L, L), generate_flmn(rng, L, L)
    lhs = ref.inner_so3(lifted.group_to_group(F, kernel), G)
    rhs = ref.inner_so3(F, lifted.group_to_group(G, kernels.flipped(kernel)))
    assert abs(lhs - rhs) < ref.TOL * abs(lhs)


# -------------------------------------------------------------------------- Corollary V.4


@pytest.mark.parametrize(("s_in", "s_out"), [(2, 1), (0, -3), (-1, 0)])
def test_constrained_kernel_projection_is_evaluation_on_the_section(rng, s_in, s_out):
    """With the column -s supported on n = -s' only, projecting equals sqrt(2 pi) times evaluation
    of Psi * Lambda_s f at gamma = 0; with a generic kernel it does not (Corollary V.4)."""
    block = ref.random_block(L, rng, N)
    constrained = np.zeros_like(block)
    constrained[:, L - 1 - s_out, L - 1 - s_in] = block[:, L - 1 - s_out, L - 1 - s_in]
    flm = generate_flm(rng, L, spin=s_in)
    samples = ref.isht(flm, L, s_in)
    errors = {}
    for name, b in (("constrained", constrained), ("generic", block)):
        on_section = (
            np.sqrt(2 * np.pi) * ref.lifted_conv_literal(b, samples, s_in, "G", N)[0]
        )
        projected = ref.isht(
            lifted.spin_to_spin(flm, ref.block_to_kernel(b, N), s_in, s_out), L, s_out
        )
        errors[name] = ref.rel_err(on_section, projected)
    assert errors["constrained"] < ref.TOL
    assert errors["generic"] > 1e-2


# --------------------------------------------------------------------------- weight forms


def test_weight_forms(rng):
    """F^l_{mn} = u^l_n f_lm, g_lm = sum_k v^l_k F^l_{mk}, G^l = F^l B^l, g_lm = c_l f_lm (Theorem V.5)."""
    s_in, s_out, n_in, n_out = 2, -1, 4, 3
    flm = generate_flm(rng, L, spin=s_in)
    flmn = generate_flmn(rng, L, n_in)
    u = rng.standard_normal((2 * n_out - 1, L)) + 1j * rng.standard_normal(
        (2 * n_out - 1, L)
    )
    v = rng.standard_normal((2 * n_in - 1, L)) + 1j * rng.standard_normal(
        (2 * n_in - 1, L)
    )
    B = rng.standard_normal(
        (L, 2 * n_in - 1, 2 * n_out - 1)
    ) + 1j * rng.standard_normal((L, 2 * n_in - 1, 2 * n_out - 1))
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)

    expected = np.zeros((2 * n_out - 1, L, 2 * L - 1), np.complex128)
    for n in range(-(n_out - 1), n_out):
        expected[n_out - 1 + n] = (
            u[n_out - 1 + n][:, None] * flm * ref.lm_mask(L, max(abs(n), abs(s_in)))
        )
    assert ref.rel_err(lifted.spin_to_group_weights(flm, u, s_in), expected) < ref.TOL

    expected = sum(
        v[n_in - 1 + k][:, None] * flmn[n_in - 1 + k] for k in range(-(n_in - 1), n_in)
    )
    assert (
        ref.rel_err(
            lifted.group_to_spin_weights(flmn, v, s_out),
            expected * ref.lm_mask(L, s_out),
        )
        < ref.TOL
    )

    expected = np.zeros((2 * n_out - 1, L, 2 * L - 1), np.complex128)
    for n in range(-(n_out - 1), n_out):
        for k in range(-(n_in - 1), n_in):
            expected[n_out - 1 + n] += (
                flmn[n_in - 1 + k] * B[:, n_in - 1 + k, n_out - 1 + n][:, None]
            )
        expected[n_out - 1 + n] *= ref.lm_mask(L, n)
    assert ref.rel_err(lifted.group_to_group_weights(flmn, B), expected) < ref.TOL

    assert (
        ref.rel_err(
            lifted.spin_multiplier(flm, c, s_in, s_out),
            ref.apply_multiplier(c, flm, s_in, s_out),
        )
        < ref.TOL
    )


# ----------------------------------------------------------------------- sampled wrappers


@pytest.mark.parametrize(
    "method",
    ["jax", "precompute", "numpy", pytest.param("jax_ssht", marks=pytest.mark.ssht)],
)
@pytest.mark.parametrize(("s_in", "s_out"), [(2, -1), (1, "G"), ("G", 0), ("G", "G")])
def test_convolve_samples_is_the_literal_operator(rng, method, s_in, s_out):
    """s2conv.convolve on samples equals the literal lifted convolution on the SO(3) grid."""
    block, kernel = _kernel(rng)
    if s_in == "G":
        x = ref.wigner_inv(generate_flmn(rng, L, N), L, N)
    else:
        x = ref.isht(generate_flm(rng, L, spin=s_in), L, s_in)
    literal = ref.lifted_conv_literal(block, x, s_in, s_out, N)
    n_out = N if s_out == "G" else None
    out = s2conv.convolve(x, kernel, L, s_in, s_out, N_out=n_out, method=method)
    assert ref.rel_err(out, literal) < ref.TOL


@pytest.mark.parametrize("sampling", ["mwss", "dh", "gl"])
@pytest.mark.parametrize(("s_in", "s_out"), [(1, -2), ("G", "G")])
def test_convolve_in_other_samplings(rng, sampling, s_in, s_out):
    """s2conv.convolve on other equiangular samplings, against the closed forms (43)-(46)."""
    block, kernel = _kernel(rng, 3)
    if s_in == "G":
        x_lm = generate_flmn(rng, L, 3)
        x = s2fft.wigner.inverse(x_lm, L, 3, sampling=sampling, method="numpy")
    else:
        x_lm = generate_flm(rng, L, spin=s_in)
        x = s2fft.inverse(x_lm, L, spin=s_in, sampling=sampling, method="numpy")
    n_out = 3 if s_out == "G" else None
    out = s2conv.convolve(x, kernel, L, s_in, s_out, N_out=n_out, sampling=sampling)
    expected = ref.theorem_v2(block, x_lm, s_in, s_out, n_out)
    if s_out == "G":
        expected = s2fft.wigner.inverse(
            expected, L, 3, sampling=sampling, method="numpy"
        )
    else:
        expected = s2fft.inverse(
            expected, L, spin=s_out, sampling=sampling, method="numpy"
        )
    assert ref.rel_err(out, expected) < ref.TOL


@pytest.mark.parametrize("method", ["jax", "precompute"])
def test_convolve_multiplier(rng, method):
    s_in, s_out = -2, 1
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    flm = generate_flm(rng, L, spin=s_in)
    out = s2conv.convolve_multiplier(
        ref.isht(flm, L, s_in), c, L, s_in, s_out, method=method
    )
    assert (
        ref.rel_err(out, ref.isht(ref.apply_multiplier(c, flm, s_in, s_out), L, s_out))
        < ref.TOL
    )


@pytest.mark.ssht
@pytest.mark.parametrize(("s_in", "s_out"), [(9, 8), (-10, 11)])
def test_convolve_multiplier_at_high_spin(rng, s_in, s_out):
    """Spins beyond the recursion limit through the precompute backend, checked with SSHT."""
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    flm = generate_flm(rng, L, spin=s_in)
    f = s2conv.inverse(flm, L, s_in, method="jax_ssht")
    out = s2conv.convolve_multiplier(f, c, L, s_in, s_out, method="precompute")
    expected = s2conv.inverse(
        ref.apply_multiplier(c, flm, s_in, s_out), L, s_out, method="jax_ssht"
    )
    assert ref.rel_err(out, expected) < ref.TOL


def test_convolve_default_output_bandlimit_with_precompute(rng):
    """With s_out = "G" the output's azimuthal band-limit defaults to L."""
    block = ref.random_block(L, rng)
    flm = generate_flm(rng, L)
    out = s2conv.convolve(
        ref.isht(flm, L), ref.block_to_kernel(block, 1), L, 0, "G", method="precompute"
    )
    assert out.shape == (2 * L - 1, L, 2 * L - 1)
    # compare at a few samples with Eq. (7) evaluated on the closed form (43)
    thetas, phis = ref.sphere_grid(L)
    gammas = ref.gamma_grid(L)
    j, t, p = np.array([0, 5, 22, 13]), np.array([0, 4, 11, 7]), np.array([0, 9, 20, 3])
    expected = ref.eval_so3(
        ref.theorem_v2(block, flm, 0, "G"), phis[p], thetas[t], gammas[j]
    )
    assert ref.rel_err(np.asarray(out)[j, t, p], expected) < ref.TOL


def test_convolve_default_output_bandlimit_is_guarded_on_the_jax_backend(rng):
    """With the default N_out = L >= 8 the "jax" backend refuses the Wigner transform."""
    kernel = kernels.identity(L, 0)
    with pytest.raises(ValueError, match="precompute"):
        s2conv.convolve(ref.isht(generate_flm(rng, L), L), kernel, L, 0, "G")
