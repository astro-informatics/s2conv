"""
JIT compilation and automatic differentiation.

The sampled convolutions compiled with ``jax.jit`` (spins, band-limits, sampling and
method static) agree with their eager evaluation, also when the caches of precomputed
arrays are first filled while tracing.  Reverse-mode gradients (``jax.grad``) through
the transforms, on both the ``"jax"`` and ``"precompute"`` backends, and through the
harmonic operators agree with central finite differences; forward mode is used only
through the harmonic operators, since the ``"jax"`` transforms of s2fft 1.4.0 support
reverse mode only.  Gradients with respect to Euler angles are finite and the norm is
rotation invariant.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm, generate_flmn

import s2conv
from s2conv import kernels, transforms
from s2conv.convolutions import classical, lifted, physics
from s2conv.utils import rotation

L = 8
N = 3
STATIC = ("L", "s_in", "s_out", "N_out", "sampling", "method")
convolve_jit = jax.jit(s2conv.convolve, static_argnames=STATIC)

CONFIGS = [(2, -1), (1, "G"), ("G", 0), ("G", "G")]


def _input_and_kernel(rng, s_in):
    """Samples x of an input of spin s_in (or on SO(3)), a direction v, and a kernel with |n|, |k| < N."""
    block = ref.random_block(L, rng, N)
    if s_in == "G":
        x = ref.wigner_inv(generate_flmn(rng, L, N), L, N)
        v = ref.wigner_inv(generate_flmn(rng, L, N), L, N)
    else:
        x = ref.isht(generate_flm(rng, L, spin=s_in), L, s_in)
        v = ref.isht(generate_flm(rng, L, spin=s_in), L, s_in)
    return x, v, block, ref.block_to_kernel(block, N)


def _central_difference(fn, h=1e-3):
    """Five-point central difference at 0, exact up to round-off for a polynomial of degree 4."""
    return (-fn(2 * h) + 8 * fn(h) - 8 * fn(-h) + fn(-2 * h)) / (12 * h)


# ------------------------------------------------------------------------------------ jit


@pytest.mark.parametrize("method", ["jax", "precompute"])
@pytest.mark.parametrize(("s_in", "s_out"), CONFIGS)
def test_jit_convolve_matches_eager(rng, method, s_in, s_out):
    x, _, _, kernel = _input_and_kernel(rng, s_in)
    n_out = N if s_out == "G" else None
    eager = s2conv.convolve(x, kernel, L, s_in, s_out, N_out=n_out, method=method)
    compiled = convolve_jit(
        x, kernel, L=L, s_in=s_in, s_out=s_out, N_out=n_out, method=method
    )
    assert ref.rel_err(compiled, eager) < 1e-12


@pytest.fixture
def cold_caches():
    """Empty s2conv's caches of precomputed arrays before the test, and again after it."""
    caches = (
        rotation.delta_matrices,
        transforms.spherical_kernel,
        transforms.wigner_kernel,
    )
    for cache in caches:
        cache.cache_clear()
    yield caches
    for cache in caches:
        cache.cache_clear()


def _cached_value(cache, *args):
    """The value cached for args, checking that it came from the cache."""
    hits = cache.cache_info().hits
    value = cache(*args)
    assert (
        cache.cache_info().hits == hits + 1
    ), "the value was not cached by the traced call"
    return value


def test_caches_first_filled_under_jit_hold_concrete_arrays(rng, cold_caches):
    """Caches of precomputed arrays filled while tracing hold concrete arrays, not tracers, so
    that later eager calls work and agree with the compiled ones."""
    L_c, N_c, spin = 6, 2, 1
    flm = generate_flm(rng, L_c, spin=spin)
    samples = ref.isht(flm, L_c, spin)
    flmn = generate_flmn(rng, L_c, N_c)
    angles = (0.3, 1.1, -0.4)
    rotated = jax.jit(rotation.rotate_flm)(flm, angles)
    forward = jax.jit(transforms.forward, static_argnames=("L", "spin", "method"))
    flm_jit = forward(samples, L=L_c, spin=spin, method="precompute")
    wigner_inverse = jax.jit(
        transforms.wigner_inverse, static_argnames=("L", "N", "method")
    )
    samples_jit = wigner_inverse(flmn, L=L_c, N=N_c, method="precompute")

    for cache in cold_caches:
        assert cache.cache_info().currsize > 0
    cached = [
        _cached_value(rotation.delta_matrices, L_c),
        _cached_value(transforms.spherical_kernel, L_c, spin, True, "mw", False, None),
        _cached_value(transforms.wigner_kernel, L_c, N_c, False, "mw", False, None),
    ]
    tracer = getattr(jax.core, "Tracer", ())
    for value in cached:
        assert not isinstance(value, tracer)
        np.asarray(value)  # a leaked tracer cannot be converted

    assert ref.rel_err(rotation.rotate_flm(flm, angles), rotated) < 1e-12
    assert ref.rel_err(rotated, ref.rotate_flm(flm, angles)) < ref.TOL
    assert (
        ref.rel_err(
            transforms.forward(samples, L_c, spin, method="precompute"), flm_jit
        )
        < 1e-12
    )
    assert ref.rel_err(flm_jit, flm) < ref.TOL
    assert (
        ref.rel_err(
            transforms.wigner_inverse(flmn, L_c, N_c, method="precompute"), samples_jit
        )
        < 1e-12
    )
    assert ref.rel_err(samples_jit, ref.wigner_inv(flmn, L_c, N_c)) < ref.TOL


def test_jit_convolve_precompute_with_cold_caches(rng, cold_caches):
    """s2conv.convolve compiles with the precompute backend before any kernel is cached."""
    L_c, n_out, s_in = 6, 3, 1
    block = ref.random_block(L_c, rng, n_out)
    kernel = ref.block_to_kernel(block, n_out)
    flm = generate_flm(rng, L_c, spin=s_in)
    x = ref.isht(flm, L_c, s_in)
    compiled = convolve_jit(
        x, kernel, L=L_c, s_in=s_in, s_out="G", N_out=n_out, method="precompute"
    )
    eager = s2conv.convolve(x, kernel, L_c, s_in, "G", N_out=n_out, method="precompute")
    assert ref.rel_err(compiled, eager) < 1e-12
    expected = ref.wigner_inv(ref.theorem_v2(block, flm, s_in, "G", n_out), L_c, n_out)
    assert ref.rel_err(compiled, expected) < ref.TOL


def test_jit_harmonic_operators(rng):
    """The harmonic operators compile with the spins and band-limits static."""
    block = ref.random_block(L, rng, N)
    kernel = ref.block_to_kernel(block, N)
    flm = generate_flm(rng, L, spin=2)
    spin_to_group = jax.jit(lifted.spin_to_group, static_argnames=("s_in", "N_out"))
    assert (
        ref.rel_err(
            spin_to_group(flm, kernel, s_in=2, N_out=N),
            ref.theorem_v2(block, flm, 2, "G", N),
        )
        < ref.TOL
    )
    evaluate = jax.jit(s2conv.utils.evaluate, static_argnames=("spin",))
    theta, phi = ref.random_points(rng, 5)
    assert (
        ref.rel_err(evaluate(flm, 2, theta, phi), ref.eval_spin(flm, 2, theta, phi))
        < ref.TOL
    )


# ------------------------------------------------------------------------------ gradients


@pytest.mark.parametrize("method", ["jax", "precompute"])
@pytest.mark.parametrize(("s_in", "s_out"), CONFIGS)
def test_grad_convolve_with_respect_to_the_input(rng, method, s_in, s_out):
    """jax.grad of |C x|^2 with respect to the complex samples x: its real inner product with a
    direction v is the directional derivative, compared with central differences."""
    x, v, _, kernel = _input_and_kernel(rng, s_in)
    n_out = N if s_out == "G" else None

    def loss(y):
        return jnp.sum(
            jnp.abs(
                s2conv.convolve(y, kernel, L, s_in, s_out, N_out=n_out, method=method)
            )
            ** 2
        )

    grad = jax.grad(loss)(jnp.asarray(x))
    assert np.all(np.isfinite(grad))
    directional = float(jnp.real(jnp.sum(grad * v)))
    difference = float(_central_difference(lambda t: loss(x + t * v)))
    assert abs(directional - difference) < 1e-8 * abs(difference)


@pytest.mark.parametrize("method", ["jax", "precompute"])
@pytest.mark.parametrize(("s_in", "s_out"), [(1, "G"), ("G", -2)])
def test_grad_convolve_with_respect_to_the_kernel(rng, method, s_in, s_out):
    x, _, _, kernel = _input_and_kernel(rng, s_in)
    direction = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    n_out = N if s_out == "G" else None

    def loss(t):
        out = s2conv.convolve(
            x, kernel + t * direction, L, s_in, s_out, N_out=n_out, method=method
        )
        return jnp.sum(jnp.abs(out) ** 2)

    grad = float(jax.grad(loss)(0.0))
    assert abs(grad - float(_central_difference(loss))) < 1e-8 * abs(grad)


@pytest.mark.parametrize("method", ["jax", "precompute"])
def test_grad_convolve_multiplier(rng, method):
    """Gradients with respect to the multiplier c_l of a spin-changing convolution."""
    s_in, s_out = 2, 0
    f = ref.isht(generate_flm(rng, L, spin=s_in), L, s_in)
    c = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    dc = rng.standard_normal(L)

    def loss(t):
        out = s2conv.convolve_multiplier(f, c + t * dc, L, s_in, s_out, method=method)
        return jnp.sum(jnp.abs(out) ** 2)

    grad = float(jax.grad(loss)(0.0))
    assert abs(grad - float(_central_difference(loss))) < 1e-8 * abs(grad)


HARMONIC_OPERATORS = {
    "spin_to_spin": lambda f, F, k: lifted.spin_to_spin(f, k, 1, -2),
    "spin_to_group": lambda f, F, k: lifted.spin_to_group(f, k, 1, N),
    "group_to_spin": lambda f, F, k: lifted.group_to_spin(F, k, 0),
    "group_to_group": lambda f, F, k: lifted.group_to_group(F, k, N),
    "directional": lambda f, F, k: classical.directional_convolution(f, k[N - 1], 1, N),
    "synthesis": lambda f, F, k: classical.directional_synthesis(F, k[N - 1], 1),
    "eth": lambda f, F, k: physics.eth(f, 1),
    "compose": lambda f, F, k: kernels.compose(k, kernels.flipped(k, N)),
    "rotate": lambda f, F, k: rotation.rotate_flm(F, (0.3, 1.0, 2.0)),
}


@pytest.mark.parametrize("name", HARMONIC_OPERATORS)
def test_grad_through_harmonic_operators(rng, name):
    """Reverse and forward mode through the harmonic operators agree with central differences
    along a direction perturbing the input and the kernel together."""
    operator = HARMONIC_OPERATORS[name]
    flm, dflm = generate_flm(rng, L, spin=1), generate_flm(rng, L, spin=1)
    flmn, dflmn = generate_flmn(rng, L, N), generate_flmn(rng, L, N)
    kernel = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    dkernel = ref.block_to_kernel(ref.random_block(L, rng, N), N)

    def loss(t):
        return jnp.sum(
            jnp.abs(operator(flm + t * dflm, flmn + t * dflmn, kernel + t * dkernel))
            ** 2
        )

    difference = float(_central_difference(loss))
    assert abs(float(jax.grad(loss)(0.0)) - difference) < 1e-8 * abs(difference)
    _, tangent = jax.jvp(loss, (0.0,), (1.0,))
    assert abs(float(tangent) - difference) < 1e-8 * abs(difference)


def test_forward_mode_through_a_linear_operator_is_the_operator(rng):
    """jax.jvp of a harmonic operator, linear in its input, is the operator applied to the tangent."""
    kernel = ref.block_to_kernel(ref.random_block(L, rng, N), N)
    flm, tangent = generate_flm(rng, L, spin=-1), generate_flm(rng, L, spin=-1)
    _, out = jax.jvp(
        lambda f: lifted.spin_to_group(f, kernel, -1, N), (flm,), (tangent,)
    )
    assert ref.rel_err(out, lifted.spin_to_group(tangent, kernel, -1, N)) < ref.TOL


# ---------------------------------------------------------------- rotations and Euler angles


@pytest.mark.parametrize(
    "angles", [(0.4, 1.2, -2.1), (2.0, 0.0, 0.5), (-1.0, np.pi, 3.0)]
)
def test_grad_with_respect_to_euler_angles(rng, angles):
    """Gradients of Re <g, R(alpha, beta, gamma) f> are finite and agree with finite differences,
    also at the coordinate singularities beta = 0 and pi of the Euler angles."""
    flm, glm = generate_flm(rng, L, spin=2), generate_flm(rng, L, spin=2)

    def loss(a):
        return jnp.real(
            jnp.sum(jnp.conj(glm) * rotation.rotate_flm(flm, (a[0], a[1], a[2])))
        )

    a0 = jnp.asarray(angles)
    grad = np.asarray(jax.grad(loss)(a0))
    assert np.all(np.isfinite(grad))
    h = 1e-4
    for i in range(3):
        e = np.eye(3)[i]
        stencil = (
            -loss(a0 + 2 * h * e)
            + 8 * loss(a0 + h * e)
            - 8 * loss(a0 - h * e)
            + loss(a0 - 2 * h * e)
        ) / (12 * h)
        assert abs(grad[i] - float(stencil)) < 1e-8 * max(1.0, abs(float(stencil)))


def test_norm_is_rotation_invariant(rng):
    """|R f|^2 = |f|^2: the gradient of the norm with respect to the Euler angles vanishes."""
    flm = generate_flm(rng, L, spin=-1)

    def norm(a):
        return jnp.sum(jnp.abs(rotation.rotate_flm(flm, (a[0], a[1], a[2]))) ** 2)

    a0 = jnp.asarray([0.7, 2.2, -0.3])
    total = float(np.sum(np.abs(flm) ** 2))
    assert abs(float(norm(a0)) - total) < ref.TOL * total
    grad = np.asarray(jax.grad(norm)(a0))
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) < 1e-10 * total


def test_grad_of_evaluation_at_a_point(rng):
    """d/d theta of a field evaluated off the grid, against the reference evaluation."""
    flm = generate_flm(rng, L, spin=0)
    theta, phi, h = 1.1, 0.4, 1e-4

    def value(t):
        return jnp.real(s2conv.utils.evaluate(flm, 0, t, phi))

    stencil = [
        np.real(ref.eval_spin(flm, 0, theta + k * h, phi)) for k in (-2, -1, 1, 2)
    ]
    derivative = (stencil[0] - 8 * stencil[1] + 8 * stencil[2] - stencil[3]) / (12 * h)
    assert abs(float(jax.grad(value)(theta)) - derivative) < 1e-8 * max(
        1.0, abs(derivative)
    )


def test_kernels_are_differentiable(rng):
    """Gradients flow through kernel constructors, e.g. a lifted directional kernel psi."""
    psi = generate_flm(rng, L, spin=0)
    direction = generate_flm(rng, L, spin=0)
    flm = generate_flm(rng, L, spin=0)

    def loss(t):
        out = lifted.spin_to_group(flm, kernels.lifted(psi + t * direction, 0), 0, N)
        return jnp.sum(jnp.abs(out) ** 2)

    grad = float(jax.grad(loss)(0.0))
    assert abs(grad - float(_central_difference(loss))) < 1e-8 * abs(grad)
