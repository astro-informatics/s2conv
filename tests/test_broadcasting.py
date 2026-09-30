"""
Broadcasting over leading axes.

The harmonic operators, kernel constructors and conversions are documented to act on
arrays with leading batch (or channel) axes.  Each is applied to a batch and compared
with a Python loop over the batch; the kernel may be shared or batched with the input.
"""

import numpy as np
import pytest
import reference as ref
from s2fft.utils.signal_generator import generate_flm, generate_flmn

from s2conv import kernels, lifting, utils
from s2conv.convolutions import classical, diagonal, lifted, physics, section

L = 8
N = 3
B = 3


def _flm(rng, spin=0):
    """A batch of B random spin fields, [B, L, 2L - 1]."""
    return np.stack([generate_flm(rng, L, spin=spin) for _ in range(B)])


def _flmn(rng, n=N):
    """A batch of B random functions on SO(3), [B, 2n - 1, L, 2L - 1]."""
    return np.stack([generate_flmn(rng, L, n) for _ in range(B)])


def _kernel(rng, n=N):
    """A random kernel with orders |n|, |k| < n, [2n - 1, L, 2L - 1]."""
    return ref.block_to_kernel(ref.random_block(L, rng, n), n)


def _kernels(rng, n=N):
    """A batch of B random kernels."""
    return np.stack([_kernel(rng, n) for _ in range(B)])


def _complex(rng, *shape):
    """Complex Gaussian weights of the given shape."""
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _assert_broadcasts(operator, *batched):
    """operator applied to the batch equals the stack of its applications to each element."""
    out = operator(*batched)
    looped = [operator(*(np.asarray(x)[i] for x in batched)) for i in range(B)]
    if isinstance(out, tuple):
        for j, component in enumerate(out):
            assert (
                ref.rel_err(component, np.stack([item[j] for item in looped])) < ref.TOL
            )
    else:
        assert np.shape(out)[0] == B
        assert ref.rel_err(out, np.stack(looped)) < ref.TOL


# Each case maps a random generator to (operator, batched arguments).
CASES = {
    # lifting (Sec. IV)
    "lift": lambda r: (lambda f: lifting.lift(f, 1, N), _flm(r, 1)),
    "project": lambda r: (lambda F: lifting.project(F, -2), _flmn(r)),
    "spin_components": lambda r: (lifting.spin_components, _flmn(r)),
    "from_spin_components": lambda r: (
        lifting.from_spin_components,
        np.stack(
            [
                np.stack([generate_flm(r, L, spin=s) for s in range(-(N - 1), N)])
                for _ in range(B)
            ]
        ),
    ),
    # the four operators with a shared kernel (Theorem V.2)
    "spin_to_spin": lambda r, k=None: (
        lambda f: lifted.spin_to_spin(f, k, 1, -2),
        _flm(r, 1),
    ),
    "spin_to_group": lambda r, k=None: (
        lambda f: lifted.spin_to_group(f, k, 1, N),
        _flm(r, 1),
    ),
    "group_to_spin": lambda r, k=None: (
        lambda F: lifted.group_to_spin(F, k, 2),
        _flmn(r),
    ),
    "group_to_group": lambda r, k=None: (
        lambda F: lifted.group_to_group(F, k, 2),
        _flmn(r),
    ),
    # weight forms (Theorem V.5)
    "spin_multiplier": lambda r: (
        lambda f, c: lifted.spin_multiplier(f, c, 2, 0),
        _flm(r, 2),
        _complex(r, B, L),
    ),
    "spin_to_group_weights": lambda r: (
        lambda f, u: lifted.spin_to_group_weights(f, u, 0),
        _flm(r),
        _complex(r, B, 2 * N - 1, L),
    ),
    "group_to_spin_weights": lambda r: (
        lambda F, v: lifted.group_to_spin_weights(F, v, -1),
        _flmn(r),
        _complex(r, B, 2 * N - 1, L),
    ),
    "group_to_group_weights": lambda r: (
        lifted.group_to_group_weights,
        _flmn(r),
        _complex(r, B, L, 2 * N - 1, 2 * N + 1),
    ),
    # classical convolutions (Tables I, II)
    "axisymmetric": lambda r: (
        lambda f, p: classical.axisymmetric_convolution(f, p, 1),
        _flm(r, 1),
        _flm(r, 1),
    ),
    "left": lambda r: (classical.left_convolution, _flm(r), _flm(r)),
    "directional": lambda r: (
        lambda f, p: classical.directional_convolution(f, p, 2, N),
        _flm(r, 2),
        _flm(r, 2),
    ),
    "directional_synthesis": lambda r: (
        lambda F, p: classical.directional_synthesis(F, p, 1),
        _flmn(r),
        _flm(r, 1),
    ),
    "axisymmetric_synthesis": lambda r: (
        lambda w, p: classical.axisymmetric_synthesis(w, p, 2),
        _flm(r),
        _flm(r, 2),
    ),
    # physics (Proposition VI.8)
    "eth": lambda r: (lambda f: physics.eth(f, 1), _flm(r, 1)),
    "ethbar": lambda r: (lambda f: physics.ethbar(f, -1), _flm(r, -1)),
    "eb_decomposition": lambda r: (physics.eb_decomposition, _flm(r, 2), _flm(r, -2)),
    "eb_decomposition_real": lambda r: (physics.eb_decomposition, _flm(r, 2)),
    "eb_to_polarisation": lambda r: (physics.eb_to_polarisation, _flm(r), _flm(r)),
    "kaiser_squires": lambda r: (physics.kaiser_squires, _flm(r)),
    "kaiser_squires_inverse": lambda r: (physics.kaiser_squires_inverse, _flm(r, 2)),
    # diagonal and section operators (Sec. VII)
    "harmonic_multiplication": lambda r: (
        diagonal.harmonic_multiplication,
        _flm(r),
        _flm(r),
    ),
    "sifting": lambda r: (diagonal.sifting_convolution, _flm(r), _flm(r)),
    "section_components": lambda r, k=None: (
        lambda f: section.section_components(f, k, 0, N),
        _flm(r),
    ),
    "standard_correlation_fourier": lambda r, k=None: (
        lambda f: section.standard_correlation_fourier(f, np.asarray(k)[N - 1]),
        _flm(r),
    ),
    # utilities
    "rotate_flm": lambda r: (
        lambda f: utils.rotate_flm(f, (0.3, 1.2, -0.7)),
        _flm(r, 2),
    ),
    "rotate_flmn": lambda r: (
        lambda F: utils.rotate_flmn(F, (0.3, 1.2, -0.7)),
        _flmn(r),
    ),
    "conjugate": lambda r: (lambda f: utils.conjugate(f, 2), _flm(r, 2)),
    "evaluate": lambda r: (
        lambda f: utils.evaluate(f, -1, np.array([0.3, 2.0]), np.array([1.0, 4.0])),
        _flm(r, -1),
    ),
    "spin_fourier": lambda r: (lambda f: utils.spin_fourier(f, 1), _flm(r, 1)),
    "evaluate_so3": lambda r: (
        lambda F: utils.evaluate_so3(F, 0.4, 1.3, 2.2),
        _flmn(r),
    ),
    # kernel constructors and conversions (Table II, Theorem V.5)
    "kernels.lifted": lambda r: (lambda p: kernels.lifted(p, -1, N), _flm(r, -1)),
    "kernels.left": lambda r: (lambda h: kernels.left(h, N), _flm(r)),
    "kernels.flipped": lambda r: (lambda k: kernels.flipped(k, 4), _kernels(r)),
    "kernels.multiplier": lambda r: (
        lambda c: kernels.multiplier(c, 1, -2),
        _complex(r, B, L),
    ),
    "kernels.compose": lambda r: (kernels.compose, _kernels(r), _kernels(r)),
    "kernels.to_multiplier": lambda r: (
        lambda k: kernels.to_multiplier(k, 1, 2),
        _kernels(r),
    ),
    "kernels.to_column": lambda r: (lambda k: kernels.to_column(k, -2, 4), _kernels(r)),
    "kernels.to_row": lambda r: (lambda k: kernels.to_row(k, 1, 4), _kernels(r)),
    "kernels.to_block": lambda r: (lambda k: kernels.to_block(k, 2, 4), _kernels(r)),
    "kernels.from_column": lambda r: (
        lambda u: kernels.from_column(u, 1),
        _complex(r, B, 2 * N - 1, L),
    ),
    "kernels.from_row": lambda r: (
        lambda v: kernels.from_row(v, -1),
        _complex(r, B, 2 * N - 1, L),
    ),
    "kernels.from_block": lambda r: (
        kernels.from_block,
        _complex(r, B, L, 2 * N - 1, 2 * N + 1),
    ),
}

WITH_KERNEL = {
    "spin_to_spin",
    "spin_to_group",
    "group_to_spin",
    "group_to_group",
    "section_components",
    "standard_correlation_fourier",
}


@pytest.mark.parametrize("name", CASES)
def test_operator_broadcasts_over_leading_axes(rng, name):
    if name in WITH_KERNEL:
        operator, *batched = CASES[name](rng, _kernel(rng))
    else:
        operator, *batched = CASES[name](rng)
    _assert_broadcasts(operator, *batched)


CHANNELWISE = {
    "spin_to_spin": (lambda f, k: lifted.spin_to_spin(f, k, 1, -2), 1, False),
    "spin_to_group": (lambda f, k: lifted.spin_to_group(f, k, 1, N), 1, False),
    "group_to_spin": (lambda F, k: lifted.group_to_spin(F, k, 2), None, True),
    "group_to_group": (lambda F, k: lifted.group_to_group(F, k, 2), None, True),
}


@pytest.mark.parametrize("name", CHANNELWISE)
def test_batched_kernels_act_channel_by_channel(rng, name):
    """A batch of kernels applied to a batch of inputs acts element by element (Theorem V.2)."""
    operator, spin, group_input = CHANNELWISE[name]
    x = _flmn(rng) if group_input else _flm(rng, spin)
    _assert_broadcasts(operator, x, _kernels(rng))


def test_several_leading_axes(rng):
    """Two leading axes, e.g. batch and channel, are kept."""
    kernel = _kernel(rng)
    flm = np.stack([_flm(rng, 1) for _ in range(2)])
    out = lifted.spin_to_group(flm, kernel, 1, N)
    assert out.shape == (2, B, 2 * N - 1, L, 2 * L - 1)
    for i in range(2):
        for j in range(B):
            assert (
                ref.rel_err(out[i, j], lifted.spin_to_group(flm[i, j], kernel, 1, N))
                < ref.TOL
            )
