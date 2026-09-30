r"""
Equivariance errors under random rotations, following the protocols of the literature.

For an operator :math:`T`, a signal :math:`f` and a rotation :math:`\mathcal{R}`, the equivariance
defect is :math:`\lVert T(\mathcal{R}f) - \mathcal{R}(Tf)\rVert`, the :math:`L^2` distance between
rotating the input and rotating the output.  It is reported relative to either of two norms:

- the rotated output, :math:`\lVert\mathcal{R}(Tf)\rVert = \lVert Tf\rVert`, which does not depend on
  the rotation (``rotated_output``, the default in this package); or
- the output of the rotated input, :math:`\lVert T(\mathcal{R}f)\rVert`, as in Cobb et al. (2021)
  and Ocampo et al. (2023) (``output_of_rotated_input``).

The two agree for an equivariant operator, and differ at relative order of the error itself
otherwise.  Averaged over random signals and rotations, for instance 100 of each as in Cobb
et al., they give the mean relative equivariance error

.. math::

    \epsilon(T) = \frac{1}{N_f N_g} \sum_{i=1}^{N_f} \sum_{j=1}^{N_g}
    \frac{\lVert T(\mathcal{R}_{g_j} f_i) - \mathcal{R}_{g_j}(T f_i)\rVert}{\lVert \mathcal{R}_{g_j}(T f_i)\rVert}.

:func:`equivariance_errors` handles operators between harmonic coefficients: spin spherical
harmonic coefficients ``[L, 2L - 1]`` or Wigner coefficients ``[2N - 1, L, 2L - 1]``.  Rotations
act on both through the Wigner matrices, on the first Wigner index for functions on the rotation
group (left translation), and the norms follow from Parseval's relation, exactly for band-limited
outputs.  :func:`equivariance_errors_on_points` handles operators whose outputs are not
band-limited, such as evaluations on a section, by comparing the outputs in real space on a
Gauss-Legendre grid.  Both are JIT compiled and vectorised over rotations.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from s2conv.utils.indexing import lm_mask, wigner_mask
from s2conv.utils.rotation import (
    euler_angles,
    random_rotation,
    rotate_flm,
    rotation_matrix,
    section,
)


class EquivarianceErrors(NamedTuple):
    r"""
    Relative equivariance errors, ``[n_signals, n_rotations]``, in both normalisations.

    ``rotated_output`` holds :math:`\lVert T(\mathcal{R}f) - \mathcal{R}(Tf)\rVert / \lVert\mathcal{R}(Tf)\rVert`,
    and ``output_of_rotated_input`` holds
    :math:`\lVert T(\mathcal{R}f) - \mathcal{R}(Tf)\rVert / \lVert T(\mathcal{R}f)\rVert`, the normalisation
    of Cobb et al. (2021) and Ocampo et al. (2023).
    """

    rotated_output: jnp.ndarray
    output_of_rotated_input: jnp.ndarray


def random_signals(key, n: int, L: int, spin: int = 0, N: int = None) -> jnp.ndarray:
    r"""
    Random band-limited signals with standard normal harmonic coefficients.

    The real and imaginary parts of each coefficient are independent standard normal
    variables; coefficients outside the band-limit, or with :math:`\ell < |s|`, are zero.

    Args:
        key: A JAX random key.
        n (int): Number of signals.
        L (int): Harmonic band-limit.
        spin (int, optional): Spin :math:`s` of fields on the sphere. Defaults to 0.
        N (int, optional): If given, draw functions on the rotation group with azimuthal
            band-limit :math:`N` instead of fields on the sphere.

    Returns:
        jnp.ndarray: Harmonic coefficients ``[n, L, 2L - 1]`` or Wigner coefficients
        ``[n, 2N - 1, L, 2L - 1]``.

    """
    if N is None:
        shape, mask = (n, L, 2 * L - 1), lm_mask(L, spin)
    else:
        shape, mask = (n, 2 * N - 1, L, 2 * L - 1), wigner_mask(L, N)
    key_re, key_im = jax.random.split(key)
    values = jax.random.normal(key_re, shape) + 1j * jax.random.normal(key_im, shape)
    return values * mask


def random_rotations(key, n: int) -> tuple:
    r"""
    Euler angles of rotations drawn from the Haar measure on the rotation group.

    Args:
        key: A JAX random key.
        n (int): Number of rotations.

    Returns:
        tuple: Arrays :math:`(\alpha, \beta, \gamma)`, each of shape ``[n]``.

    """
    return jax.vmap(random_rotation)(jax.random.split(key, n))


def quadrature(L: int) -> tuple:
    r"""
    Gauss-Legendre nodes on the sphere and their weights, flattened.

    The :math:`L` nodes in :math:`\cos\theta` and :math:`2L - 1` equispaced longitudes, the
    Gauss-Legendre sampling of s2fft, integrate :math:`|g|^2` exactly for a spin field
    :math:`g` of band-limit :math:`L`; the nodes avoid the poles.

    Args:
        L (int): Band-limit up to which the squared modulus is integrated exactly.

    Returns:
        tuple: Colatitudes, longitudes and weights, each of shape ``[L (2L - 1)]``.

    """
    thetas, phis, weights = _gl_grid(L)
    theta, phi = np.meshgrid(thetas, phis, indexing="ij")
    return theta.ravel(), phi.ravel(), np.repeat(weights, len(phis))


def _gl_grid(L: int) -> tuple:
    """Colatitudes (ascending, as in s2fft), longitudes and the weights of each colatitude row."""
    x, w = np.polynomial.legendre.leggauss(L)
    order = np.argsort(np.arccos(x))
    n = 2 * L - 1
    return np.arccos(x)[order], 2 * np.pi * np.arange(n) / n, w[order] * 2 * np.pi / n


def _parseval_weights(shape: tuple):
    """Weights of Parseval's relation for coefficients of the given (unbatched) shape."""
    if len(shape) == 2:
        return 1.0
    if len(shape) == 3:
        return ((2 * np.arange(shape[-2]) + 1) / (8 * np.pi**2))[:, None]
    raise ValueError(
        "Operator outputs must be harmonic coefficients [L, 2L - 1] or Wigner coefficients "
        f"[2N - 1, L, 2L - 1]; got shape {shape}."
    )


def _relative_errors(lhs, rhs, weights) -> tuple:
    """The defect ||lhs - rhs|| relative to ||rhs|| and to ||lhs||, with the given weights."""

    def norm(x):
        return jnp.sqrt(jnp.sum(weights * jnp.abs(x) ** 2))

    defect = norm(lhs - rhs)

    def ratio(reference):
        safe = jnp.where(reference > 0, reference, 1.0)
        return jnp.where(
            reference > 0, defect / safe, jnp.where(defect > 0, jnp.inf, 0.0)
        )

    return ratio(norm(rhs)), ratio(norm(lhs))


def _collect(
    per_signal: Callable, signals, rotations, batch_size: int
) -> EquivarianceErrors:
    """Evaluate per_signal(x, alpha, beta, gamma) signal by signal, over rotation batches."""
    alpha, beta, gamma = (jnp.atleast_1d(jnp.asarray(angle)) for angle in rotations)
    rows_output, rows_input = [], []
    for x in signals:
        parts = [
            per_signal(
                x, *(angle[i : i + batch_size] for angle in (alpha, beta, gamma))
            )
            for i in range(0, alpha.shape[0], batch_size)
        ]
        rows_output.append(jnp.concatenate([p[0] for p in parts]))
        rows_input.append(jnp.concatenate([p[1] for p in parts]))
    return EquivarianceErrors(jnp.stack(rows_output), jnp.stack(rows_input))


def equivariance_errors(
    operator: Callable, signals, rotations, batch_size: int = 100
) -> EquivarianceErrors:
    r"""
    Relative equivariance errors of an operator between harmonic coefficients.

    For every signal :math:`f_i` and rotation :math:`g_j`, compares :math:`T(\mathcal{R}_{g_j} f_i)`
    with :math:`\mathcal{R}_{g_j}(T f_i)`, rotating harmonic coefficients by the Wigner matrices
    (Eq. 10), Wigner coefficients by left translation, and measuring both in :math:`L^2` by
    Parseval's relation, :math:`\sum_{\ell m} |f_{\ell m}|^2` on the sphere and
    :math:`\sum_\ell \frac{2\ell+1}{8\pi^2} \sum_{mn} |F^{\ell}_{mn}|^2` on the rotation group.

    Args:
        operator: Function mapping one input, ``[L, 2L - 1]`` or ``[2N - 1, L, 2L - 1]``, to one
            output of either form.  It must be a JAX function of its input.
        signals: Inputs stacked along a leading axis, as from :func:`random_signals`.
        rotations: Euler angles :math:`(\alpha, \beta, \gamma)`, arrays of equal length, as from
            :func:`random_rotations`.
        batch_size (int, optional): Number of rotations evaluated together, which bounds
            memory. Defaults to 100.

    Returns:
        EquivarianceErrors: The errors, ``[n_signals, n_rotations]``, in both normalisations.

    """
    signals = jnp.asarray(signals)
    weights = _parseval_weights(jax.eval_shape(operator, signals[0]).shape)

    @jax.jit
    def per_signal(x, alpha, beta, gamma):
        tx = operator(x)

        def pair(a, b, g):
            lhs = operator(rotate_flm(x, (a, b, g)))
            return _relative_errors(lhs, rotate_flm(tx, (a, b, g)), weights)

        return jax.vmap(pair)(alpha, beta, gamma)

    return _collect(per_signal, signals, rotations, batch_size)


def equivariance_errors_on_points(
    evaluator: Callable,
    signals,
    rotations,
    spin_out: int = 0,
    L_quad: int = None,
    batch_size: int = 10,
) -> EquivarianceErrors:
    r"""
    Relative equivariance errors of an operator with outputs on the sphere, compared in real space.

    For outputs that are not band-limited, such as evaluations on a section (Proposition VII.1),
    compares :math:`T(\mathcal{R}f)(\omega)` with
    :math:`(\mathcal{R}\,Tf)(\omega) = e^{-\mathrm{i}s'\vartheta}(Tf)(R^{-1}\omega)` (Eq. 12), with
    :math:`\vartheta` the third Euler angle of :math:`R^{-1}s(\omega)`, and integrates on a
    Gauss-Legendre grid (:func:`quadrature`).

    Args:
        evaluator: Function mapping one input ``[L, 2L - 1]`` to a function of
            ``(theta, phi)`` returning the output, a spin-:math:`s'` field in the coordinate
            frame, at those points.  It must be written with JAX.
        signals: Inputs ``[n, L, 2L - 1]``.
        rotations: Euler angles :math:`(\alpha, \beta, \gamma)`, arrays of equal length.
        spin_out (int, optional): Spin :math:`s'` of the output. Defaults to 0.
        L_quad (int, optional): Band-limit of the quadrature.  Defaults to :math:`2L`, which
            oversamples band-limited outputs by two.
        batch_size (int, optional): Number of rotations evaluated together. Defaults to 10.

    Returns:
        EquivarianceErrors: The errors, ``[n_signals, n_rotations]``, in both normalisations.

    """
    signals = jnp.asarray(signals)
    L_quad = 2 * signals.shape[-2] if L_quad is None else L_quad
    theta, phi, weights = quadrature(L_quad)
    frames = section(theta, phi)

    @jax.jit
    def per_signal(x, alpha, beta, gamma):
        tx = evaluator(x)

        def pair(a, b, g):
            lhs = evaluator(rotate_flm(x, (a, b, g)))(theta, phi)
            ra, rb, rc = euler_angles(rotation_matrix(a, b, g).T @ frames)
            rhs = jnp.exp(-1j * spin_out * rc) * tx(rb, ra)
            return _relative_errors(lhs, rhs, weights)

        return jax.vmap(pair)(alpha, beta, gamma)

    return _collect(per_signal, signals, rotations, batch_size)


def equivariance_errors_of_section(
    kernel,
    s_in: int,
    N: int,
    signals,
    rotations,
    chi: Callable = None,
    L_quad: int = None,
    batch_size: int = 10,
    method: str = "jax",
) -> EquivarianceErrors:
    r"""
    Relative equivariance errors of an evaluation on a section, from its spin components.

    The evaluation on a section :math:`E_\sigma[\Psi]\,{}_sf = \sum_{|s'|<N}
    e^{-\mathrm{i}s'\chi_\sigma}(2\pi)^{-1/2}\mathcal{C}_{s\to s'}[\Psi]\,{}_sf` (Proposition VII.1,
    Eq. 63) is a scalar field in the coordinate frame that is not band-limited.  Both
    :math:`E_\sigma(\mathcal{R}f)` and :math:`\mathcal{R}(E_\sigma f)` are sums of band-limited spin
    components weighted by frame phases: the components of :math:`\mathcal{R}f`, and the rotated
    components of :math:`f`, since :math:`g(R^{-1}\omega) = e^{\mathrm{i}s'\vartheta}(\mathcal{R}g)(\omega)`
    (Eq. 12).  Both are therefore evaluated by inverse spin transforms on the Gauss-Legendre grid of
    band-limit ``L_quad``, rather than point by point, and the errors follow by quadrature.
    The components are band-limited, so their values on the grid are exact; only the quadrature
    of the error, which is not band-limited, is approximate.  This is much faster than
    :func:`equivariance_errors_on_points` at large band-limits.

    Standard correlation is ``chi=None`` with the kernel :math:`\Lambda_0\psi`; the commutative
    anisotropic convolution of Sadeghi et al. is ``chi = lambda theta, phi: np.pi - phi`` with the
    kernel :math:`\Lambda_0 h^*`.

    Args:
        kernel: The kernel :math:`\Psi` (see :mod:`s2conv.kernels`).
        s_in (int): Spin :math:`s` of the input.
        N (int): Azimuthal band-limit: the spin components :math:`|s'| < N` are kept.
        signals: Inputs ``[n, L, 2L - 1]``.
        rotations: Euler angles :math:`(\alpha, \beta, \gamma)`, arrays of equal length.
        chi (callable, optional): The twist :math:`\chi_\sigma(\theta, \varphi)` of the section,
            written with JAX; ``None`` for the coordinate section.
        L_quad (int, optional): Band-limit of the quadrature grid, at least :math:`L`.
            Defaults to :math:`L`; a larger value refines the quadrature of the error.
        batch_size (int, optional): Number of rotations evaluated together. Defaults to 10.
        method (str, optional): Transform backend, one of :data:`s2conv.transforms.METHODS`.
            Defaults to ``"jax"``, accurate for :math:`N \le 8`.

    Returns:
        EquivarianceErrors: The errors, ``[n_signals, n_rotations]``, in both normalisations.

    """
    # imported here, since s2conv.convolutions imports s2conv.utils
    from s2conv import transforms
    from s2conv.convolutions.section import section_components

    signals = jnp.asarray(signals)
    L = signals.shape[-2]
    L_quad = L if L_quad is None else L_quad
    thetas, phis, row_weights = _gl_grid(L_quad)
    theta, phi = np.meshgrid(thetas, phis, indexing="ij")
    weights = row_weights[:, None]
    frames = section(theta, phi)
    spins = [int(sp) for sp in range(-(N - 1), N)]
    phase = np.asarray(spins)[:, None, None]
    out_phase = 1.0 if chi is None else jnp.exp(-1j * phase * chi(theta, phi)[None])

    def on_grid(components):
        padded = jnp.zeros(
            components.shape[:-2] + (L_quad, 2 * L_quad - 1), components.dtype
        )
        padded = padded.at[..., :L, L_quad - L : L_quad + L - 1].set(components)
        return jnp.stack(
            [
                transforms.inverse(padded[k], L_quad, sp, "gl", method)
                for k, sp in enumerate(spins)
            ]
        )

    @jax.jit
    def per_signal(x, alpha, beta, gamma):
        comps = section_components(x, kernel, s_in, N)

        def pair(a, b, g):
            rotated = section_components(rotate_flm(x, (a, b, g)), kernel, s_in, N)
            lhs = jnp.sum(out_phase * on_grid(rotated), axis=0)
            ra, rb, rc = euler_angles(rotation_matrix(a, b, g).T @ frames)
            chi_rotated = 0.0 if chi is None else chi(rb, ra)
            rhs_phase = jnp.exp(-1j * phase * (chi_rotated - rc)[None])
            rhs = jnp.sum(rhs_phase * on_grid(rotate_flm(comps, (a, b, g))), axis=0)
            return _relative_errors(lhs, rhs, weights)

        return jax.vmap(pair)(alpha, beta, gamma)

    return _collect(per_signal, signals, rotations, batch_size)


def summarise(errors: EquivarianceErrors) -> dict:
    """
    Mean, standard deviation and maximum of the errors in each normalisation.

    Args:
        errors (EquivarianceErrors): Errors from :func:`equivariance_errors` or
            :func:`equivariance_errors_on_points`.

    Returns:
        dict: ``{normalisation: {"mean": ..., "std": ..., "max": ...}}``, with float values.

    """
    return {
        name: {
            "mean": float(jnp.mean(values)),
            "std": float(jnp.std(values)),
            "max": float(jnp.max(values)),
        }
        for name, values in errors._asdict().items()
    }
