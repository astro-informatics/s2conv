r"""
Lifted convolutions of sampled signals.

These wrap the harmonic operators of :mod:`s2conv.convolutions.lifted` between forward and
inverse transforms, so that samples on the sphere or rotation group go in and come out.
With the ``"jax"`` or ``"precompute"`` backends they can be JIT compiled, with the
band-limits, spins, sampling and method static, for example::

    convolve_jit = jax.jit(
        s2conv.convolve,
        static_argnames=("L", "s_in", "s_out", "N_out", "sampling", "method"),
    )

With ``"precompute"`` the cached transform kernels, of sizes :math:`\mathcal{O}(L^3)` and
:math:`\mathcal{O}(NL^3)`, become constants of the compiled program: it then runs fastest,
but takes longer to compile at large band-limits than with ``"jax"``, which computes
the Wigner functions on the fly.
"""

from s2conv import transforms
from s2conv.convolutions.lifted import lifted_convolution, spin_multiplier


def convolve(
    x,
    kernel,
    L: int,
    s_in,
    s_out,
    N_out: int = None,
    sampling: str = "mw",
    method: str = "jax",
    nside: int = None,
):
    r"""
    The generalised lifted convolution :math:`\mathcal{C}_{s\to s'}[\Psi]` of sampled signals (Definition V.1).

    Args:
        x: Samples of a spin-:math:`s` field on the sphere or, if ``s_in == "G"``,
            samples on the rotation group with shape ``[2N_in - 1, ...]``.
        kernel: The kernel :math:`\Psi`, ``[2N_k - 1, L, 2L - 1]`` (see :mod:`s2conv.kernels`).
        L (int): Harmonic band-limit.
        s_in (int or "G"): Input spin, or ``"G"`` for the rotation group.
        s_out (int or "G"): Output spin, or ``"G"`` for the rotation group.
        N_out (int, optional): Azimuthal band-limit of an output on the rotation group.
            Defaults to :math:`L`.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        method (str, optional): Transform backend, one of
            :data:`s2conv.transforms.METHODS`. Defaults to "jax".
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        Samples of the output on the sphere or, if ``s_out == "G"``, on the rotation group.

    """
    if s_in == "G":
        N_in = (x.shape[0] + 1) // 2
        x_lm = transforms.wigner_forward(x, L, N_in, sampling, method, False, nside)
    else:
        x_lm = transforms.forward(x, L, s_in, sampling, method, False, nside)
    if s_out == "G":
        N_out = L if N_out is None else N_out
        y = lifted_convolution(x_lm, kernel, s_in, s_out, N_out)
        return transforms.wigner_inverse(y, L, N_out, sampling, method, False, nside)
    y = lifted_convolution(x_lm, kernel, s_in, s_out)
    return transforms.inverse(y, L, s_out, sampling, method, False, nside)


def convolve_multiplier(
    f,
    c,
    L: int,
    s_in: int,
    s_out: int,
    sampling: str = "mw",
    method: str = "jax",
    nside: int = None,
):
    r"""
    A spin-changing convolution with multiplier :math:`c_\ell` of a sampled field (Proposition VI.5).

    Args:
        f: Samples of a spin-:math:`s` field on the sphere.
        c: Multiplier :math:`c_\ell`, ``[L]``.
        L (int): Harmonic band-limit.
        s_in (int): Input spin :math:`s`.
        s_out (int): Output spin :math:`s'`.
        sampling (str, optional): Sampling scheme. Defaults to "mw".
        method (str, optional): Transform backend. Defaults to "jax".
        nside (int, optional): HEALPix resolution, if ``sampling="healpix"``.

    Returns:
        Samples of the spin-:math:`s'` output on the sphere.

    """
    flm = transforms.forward(f, L, s_in, sampling, method, False, nside)
    return transforms.inverse(
        spin_multiplier(flm, c, s_in, s_out), L, s_out, sampling, method, False, nside
    )
