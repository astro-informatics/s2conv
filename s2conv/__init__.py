r"""
s2conv: differentiable and accelerated spherical convolutions with JAX.

s2conv implements the generalised lifted spherical convolution, in which a spin-:math:`s`
field is lifted to the rotation group, convolved there with a kernel :math:`\Psi` and
projected back to the sphere through the adjoint of the lift for a possibly different
spin :math:`s'`, and the spherical convolutions of the literature that it unifies.  The
harmonic transforms are those of s2fft, which warns on import if JAX is not using 64-bit
precision.
"""

from . import convolutions, kernels, lifting, transforms, utils
from .convolutions import (
    classical,
    diagonal,
    group_convolution,
    group_to_group,
    group_to_spin,
    lifted,
    lifted_convolution,
    physics,
    sampled,
    section,
    spin_multiplier,
    spin_to_group,
    spin_to_spin,
)
from .convolutions.sampled import convolve, convolve_multiplier
from .lifting import lift, project
from .transforms import forward, inverse, wigner_forward, wigner_inverse

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - only without an installed version file
    __version__ = "unknown"
