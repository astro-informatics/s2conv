"""
The public interface of s2conv: exported names and the precision warning.
"""

import os
import subprocess
import sys

import s2conv
from s2conv import convolutions, kernels, lifting, transforms, utils
from s2conv.convolutions import classical, lifted, physics, sampled, section


def test_top_level_exports():
    assert s2conv.lift is lifting.lift
    assert s2conv.project is lifting.project
    assert s2conv.forward is transforms.forward
    assert s2conv.inverse is transforms.inverse
    assert s2conv.wigner_forward is transforms.wigner_forward
    assert s2conv.wigner_inverse is transforms.wigner_inverse
    assert s2conv.convolve is sampled.convolve
    assert s2conv.convolve_multiplier is sampled.convolve_multiplier
    assert s2conv.lifted_convolution is lifted.lifted_convolution
    for name in (
        "spin_to_spin",
        "spin_to_group",
        "group_to_spin",
        "group_to_group",
        "group_convolution",
        "spin_multiplier",
    ):
        assert getattr(s2conv, name) is getattr(lifted, name)
    for module in (classical, physics, section, kernels, utils):
        assert module.__name__.startswith("s2conv.")
    assert isinstance(s2conv.__version__, str)


def test_convolutions_exports():
    assert convolutions.group_convolution is convolutions.group_to_group
    assert convolutions.spin_changing_convolution is convolutions.spin_multiplier
    assert convolutions.standard_correlation is section.standard_correlation
    assert convolutions.eth is physics.eth
    assert utils.rotate_flmn is utils.rotate_flm


def test_warning_without_double_precision():
    """Importing s2conv with JAX in single precision warns (through s2fft) that precision is lost."""
    env = dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS="cpu")
    # the current interpreter on a literal command
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", "import s2conv"],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(s2conv.__file__)),
        check=True,
    )
    assert "64-bit precision" in result.stderr
