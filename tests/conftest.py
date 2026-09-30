"""
Shared configuration of the s2conv test suite.

64-bit precision is enabled here, before s2conv, s2fft or the reference module are
imported by any test module, since the transforms are exact only to single precision
otherwise.  The ``ssht`` marker skips tests of the ``"jax_ssht"`` backend when pyssht
(or ducc0, which s2fft asks it to use) is not installed.
"""

import importlib.util
import os
import zlib

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HAVE_SSHT = all(
    importlib.util.find_spec(name) is not None for name in ("pyssht", "ducc0")
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line(
        "markers", "ssht: test requiring pyssht (the jax_ssht backend)"
    )
    # s2fft's NumPy recursions, used by the references, divide by zero harmlessly
    config.addinivalue_line("filterwarnings", "ignore::RuntimeWarning:s2fft")


def pytest_runtest_setup(item):
    if item.get_closest_marker("ssht") is not None and not HAVE_SSHT:
        pytest.skip("the jax_ssht backend needs pyssht and ducc0")


@pytest.fixture
def rng(request):
    """A NumPy generator with a fixed seed derived from the test's node id.

    Every test, and every parameter set of a test, draws its own reproducible numbers.
    """
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))
