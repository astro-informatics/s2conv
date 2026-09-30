Installation
============

s2conv is a pure-Python package that requires Python 3.11 or later, and is tested on
Python 3.11 and 3.12.  Its dependencies, NumPy, JAX and
`s2fft <https://github.com/astro-informatics/s2fft>`_, are installed automatically.
Python 3.13 awaits the next release of s2fft: version 1.4.0 fails to import on it,
because its compiled extension is built for Python 3.12 only.

Installing from source
----------------------

s2conv is currently installed from source.
Clone the repository and install the package with pip:

.. code-block:: bash

   git clone https://github.com/astro-informatics/s2conv.git
   cd s2conv
   pip install .

The version of the package is determined from the git history and tags by
`setuptools-scm <https://setuptools-scm.readthedocs.io>`_, so install from a clone of
the repository rather than from an archive of the source files, which has no git
history and is given a placeholder version.

To modify the code, install the package in editable mode instead:

.. code-block:: bash

   pip install -e .

You can check the installation by importing the package:

.. code-block:: bash

   python -c "import s2conv"

Optional dependencies
---------------------

Further dependencies are available as extras, which can be combined:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Extra
     - Installs
     - Needed for
   * - ``tests``
     - pytest and pytest-cov
     - running the test suite
   * - ``docs``
     - Sphinx, pydata-sphinx-theme and sphinx-copybutton
     - building this documentation
   * - ``notebooks``
     - marimo and matplotlib
     - running the notebook in the ``notebooks`` directory
   * - ``ssht``
     - pyssht and ducc0
     - the ``"jax_ssht"`` backend of the transforms

For example, to install the package with the test dependencies and the dependencies
of the ``"jax_ssht"`` backend:

.. code-block:: bash

   pip install ".[tests,ssht]"

GPUs and precision
------------------

s2conv runs on whichever devices JAX finds.
To use a GPU, install a GPU-enabled build of JAX by following the
`JAX installation instructions <https://docs.jax.dev/en/latest/installation.html>`_.

s2fft warns that single precision has a dramatic effect on the accuracy of the
transforms, even at moderate band-limits.
We therefore recommend enabling double precision in JAX before importing s2fft or
s2conv:

.. code-block:: python

   import jax

   jax.config.update("jax_enable_x64", True)

   import s2conv
