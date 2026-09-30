Equivariance tests
==================

s2conv measures the rotational equivariance of an operator :math:`T` with the protocol of
Cobb et al. (2021) and Ocampo et al. (2023), implemented in :mod:`s2conv.utils.equivariance`.
For random band-limited signals :math:`f_i`, whose harmonic coefficients are drawn from the
standard normal distribution, and rotations :math:`g_j` drawn from the Haar measure, it
computes the relative errors

.. math::

    \epsilon_{ij} = \frac{\lVert T(\mathcal{R}_{g_j} f_i) - \mathcal{R}_{g_j}(T f_i)\rVert}{\lVert \mathcal{R}_{g_j}(T f_i)\rVert},

in the :math:`L^2` norm on the sphere or the rotation group, and averages them over all pairs.

The numerator is the equivariance defect: the difference between rotating the input and
rotating the output.  The denominator :math:`\lVert\mathcal{R}_{g_j}(Tf_i)\rVert = \lVert Tf_i\rVert` is
the rotated output, which does not depend on the rotation.  Cobb et al. and Ocampo et al.
normalise instead by the output of the rotated input, :math:`\lVert T(\mathcal{R}_{g_j}f_i)\rVert`.
The two coincide for an equivariant operator and differ at relative order of the error itself
otherwise, so s2conv reports both, as ``rotated_output`` and ``output_of_rotated_input``.

Measuring an operator
---------------------

.. code-block:: python

    import jax

    jax.config.update("jax_enable_x64", True)

    import s2conv
    from s2conv import kernels
    from s2conv.utils import equivariance as eq

    L = 32
    key_psi, key_f, key_g = jax.random.split(jax.random.key(0), 3)
    kernel = kernels.lifted(eq.random_signals(key_psi, 1, L, spin=2)[0], 2)
    signals = eq.random_signals(key_f, 100, L, spin=2)
    rotations = eq.random_rotations(key_g, 100)

    errors = eq.equivariance_errors(
        lambda f: s2conv.lifted_convolution(f, kernel, 2, "G", 5), signals, rotations
    )
    eq.summarise(errors)  # mean, standard deviation and maximum in each normalisation

Three functions cover the kinds of output:

* :func:`~s2conv.utils.equivariance.equivariance_errors`, for operators between harmonic or
  Wigner coefficients.  Rotations act on the coefficients, by left translation on the rotation
  group, and the norms follow from Parseval's relation, exactly for band-limited outputs.
* :func:`~s2conv.utils.equivariance.equivariance_errors_on_points`, for outputs given as
  functions of :math:`(\theta, \varphi)`, compared in real space with the spin phase of the
  rotated output on a Gauss-Legendre grid.
* :func:`~s2conv.utils.equivariance.equivariance_errors_of_section`, for evaluations on a
  section, such as standard correlation, which are not band-limited; they are compared through
  their spin components, which is much faster than point by point.

All three are JIT compiled and vectorised over rotations.

The benchmark
-------------

The script ``benchmarks/equivariance.py`` applies the protocol to every operator of the package
at the scale of the papers: 100 signals and 100 rotations per operator by default.  The operators
are the four lifted convolutions with random kernels, the classical convolutions and syntheses,
the operators of mathematical physics, and, for contrast, the section-evaluated and
harmonic-diagonal operators, which are not equivariant.  For each it also reports a reduction
error, the largest relative difference between the operator's closed form and its form as a
lifted convolution, and the run time end to end, with the transforms.  Run it from the
repository root, with s2conv installed:

.. code-block:: bash

    python benchmarks/equivariance.py -L 16 32 64 -o equivariance.json --latex equivariance.tex
    python benchmarks/equivariance.py -L 16 --precision single

It prints a markdown table for each band-limit and writes the results, with the system
information, as JSON, and a LaTeX table with the columns operator, reduction error,
equivariance error and time.  ``python benchmarks/equivariance.py --help`` lists the options,
including the numbers of signals and rotations, the operators to run and the band-limit of the
timings.

Results
-------

On an NVIDIA GB10 GPU, with 100 signals and 100 rotations per operator, the run at
:math:`L = 16`, 32 and 64 with timings at :math:`L = 128` takes about six minutes.  In double
precision the equivariant operators give mean errors of 3 to :math:`4\times10^{-16}` at
:math:`L = 16`, growing slowly to :math:`1.2\times10^{-15}` at :math:`L = 64`, and reduction errors
below :math:`3\times10^{-16}`.  The section-evaluated and harmonic-diagonal operators give errors
of 1.3 to 1.4, of order one, as they should.  In single precision the equivariant operators give
about :math:`2\times10^{-7}`, in line with the single-precision errors of Cobb et al.

On NVIDIA GPUs, JAX computes float32 matrix products in TF32 by default, which limits
single-precision errors to about :math:`4\times10^{-4}`.  The benchmark therefore computes them
in full float32 unless ``--matmul-precision default`` is given; double precision is not affected.
To run s2conv in single precision on a GPU with full accuracy, set
``jax.config.update("jax_default_matmul_precision", "highest")``.

Tests
-----

The test suite checks the protocol itself in ``tests/test_equivariance_metric.py``: exact zero
error for the identity, agreement of the two normalisations for equivariant operators, and
agreement between the three kinds of comparison.  Its slow tests, marked ``slow``, run every
operator of the benchmark at 20 signals and 20 rotations; continuous integration runs them on
pushes to the main branch, weekly and on demand, but not on pull requests.
