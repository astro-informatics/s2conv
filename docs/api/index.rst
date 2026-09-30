:html_theme.sidebar_secondary.remove:

API
===

Automatically generated documentation of the ``s2conv`` package.
Each function is documented on the page of the module that defines it; the main ones are
also available from the top-level ``s2conv`` namespace.
The modules are as follows.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Contents
   * - :mod:`s2conv.transforms`
     - Spin spherical harmonic and Wigner transforms through s2fft, with a choice of
       backend
   * - :mod:`s2conv.lifting`
     - The lift :math:`\Lambda_s` and its adjoint projection
   * - :mod:`s2conv.kernels`
     - Kernel constructors and conversions
   * - :mod:`s2conv.convolutions.lifted`
     - The four lifted operators (Theorem V.2) and their weight forms (Theorem V.5)
   * - :mod:`s2conv.convolutions.classical`
     - The named convolutions of the literature
   * - :mod:`s2conv.convolutions.physics`
     - The spin raising and lowering operators, the E/B decomposition and the
       Kaiser–Squires operator
   * - :mod:`s2conv.convolutions.section`
     - Section evaluation: the standard correlation and the commutative anisotropic
       convolution
   * - :mod:`s2conv.convolutions.diagonal`
     - Harmonic multiplication and sifting
   * - :mod:`s2conv.convolutions.sampled`
     - The operators applied to samples of signals
   * - :mod:`s2conv.utils.rotation`
     - Wigner d- and D-functions and rotations
   * - :mod:`s2conv.utils.evaluation`
     - Evaluation of band-limited fields at arbitrary points
   * - :mod:`s2conv.utils.indexing`
     - Index conventions for harmonic and Wigner coefficients

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Modules

   transforms
   lifting
   kernels
   convolutions/index
   utils/index
