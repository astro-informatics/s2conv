Convolutions
============

The generalised lifted convolution, and the spherical convolutions that it unifies.
The modules are as follows.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Contents
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

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Convolutions

   lifted
   classical
   physics
   section
   diagonal
   sampled
