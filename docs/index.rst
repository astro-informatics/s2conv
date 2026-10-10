s2conv
======

**Differentiable and accelerated spherical convolutions with JAX.**

s2conv implements the *generalised lifted spherical convolution* of the article
*Unifying spherical convolutions through lifting* (J. D. McEwen, 2026, in
preparation).
A spin-:math:`s` field on the sphere is *lifted* to the rotation group
:math:`\mathrm{SO}(3)`, convolved there with a kernel :math:`\Psi`, and projected back
to the sphere through the adjoint of the lift for a possibly different spin
:math:`s'`.
Every rotation-equivariant spherical convolution considered in the article is obtained
by a choice of the kernel :math:`\Psi`.

The package is built on `s2fft <https://github.com/astro-informatics/s2fft>`_, which
provides the harmonic transforms, and is written in `JAX <https://docs.jax.dev>`_, so
it is differentiable and runs on GPUs.

The four lifted operators
-------------------------

Let :math:`\Lambda_s` denote the lift of a spin-:math:`s` field to the rotation group
:math:`G = \mathrm{SO}(3)`, let :math:`\Lambda_{s'}^{\dagger}` denote the adjoint of the
lift for spin :math:`s'`, which projects a function on :math:`G` back to the sphere, and
let :math:`\star` denote the convolution on :math:`G` with the kernel :math:`\Psi`.
The kernel defines four operators, according to whether the input and the output are
fields on the sphere or functions on :math:`G`:

.. math::

   \mathcal{C}_{s \to s'}[\Psi] &= \Lambda_{s'}^{\dagger}\,(\Psi \star \Lambda_s\,\cdot\,), \\
   \mathcal{C}_{s \to G}[\Psi] &= \Psi \star \Lambda_s\,\cdot\,, \\
   \mathcal{C}_{G \to s'}[\Psi] &= \Lambda_{s'}^{\dagger}\,(\Psi \star \cdot\,), \\
   \mathcal{C}_{G \to G}[\Psi] &= \Psi \star \cdot\,.

In harmonic space each operator reads part of the Wigner coefficients
:math:`\Psi^{\ell}` of the kernel, which form a matrix at each degree :math:`\ell`
(Theorem V.2 of the article):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Operator
     - Maps
     - Reads from the kernel
   * - :math:`\mathcal{C}_{s \to s'}`
     - a spin-:math:`s` field to a spin-:math:`s'` field
     - the single entry :math:`(-s', -s)` of :math:`\Psi^{\ell}`
   * - :math:`\mathcal{C}_{s \to G}`
     - a spin-:math:`s` field to a function on :math:`G`
     - the column :math:`-s`
   * - :math:`\mathcal{C}_{G \to s'}`
     - a function on :math:`G` to a spin-:math:`s'` field
     - the row :math:`-s'`
   * - :math:`\mathcal{C}_{G \to G}`
     - a function on :math:`G` to a function on :math:`G`
     - the whole block

Convolutions of the literature as kernels
-----------------------------------------

The convolutions of the literature differ only in their kernels.
They include

* the axisymmetric, left (Driscoll–Healy), directional and group convolutions;
* the spin directional and spin axisymmetric convolutions;
* Haar synthesis;
* spin-changing convolutions, such as those of Esteves et al. and Tessore et al.;
* the spin raising and lowering operators, the E/B decomposition and the
  Kaiser–Squires operator.

Beyond equivariance
-------------------

Sec. VII of the article gives further anisotropic constructions, which keep the output
on the sphere but are not lifted convolutions in general and are not fully rotation
equivariant.
s2conv implements these too:

* the standard correlation and the commutative anisotropic convolution, which are
  evaluations of a lifted convolution on a section of the rotation group over the
  sphere;
* harmonic multiplication and sifting, which are harmonic-diagonal multipliers.

Implementation
--------------

Spin spherical harmonic and Wigner transforms are computed by s2fft, with a choice of
backend: ``"jax"``, ``"precompute"``, ``"jax_ssht"`` or ``"numpy"``
(see :mod:`s2conv.transforms`).
The modules of the package are described in the :doc:`API documentation <api/index>`.

Citation
--------

The article is in preparation.
Please see the file ``CITATION.cff`` in the repository for how to cite s2conv.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User guide

   user_guide/install
   user_guide/equivariance
   user_guide/scaling
   user_guide/notebooks

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/index
