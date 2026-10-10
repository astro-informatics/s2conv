<div style="text-align: center;" align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/astro-informatics/s2conv/main/docs/assets/s2conv_logo_dark.png">
  <img width="98" alt="s2conv logo - a hexagon-tiled sphere with the fibre of frames over one point, its rim coloured by phase" src="https://raw.githubusercontent.com/astro-informatics/s2conv/main/docs/assets/s2conv_logo.png">
</picture>

# S2CONV: differentiable and accelerated spherical convolutions

[![Tests status](https://github.com/astro-informatics/s2conv/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2conv/actions/workflows/tests.yml)
[![Linting status](https://github.com/astro-informatics/s2conv/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2conv/actions/workflows/linting.yml)
[![Documentation status](https://github.com/astro-informatics/s2conv/actions/workflows/docs.yml/badge.svg?branch=main)](https://astro-informatics.github.io/s2conv/)
[![Codecov](https://codecov.io/gh/astro-informatics/s2conv/branch/main/graph/badge.svg)](https://codecov.io/gh/astro-informatics/s2conv)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![All Contributors](https://img.shields.io/github/all-contributors/astro-informatics/s2conv?color=ee8449&style=flat-square)](#contributors-)
[![Linter](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

`S2CONV` is a JAX package implementing the **generalised lifted spherical convolution** and the spherical convolutions it unifies, for scalar fields, spin fields and functions on the rotation group.  All operators are differentiable, can be JIT compiled and run on GPUs, and are exact for band-limited signals.  The harmonic transforms are those of [`S2FFT`](https://github.com/astro-informatics/s2fft).

## The generalised lifted convolution 🌐

A spin-$s$ field on the sphere is lifted to the rotation group $\mathrm{SO}(3)$ through the irreducible representation of spin $s$ of the stabiliser, convolved on $\mathrm{SO}(3)$ with a kernel $\Psi$, and projected back to the sphere through the adjoint of the lift for a possibly different spin $s'$:

$$
\mathcal{C}_{s\to s'}[\Psi] = \Lambda_{s'}^{\dagger}\,(\Psi\star\Lambda_s\,\cdot\,),\qquad
\mathcal{C}_{s\to G}[\Psi] = \Psi\star\Lambda_s\,\cdot\,,\qquad
\mathcal{C}_{G\to s'}[\Psi] = \Lambda_{s'}^{\dagger}\,(\Psi\star\cdot\,),\qquad
\mathcal{C}_{G\to G}[\Psi] = \Psi\star\cdot\,.
$$

In harmonic space each operator reads a single entry, column, row or the whole block of the Wigner coefficients of the kernel.  For example, a spin $s \to s'$ convolution multiplies each harmonic coefficient by a single Wigner coefficient of the kernel,

$$
\left(\mathcal{C}_{s\to s'}[\Psi]\,{}_sf\right)_{\ell m} = (-1)^{s+s'}\,{}_sf_{\ell m}\,\Psi^{\ell *}_{-s',-s},
$$

and every bounded rotation-equivariant map between spin fields is of this form.  The convolutions of the literature differ only in their kernels.

| Convolution | Lifted form | `s2conv` |
|:--|:--|:--|
| Axisymmetric, scalar and spin | $(2\pi)^{-1/2}\mathcal{C}_{s\to0}[\Lambda_s\psi]$ | `classical.axisymmetric_convolution` |
| Left convolution (Driscoll & Healy) | $\mathcal{C}_{0\to0}[\Psi_h]$ | `classical.left_convolution` |
| Directional, scalar and spin | $\mathcal{C}_{s\to G}[\Lambda_s\psi]$ | `classical.directional_convolution` |
| Haar synthesis | $\mathcal{C}_{G\to s}[\widetilde{\Lambda_s\psi}]$ | `classical.directional_synthesis` |
| Group convolution | $\mathcal{C}_{G\to G}[\Psi]$ | `group_convolution` |
| Spin-changing (e.g. Esteves et al., Tessore et al., spin needlets) | $\mathcal{C}_{s\to s'}[\Psi_T]$ | `spin_multiplier`, `kernels.multiplier` |
| Steerable kernels as tuples of spins | $\sum_{\lvert s'\rvert<N}\Lambda_{s'}\mathcal{C}_{s\to s'}[\Psi]$ | `lifting.spin_components` |
| $\eth$, $\bar\eth$, $E/B$, Kaiser–Squires | $\mathcal{C}_{s\to s'}[\Psi_T]$ | `physics.eth`, `physics.ethbar`, `physics.eb_decomposition`, `physics.kaiser_squires` |

The non-equivariant constructions that keep the output of a directional kernel on the sphere are also provided, as evaluations of a lifted convolution on a section of the bundle (standard correlation, commutative anisotropic convolution; `section`) or as harmonic-diagonal operators (harmonic multiplication, sifting; `diagonal`).

## Transform backends ⚡

`S2CONV` delegates the spin spherical harmonic and Wigner transforms to `S2FFT`, through `s2conv.transforms`, with a choice of backend:

| `method` | Transforms | Spins and azimuthal band-limits | Memory |
|:--|:--|:--|:--|
| `"jax"` | JAX, recursions on the fly | $\lvert s\rvert, N < 8$ | $\mathcal{O}(L^2)$ |
| `"precompute"` | JAX, cached precomputed kernels | any | $\mathcal{O}(L^3)$, $\mathcal{O}(NL^3)$ |
| `"jax_ssht"` | JAX frontend to SSHT, CPU only | any | $\mathcal{O}(L^2)$ |
| `"numpy"` | NumPy reference | $\lvert s\rvert, N < 8$ | $\mathcal{O}(L^2)$ |

All backends except `"numpy"` are differentiable in reverse mode (`jax.grad`, `jax.vjp`); `"jax"` and `"precompute"` can be JIT compiled and run on GPUs.  Forward mode (`jax.jvp`, `jax.jacfwd`) works through `"precompute"`, and through `"jax"` with `S2FFT` newer than 1.4.0, but not through `"jax_ssht"`.  The harmonic operators themselves support both modes.

## Installation 💻

`S2CONV` requires Python 3.11 or later, and is tested on Python 3.11 and 3.12.  Python 3.13 awaits the next release of `S2FFT`: version 1.4.0 fails to import on it, because its compiled extension is built for Python 3.12 only.  From a clone of this repository,

```bash
pip install .
```

installs it with its dependencies, [`JAX`](https://github.com/jax-ml/jax) and `S2FFT`.  The optional extras are `tests` (the test suite), `notebooks` (the Marimo notebook), `docs` (the documentation) and `ssht` (the `"jax_ssht"` backend).  For GPU support install a CUDA build of JAX, following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Usage 🚀

Enable double precision before using `S2CONV`, as for `S2FFT`:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import s2conv
from s2fft.utils.signal_generator import generate_flm

L = 32
rng = np.random.default_rng(0)
f = s2conv.inverse(generate_flm(rng, L, spin=2), L, spin=2)  # a spin-2 field on the MW grid
```

A spin $2 \to 1$ convolution with a Gaussian multiplier, directly or through its canonical kernel and the generalised lifted convolution:

```python
el = np.arange(L)
c = np.exp(-0.5 * el * (el + 1) * 0.05**2)
g = s2conv.convolve_multiplier(f, c, L, s_in=2, s_out=1)

kernel = s2conv.kernels.multiplier(c, s_in=2, s_out=1)
g = s2conv.convolve(f, kernel, L, s_in=2, s_out=1)
```

A directional convolution with a steerable spin-2 kernel, whose output on the rotation group is a tuple of spin fields:

```python
flm = s2conv.forward(f, L, spin=2)
psi = generate_flm(rng, L, spin=2) * (np.abs(np.arange(-(L - 1), L)) < 4)  # orders |n| < 4
F = s2conv.classical.directional_convolution(flm, psi, spin=2, N=4)  # Wigner coefficients
F_samples = s2conv.wigner_inverse(F, L, N=4)  # samples on SO(3)
spins = s2conv.lifting.spin_components(F)  # spin fields s' = -3, ..., 3
```

The operators are JAX functions, so they can be compiled and differentiated:

```python
@jax.jit
def loss(c):
    return jnp.sum(jnp.abs(s2conv.convolve_multiplier(f, c, L, s_in=2, s_out=1)) ** 2)

grad = jax.grad(loss)(jnp.asarray(c))
```

Harmonic-space operators (`s2conv.lifted`, `s2conv.classical`, `s2conv.physics`, `s2conv.diagonal`) act on coefficients in the `S2FFT` layout and broadcast over leading axes, so that batches, channels or spins can be carried as leading dimensions or mapped with `jax.vmap`.

## Notebooks 📓

Two Marimo notebooks run in the browser on [molab](https://molab.marimo.io), which installs their dependencies from the list at the top of each file:

- [`notebooks/lifted_convolutions.py`](https://github.com/astro-informatics/s2conv/blob/main/notebooks/lifted_convolutions.py) demonstrates each convolution with `S2CONV` and checks it numerically against the closed forms of the article, a literal evaluation on a sampling of the rotation group and the real-space definitions.  [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/astro-informatics/s2conv/blob/main/notebooks/lifted_convolutions.py)
- [`notebooks/real_fields.py`](https://github.com/astro-informatics/s2conv/blob/main/notebooks/real_fields.py) applies the lifted convolutions to real spin fields of the Earth and the sky.  [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/astro-informatics/s2conv/blob/main/notebooks/real_fields.py)

To run them locally, install the package with the notebook extras,

```bash
pip install ".[notebooks]"
marimo edit notebooks/lifted_convolutions.py
```

or let Marimo install the dependencies in an isolated environment with [uv](https://docs.astral.sh/uv/), `marimo edit --sandbox notebooks/lifted_convolutions.py`.

## Equivariance tests 📐

`s2conv.utils.equivariance` measures the rotational equivariance of any operator with the protocol of Cobb et al. (2021) and Ocampo et al. (2023).  It takes the mean, over random band-limited signals and Haar-random rotations, of the relative error ‖T(𝓡f) − 𝓡(Tf)‖ / ‖𝓡(Tf)‖, and reports alongside it the normalisation ‖T(𝓡f)‖ of those papers.  The script `benchmarks/equivariance.py` applies this to every operator of the package at the scale of the papers, 100 signals × 100 rotations.  It tabulates the equivariance errors, reduction errors and timings as markdown, JSON and LaTeX, each stamped with the git commit of the code that produced it:

```bash
python benchmarks/equivariance.py -L 16 32 64 -o equivariance.json --latex equivariance.tex
```

In double precision the equivariant operators give errors of about 10⁻¹⁵ (about 2 × 10⁻⁷ in single precision), while the section-evaluated and harmonic-diagonal operators give errors of order one.  The [user guide](https://astro-informatics.github.io/s2conv/user_guide/equivariance.html) has the details.

The script `benchmarks/scaling.py` times the four lifted convolutions against band-limit L, with the azimuthal band-limit N fixed, on one CPU core, and plots the times against an L³ reference, the scaling of their transforms ([user guide](https://astro-informatics.github.io/s2conv/user_guide/scaling.html)):

```bash
python benchmarks/scaling.py -o scaling.json --plot scaling.pdf > scaling.md
```

## Tests 🚦

```bash
pip install .[tests]
pytest
```

The tests marked `slow` run every operator of the equivariance benchmark at 20 signals × 20 rotations; `pytest -m "not slow"` skips them.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jasonmcewen.org"><img src="https://avatars.githubusercontent.com/u/3181701?v=4?s=100" width="100px;" alt="Jason McEwen"/><br /><sub><b>Jason McEwen</b></sub></a><br /><a href="https://github.com/astro-informatics/s2conv/commits?author=jasonmcewen" title="Code">💻</a> <a href="#ideas-jasonmcewen" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-jasonmcewen" title="Research">🔬</a> <a href="https://github.com/astro-informatics/s2conv/commits?author=jasonmcewen" title="Documentation">📖</a> <a href="#maintenance-jasonmcewen" title="Maintenance">🚧</a> <a href="https://github.com/astro-informatics/s2conv/pulls?q=is%3Apr+reviewed-by%3Ajasonmcewen" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification, and we welcome contributions of any kind; see [CONTRIBUTING.md](https://github.com/astro-informatics/s2conv/blob/main/CONTRIBUTING.md).

## Attribution 📚

`S2CONV` implements the construction of the article *Unifying spherical convolutions through lifting* (J. D. McEwen, in preparation); please cite it, and `S2FFT`, if you use this code:

```
@unpublished{mcewen2026unifying,
   author      = "Jason D. McEwen",
   title       = "Unifying spherical convolutions through lifting",
   year        = "2026",
   note        = "In preparation"
}

@article{price:s2fft,
   author      = "Matthew A. Price and Jason D. McEwen",
   title       = "Differentiable and accelerated spherical harmonic and Wigner transforms",
   journal     = "Journal of Computational Physics",
   year        = "2024",
   volume      = "510",
   pages       = "113109",
   eprint      = "arXiv:2311.14670",
   doi         = "10.1016/j.jcp.2024.113109"
}
```

## License 📝

`S2CONV` is released under the MIT licence (see [LICENCE.txt](https://github.com/astro-informatics/s2conv/blob/main/LICENCE.txt)).
