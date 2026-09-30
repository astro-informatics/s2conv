# Numerical companion to "Unifying spherical convolutions through lifting" (J. D. McEwen), built on the s2conv package.
# Install s2conv with the notebook extras from the repository root, pip install -e ".[notebooks]", then run
#     marimo edit notebooks/lifted_convolutions.py   (or: marimo run notebooks/lifted_convolutions.py)
# Transforms use s2fft's JAX backend through s2conv, on a GPU when JAX finds one.

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Lifted spherical convolutions")

with app.setup:
    # Imports and numerical configuration shared by every cell and function below.
    import functools
    import io
    import os
    import time
    import urllib.request
    from pathlib import Path

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # allocate GPU memory as needed

    import jax

    jax.config.update("jax_enable_x64", True)  # must precede the import of s2fft and s2conv

    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import s2conv
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle
    from s2conv import kernels, lifting, transforms
    from s2conv.convolutions import classical, diagonal, lifted, physics
    from s2conv.utils.evaluation import evaluate, evaluate_fourier, evaluate_so3
    from s2conv.utils.indexing import azimuthal_bandlimit, conjugate, lm_mask, m_axis
    from s2conv.utils.rotation import delta_matrices, euler_angles, rotate_flm, rotate_flmn, rotation_matrix, section, wigner_d
    from s2fft.recursions import risbo  # an independent recursion for the Wigner functions, used as a reference
    from s2fft.utils.signal_generator import generate_flm, generate_flmn

    # s2conv's transforms default to McEwen-Wiaux sampling and s2fft's JAX backend, which is JIT compiled and
    # accurate for spins and azimuthal band-limits N below 8.
    FOUR_PI = 4 * np.pi
    EIGHT_PI2 = 8 * np.pi**2
    L_CHECK = 12  # band-limit of the quadrature checks, which are exact at any band-limit
    N_CHECK = 5  # azimuthal band-limit of the Wigner transforms in the checks: orders |n| <= 4
    TOL = 1e-10  # a check passes when its relative error is below TOL
    N_EQ_FIELDS = 10  # random input fields per equivariance test (100 in Cobb et al., 20 in Ocampo et al.)
    N_EQ_ROTATIONS = 10  # Haar-random rotations per equivariance test (likewise 100 and 20)
    R_EARTH = 6.371e6  # radius of the Earth in metres, for the wind and ocean currents of Sec. 13

    # The equivariance tests call these s2conv functions thousands of times on small arrays, so they use them
    # JIT compiled: each call then runs as one compiled program rather than as a sequence of array operations.
    evaluate_jit = jax.jit(evaluate, static_argnames=("spin",))
    rotate_flm_jit = jax.jit(rotate_flm)
    rotate_flmn_jit = jax.jit(rotate_flmn)
    lifted_convolution_jit = jax.jit(s2conv.lifted_convolution, static_argnames=("s_in", "s_out", "N_out"))
    section_components_jit = jax.jit(s2conv.section.section_components, static_argnames=("s_in", "N"))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Unifying spherical convolutions through lifting
    ## A numerical companion

    This notebook demonstrates the **generalised lifted spherical convolution** of the article *Unifying spherical convolutions through lifting*, as implemented in the [`s2conv`](https://github.com/astro-informatics/s2conv) package, and recovers the convolutions of Tables I and II of the article, and the conversions of the accompanying notes, through the choice of a harmonic kernel.  A spin-$s$ field on the sphere is lifted to the rotation group, convolved on $\mathrm{SO}(3)$ with a kernel $\Psi$, and projected back to the sphere through the adjoint of the lift for a possibly different spin $s'$ (Definition V.1):

    $$
    \mathcal{C}_{s\to s'}[\Psi] = \Lambda_{s'}^{\dagger}\,(\Psi\star\Lambda_s\,\cdot\,),\qquad
    \mathcal{C}_{s\to G}[\Psi] = \Psi\star\Lambda_s\,\cdot\,,\qquad
    \mathcal{C}_{G\to s'}[\Psi] = \Lambda_{s'}^{\dagger}\,(\Psi\star\cdot\,),\qquad
    \mathcal{C}_{G\to G}[\Psi] = \Psi\star\cdot\,.
    $$

    In harmonic space each operator reads a single entry, column, row or the whole block of the Wigner coefficients $\Psi^{\ell}_{nm'}$ of the kernel (Theorem V.2 and Fig. 5), so that the convolutions of the literature differ only in their kernels (Tables I and II).  s2conv computes all transforms exactly for band-limited signals with [`s2fft`](https://github.com/astro-informatics/s2fft), which implements the sampling theorems of McEwen & Wiaux on the sphere and of McEwen *et al.* on the rotation group.  We use its JAX backend, which compiles the transforms and runs them on a GPU when one is available.  Its recursions are accurate for spins and azimuthal band-limits below 8, so wherever a transform is involved the notebook keeps spins to $|s|\le4$ and azimuthal band-limits to $N\le5$; purely harmonic computations are unrestricted.

    Every result is checked numerically in up to three independent ways:

    * against the **closed-form harmonic expressions** of the article (Theorem V.2, Tables I and II);
    * against a **literal evaluation**, in which fields are lifted on a sampling of $\mathrm{SO}(3)$, Wigner transformed, convolved and projected back by quadrature along the fibre;
    * against the **real-space definitions** of the classical convolutions (Definitions III.1–III.10), evaluated by exact quadrature with explicit rotations, including the spin phase of Eq. (12).

    The operators, kernels, transforms and utilities all come from s2conv; the notebook itself writes out only these independent evaluations, the test fields and the figures.

    Each table reports the maximum relative error of a check, except the equivariance table of Sec. 12, which reports the mean relative $L^2$ error of Cobb *et al.* (2021) and Ocampo *et al.* (2023).  A ✅ marks agreement to better than $10^{-10}$ or, for an operator that the article says should *not* agree (a non-equivariant operator, for instance), a clear disagreement.  Equation, definition, theorem, figure and table numbers refer to the article.
    """)
    return


@app.cell(hide_code=True)
def _():
    L_ui = mo.ui.slider(16, 64, step=8, value=32, show_value=True, label="band-limit $L$ of the demonstrations")
    seed_ui = mo.ui.number(start=0, stop=99999, value=2026, label="random seed")
    mo.vstack(
        [
            mo.md(
                "**Controls.** The band-limit and seed apply throughout.  The quadrature checks use their own "
                "band-limit $L = 12$, since they are exact at any band-limit.  The `s2fft` transforms run on "
                f"the JAX device `{jax.devices()[0]}`."
            ),
            mo.hstack([L_ui, seed_ui], justify="start", gap=2),
        ]
    )
    return L_ui, seed_ui


@app.cell
def _(L_ui, seed_ui):
    L = L_ui.value
    seed = seed_ui.value
    return L, seed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Harmonic analysis on the sphere and rotation group (Sec. II)

    We adopt the conventions of Sec. II, which are those of `s2fft`: $zyz$ Euler angles $\rho=(\alpha,\beta,\gamma)$ with $R(\alpha,\beta,\gamma)=R_z(\alpha)R_y(\beta)R_z(\gamma)$ (Eq. 1), Wigner functions $D^{\ell}_{mn}(\alpha,\beta,\gamma)=e^{-\mathrm{i}m\alpha}\,d^{\ell}_{mn}(\beta)\,e^{-\mathrm{i}n\gamma}$ (Eq. 6), the Wigner transform

    $$F^{\ell}_{mn}=\int_{\mathrm{SO}(3)}F\,D^{\ell}_{mn}\,\mathrm{d}\mu,\qquad F=\sum_\ell\frac{2\ell+1}{8\pi^2}\sum_{m,n}F^{\ell}_{mn}\,D^{\ell*}_{mn}\qquad\text{(Eq. 7)},$$

    and spin spherical harmonics ${}_sY_{\ell m}(\theta,\varphi)=(-1)^s\sqrt{\tfrac{2\ell+1}{4\pi}}\,d^{\ell}_{m,-s}(\theta)\,e^{\mathrm{i}m\varphi}$ (Eq. A7).  A rotation acts on the harmonic coefficients of every spin through the same Wigner matrices, $(\mathcal{R}_\rho\,{}_sf)_{\ell m}=\sum_n D^{\ell}_{mn}(\rho)\,{}_sf_{\ell n}$ (Eq. 10).  In real space this is $(\mathcal{R}_\rho\,{}_sf)(\omega)=e^{-\mathrm{i}s\vartheta}\,{}_sf(R_\rho^{-1}\omega)$, where $\vartheta$ is the third Euler angle of $R_\rho^{-1}R_\omega$ (Eq. 12); the phase is the character $\rho^{(s)}$ evaluated on the inverse twist (Proposition IV.10, Remark IV.11).

    **Arrays.**
    * Harmonic coefficients of any spin are stored as `flm[el, L-1+m]`.
    * Wigner coefficients use the `s2fft` layout `flmn[N-1+n, el, L-1+m]` $=F^{\ell}_{mn}$, with azimuthal band-limit $N$.
    * A kernel is a function on the rotation group, stored by its Wigner coefficients, `kernel[N-1+k, el, L-1+n]` $=\Psi^{\ell}_{nk}$ (Sec. 3); the figures and the closed forms written out in `theorem_v2` use the block `Psi[el, L-1+n, L-1+k]`.
    * Samples are `f[t, p]` at $(\theta_t,\varphi_p)$ on the sphere and `F[j, t, p]` at $(\alpha_p,\beta_t,\gamma_j)$ on the rotation group.

    The real-space checks need fields at arbitrary points.  For these `s2conv.utils.evaluation.evaluate` uses the factorisation $d^{\ell}_{mn}(\theta)=\mathrm{i}^{m-n}\sum_k\Delta^{\ell}_{km}\Delta^{\ell}_{kn}\,e^{-\mathrm{i}k\theta}$, with $\Delta^{\ell}=d^{\ell}(\pi/2)$ (Eq. B25), which turns a band-limited spin field into a finite two-dimensional Fourier series in $(\theta,\varphi)$.
    """)
    return


@app.function
def rel_err(a, b):
    """Maximum absolute difference, relative to the maximum absolute value of the reference b."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@app.function
def check_table(rows, first="Article"):
    """Render verification rows {paper, check, error[, expect]} as a markdown table."""
    lines = [f"| {first} | Numerical check | Rel. error | |", "|:--|:--|--:|:-:|"]
    for row in rows:
        agrees = row["error"] < TOL
        ok = agrees if row.get("expect", "agree") == "agree" else not agrees
        paper, check = (row[key].replace("|", "\\vert ") for key in ("paper", "check"))  # a bare | splits a cell
        lines.append(f"| {paper} | {check} | {row['error']:.1e} | {'✅' if ok else '❌'} |")
    return mo.md("\n".join(lines))


@app.function
def unit_vector(theta, phi):
    """Cartesian unit vectors of the points (theta, phi)."""
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


@app.function
def angles_of(v):
    """Colatitudes and longitudes of the unit vectors v[..., 3]."""
    return np.arccos(np.clip(v[..., 2], -1.0, 1.0)), np.mod(np.arctan2(v[..., 1], v[..., 0]), 2 * np.pi)


@app.function
def random_rotation(rng):
    """Euler angles of a Haar-random rotation."""
    return float(rng.uniform(0, 2 * np.pi)), float(np.arccos(rng.uniform(-1, 1))), float(rng.uniform(0, 2 * np.pi))


@app.cell
def _(seed):
    _rng = np.random.default_rng(seed)
    _L = L_CHECK
    _th, _ph = transforms.sphere_angles(_L)
    _T, _P = np.meshgrid(_th, _ph, indexing="ij")
    conventions_rows = []

    _errs = []
    for _s in (0, 1, -2, 3):
        _f = generate_flm(_rng, _L, spin=_s)
        _errs.append(rel_err(evaluate(_f, _s, _T, _P), transforms.inverse(_f, _L, _s)))
    conventions_rows.append(
        dict(
            paper="Eqs. (A7), (B25)",
            check="spin harmonics from the $\\Delta$ factorisation (`evaluate`) vs `transforms.inverse`, $s=0,1,-2,3$",
            error=max(_errs),
        )
    )

    _beta = _rng.uniform(0, np.pi)
    _d = np.zeros((_L, 2 * _L - 1, 2 * _L - 1))
    _plane = np.zeros((2 * _L - 1, 2 * _L - 1))
    for _el in range(_L):
        _plane = risbo.compute_full(_plane, _beta, _L, _el)
        _d[_el] = _plane
    conventions_rows.append(
        dict(paper="Eq. (B25)", check="$d^{\\ell}_{mn}(\\beta)$ from the factorisation vs the Risbo recursion", error=rel_err(wigner_d(_beta, _L), _d))
    )

    _errs = []
    for _s in (0, 2, -1):
        _f = generate_flm(_rng, _L, spin=_s)
        _rot = random_rotation(_rng)
        _pt, _pp = np.arccos(_rng.uniform(-0.95, 0.95, 50)), _rng.uniform(0, 2 * np.pi, 50)
        _a, _b, _c = euler_angles(rotation_matrix(*_rot).T @ section(_pt, _pp))
        _errs.append(rel_err(evaluate(rotate_flm(_f, _rot), _s, _pt, _pp), np.exp(-1j * _s * _c) * evaluate(_f, _s, _b, _a)))
    conventions_rows.append(
        dict(
            paper="Eqs. (10), (12), Prop. IV.10",
            check="rotation of spin fields: Wigner matrices on coefficients vs real space with the twist phase, $s=0,2,-1$",
            error=max(_errs),
        )
    )

    _errs = []
    for _s in (0, 2, -3):
        _f = generate_flm(_rng, _L, spin=_s)
        _errs.append(rel_err(transforms.wigner_forward(lifting.lift_samples(transforms.inverse(_f, _L, _s), _s, 4), _L, 4), lifting.lift(_f, _s, 4)))
    conventions_rows.append(
        dict(
            paper="Lemma IV.7(c), Eq. (39)",
            check="lift on the $\\mathrm{SO}(3)$ grid + `transforms.wigner_forward` vs the column $n=-s$, $s=0,2,-3$",
            error=max(_errs),
        )
    )

    _G = generate_flmn(_rng, _L, 4)
    _Fs = transforms.wigner_inverse(_G, _L, 4)
    conventions_rows.append(
        dict(
            paper="Lemma IV.7(c), Eq. (39)",
            check="fibre integral of Wigner samples + spin transform vs reading the column $n=-s$, $s=0,2,-3$",
            error=max(rel_err(transforms.forward(lifting.project_samples(_Fs, _s), _L, _s), lifting.project(_G, _s)) for _s in (0, 2, -3)),
        )
    )
    conventions_rows.append(
        dict(
            paper="Lemma IV.7(a), Cor. IV.12",
            check="fibre decomposition $F=\\sum_{|s|<N}\\Lambda_s\\Lambda_s^{\\dagger}F$ on samples, $N=4$",
            error=rel_err(sum(lifting.lift_samples(lifting.project_samples(_Fs, _s), _s, 4) for _s in range(-3, 4)), _Fs),
        )
    )
    check_table(conventions_rows)
    return (conventions_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The generalised lifted convolution (Sec. V)

    The lift and its adjoint are (Definition IV.6)

    $$(\Lambda_s f)(\alpha,\beta,\gamma)=\tfrac{1}{\sqrt{2\pi}}\,e^{-\mathrm{i}s\gamma}f(\beta,\alpha),\qquad(\Lambda_s^\dagger F)(\beta,\alpha)=\tfrac{1}{\sqrt{2\pi}}\int_0^{2\pi}e^{\mathrm{i}s\gamma}F(\alpha,\beta,\gamma)\,\mathrm{d}\gamma.$$

    In harmonic space the lift fills the single Wigner column $n=-s$ and the projection reads it back (Lemma IV.7(c), Eq. 39):

    $$(\Lambda_s f)^{\ell}_{mn}=(-1)^s\sqrt{\tfrac{8\pi^2}{2\ell+1}}\;{}_sf_{\ell m}\,\delta_{n,-s},\qquad(\Lambda^\dagger_s F)_{\ell m}=(-1)^s\sqrt{\tfrac{2\ell+1}{8\pi^2}}\;F^{\ell}_{m,-s}.$$

    The group convolution $(\Psi\star F)(\rho)=\int\Psi^*(\rho^{-1}\rho')\,F(\rho')\,\mathrm{d}\mu(\rho')$ is a matrix product at each degree, $(\Psi\star F)^{\ell}_{mn}=\sum_{m'}F^{\ell}_{mm'}\Psi^{\ell*}_{nm'}$ (Eqs. 19, 20).  Composing the three gives Theorem V.2:

    $$
    \begin{aligned}
    (\mathcal{C}_{s\to G}[\Psi]\,{}_sf)^{\ell}_{mn} &= (-1)^s\sqrt{\tfrac{8\pi^2}{2\ell+1}}\;{}_sf_{\ell m}\,\Psi^{\ell*}_{n,-s}, &\text{(43)}\\
    (\mathcal{C}_{s\to s'}[\Psi]\,{}_sf)_{\ell m} &= (-1)^{s+s'}\;{}_sf_{\ell m}\,\Psi^{\ell*}_{-s',-s}, &\text{(44)}\\
    (\mathcal{C}_{G\to s'}[\Psi]\,F)_{\ell m} &= (-1)^{s'}\sqrt{\tfrac{2\ell+1}{8\pi^2}}\;\textstyle\sum_{m'}F^{\ell}_{mm'}\,\Psi^{\ell*}_{-s',m'}, &\text{(45)}\\
    (\mathcal{C}_{G\to G}[\Psi]\,F)^{\ell}_{mn} &= \textstyle\sum_{m'}F^{\ell}_{mm'}\,\Psi^{\ell*}_{nm'}. &\text{(46)}
    \end{aligned}
    $$

    s2conv implements the four operators by these closed forms, in `s2conv.lifted_convolution`.  The table below checks the closed forms against `lifted_conv_literal`, which follows Definition V.1 on samples: it lifts on the sampling of $\mathrm{SO}(3)$, applies Wigner transforms, convolves and integrates along the fibre.  It then checks `s2conv.lifted_convolution`, and `definition_v1`, which composes s2conv's lift, group convolution and projection in harmonic space, against the closed forms as written out independently in `theorem_v2`.
    """)
    return


@app.function
def lifted_conv_literal(samples, kernel, s_in, s_out, N):
    """C_{s_in -> s_out}[Psi] evaluated on samples with the pieces of s2conv: the lift on the SO(3) grid, the Wigner
    transform, the group convolution (20), the inverse Wigner transform and the fibre integral.  The kernel must be
    azimuthally band-limited, Psi^l_{nk} = 0 for |n| >= N or |k| >= N, and group inputs and outputs are sampled at N."""
    L = kernel.shape[1]
    N_in = N if s_in == "G" else abs(s_in) + 1
    F = transforms.wigner_forward(samples if s_in == "G" else lifting.lift_samples(samples, s_in, N_in), L, N_in)
    G = transforms.wigner_inverse(lifted.group_convolution(F, kernel, N), L, N)
    return G if s_out == "G" else lifting.project_samples(G, s_out)


@app.function
def definition_v1(x, kernel, s_in, s_out, N_out=None):
    """C_{s_in -> s_out}[Psi] composed as in Definition V.1 from s2conv's lift, group convolution and adjoint
    projection, in harmonic space; outputs on the rotation group keep the orders |n| < N_out (default L)."""
    L = kernel.shape[1]
    F = x if s_in == "G" else lifting.lift(x, s_in)
    if s_out == "G":
        return lifted.group_convolution(F, kernel, L if N_out is None else N_out)
    return lifting.project(lifted.group_convolution(F, kernel, abs(s_out) + 1), s_out)


@app.function
def theorem_v2(x, kernel, s_in, s_out, N_out=None):
    """The closed forms (43)-(46) of Theorem V.2, written out from the kernel block Psi^l_{nk} independently of
    s2conv's operators; outputs on the rotation group keep the orders |n| < N_out (default L)."""
    Psi = block_from_flmn(kernel)
    L = Psi.shape[0]
    el = np.arange(L)
    N_out = L if N_out is None else N_out
    if s_in != "G" and s_out != "G":
        c = (-1) ** (s_in + s_out) * np.conj(Psi[:, L - 1 - s_out, L - 1 - s_in])
        return c[:, None] * x * lm_mask(L, max(abs(s_in), abs(s_out)))
    if s_in != "G":
        column = np.conj(Psi[:, L - N_out : L + N_out - 1, L - 1 - s_in]).T
        pref = (-1) ** s_in * np.sqrt(EIGHT_PI2 / (2 * el + 1))
        return column[:, :, None] * (pref[:, None] * x * lm_mask(L, s_in))[None]
    N = azimuthal_bandlimit(x)
    if s_out != "G":
        row = np.conj(Psi[:, L - 1 - s_out, L - N : L + N - 1])
        pref = (-1) ** s_out * np.sqrt((2 * el + 1) / EIGHT_PI2)
        return pref[:, None] * np.einsum("klm,lk->lm", x, row) * lm_mask(L, s_out)
    return np.einsum("klm,lnk->nlm", x, np.conj(Psi[:, L - N_out : L + N_out - 1, L - N : L + N - 1]))


@app.function
def random_kernel(L, rng, N=None):
    """A random kernel in s2conv's layout, kernel[N-1+k, el, L-1+n] = Psi^l_{nk}, with every Wigner coefficient,
    |n|, |k| <= l, populated; with N given, only the orders |n|, |k| < N, i.e. azimuthally band-limited on both sides."""
    N_k = L if N is None else N
    k = np.arange(-(N_k - 1), N_k)[:, None, None]
    el = np.arange(L)[None, :, None]
    n = m_axis(L)[None, None, :]
    support = (np.abs(n) <= el) & (np.abs(k) <= el) & (np.abs(n) < N_k)
    shape = (2 * N_k - 1, L, 2 * L - 1)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * support


@app.function
def kernel_block_plot(ax, kernel, el, read=(), title=None):
    """|Psi^l_{nk}| of a kernel at one degree, with the entries read by an operator outlined."""
    Psi = block_from_flmn(kernel)
    L = Psi.shape[0]
    block = np.abs(Psi[el, L - 1 - el : L + el, L - 1 - el : L + el])
    ax.imshow(block, cmap="magma", extent=(-el - 0.5, el + 0.5, el + 0.5, -el - 0.5), vmin=0)
    for kind, index in read:
        if kind == "entry":
            box = ((index[1] - 0.5, index[0] - 0.5), 1, 1)
        elif kind == "column":
            box = ((index - 0.5, -el - 0.5), 1, 2 * el + 1)
        elif kind == "row":
            box = ((-el - 0.5, index - 0.5), 2 * el + 1, 1)
        else:
            box = ((-el - 0.5, -el - 0.5), 2 * el + 1, 2 * el + 1)
        ax.add_patch(Rectangle(*box, fill=False, ec="#00e5ff", lw=2.2))
    ax.set_xlabel("second index $k$", fontsize=8)
    ax.set_ylabel("first index $n$", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9)


@app.cell(hide_code=True)
def _():
    s_ui = mo.ui.slider(-3, 3, value=2, show_value=True, label="input spin $s$")
    sp_ui = mo.ui.slider(-3, 3, value=1, show_value=True, label="output spin $s'$")
    ell_ui = mo.ui.slider(3, 8, value=4, show_value=True, label="degree $\\ell$ shown")
    mo.vstack(
        [
            mo.md(
                "Choose the spins; the figure shows the part of a random kernel read by each operator "
                "(the interactive version of Fig. 5), and the table checks Theorem V.2 at these spins.  The "
                "literal evaluations use a random kernel with orders $|n|,|k|<5$, so that the Wigner transforms "
                "on $\\mathrm{SO}(3)$ need only the azimuthal band-limit $N=5$."
            ),
            mo.hstack([s_ui, sp_ui, ell_ui], justify="start", gap=2),
        ]
    )
    return ell_ui, s_ui, sp_ui


@app.cell
def _(ell_ui, s_ui, seed, sp_ui):
    _s, _sp, _el = s_ui.value, sp_ui.value, ell_ui.value
    _rng = np.random.default_rng(seed + 1)
    _L, _N = L_CHECK, N_CHECK
    _Psi = random_kernel(_L, _rng)  # every entry: for the figure and the harmonic checks
    _PsiN = random_kernel(_L, _rng, _N)  # orders |n|, |k| < N: for the literal checks through Wigner transforms
    _f = generate_flm(_rng, _L, spin=_s)
    _g = generate_flm(_rng, _L, spin=_sp)
    _F = generate_flmn(_rng, _L, _N)
    _f_samples, _F_samples = transforms.inverse(_f, _L, _s), transforms.wigner_inverse(_F, _L, _N)

    framework_rows = [
        dict(
            paper="Thm V.2, Eq. (44)",
            check=f"$\\mathcal{{C}}_{{{_s}\\to{_sp}}}$: literal (lift on grid, Wigner, group convolution, fibre integral) vs closed form",
            error=rel_err(transforms.forward(lifted_conv_literal(_f_samples, _PsiN, _s, _sp, _N), _L, _sp), theorem_v2(_f, _PsiN, _s, _sp)),
        ),
        dict(
            paper="Thm V.2, Eq. (43)",
            check=f"$\\mathcal{{C}}_{{{_s}\\to G}}$: literal vs closed form",
            error=rel_err(transforms.wigner_forward(lifted_conv_literal(_f_samples, _PsiN, _s, "G", _N), _L, _N), theorem_v2(_f, _PsiN, _s, "G", _N)),
        ),
        dict(
            paper="Thm V.2, Eq. (45)",
            check=f"$\\mathcal{{C}}_{{G\\to{_sp}}}$: literal vs closed form",
            error=rel_err(transforms.forward(lifted_conv_literal(_F_samples, _PsiN, "G", _sp, _N), _L, _sp), theorem_v2(_F, _PsiN, "G", _sp)),
        ),
        dict(
            paper="Thm V.2, Eq. (46)",
            check="$\\mathcal{C}_{G\\to G}$: literal vs closed form",
            error=rel_err(transforms.wigner_forward(lifted_conv_literal(_F_samples, _PsiN, "G", "G", _N), _L, _N), theorem_v2(_F, _PsiN, "G", "G", _N)),
        ),
        dict(
            paper="Def. V.1",
            check="`s2conv.lifted_convolution` and the composition `definition_v1` vs closed forms (43)–(46)",
            error=max(
                rel_err(_operator(_x, _Psi, _si, _so), theorem_v2(_x, _Psi, _si, _so))
                for _operator in (s2conv.lifted_convolution, definition_v1)
                for _x, _si, _so in ((_f, _s, _sp), (_f, _s, "G"), (_F, "G", _sp), (_F, "G", "G"))
            ),
        ),
    ]
    _flip = kernels.flipped(_Psi)
    framework_rows.append(
        dict(
            paper="Thm V.2 (adjoints)",
            check=f"$\\langle\\mathcal{{C}}_{{{_s}\\to{_sp}}}[\\Psi]f,g\\rangle=\\langle f,\\mathcal{{C}}_{{{_sp}\\to{_s}}}[\\widetilde\\Psi]g\\rangle$",
            error=abs(np.sum(s2conv.lifted_convolution(_f, _Psi, _s, _sp) * np.conj(_g)) - np.sum(_f * np.conj(s2conv.lifted_convolution(_g, _flip, _sp, _s))))
            / abs(np.sum(s2conv.lifted_convolution(_f, _Psi, _s, _sp) * np.conj(_g))),
        )
    )
    # Corollary V.4: a kernel whose column -s is supported on n = -s' makes the projection lossless, so that
    # projecting equals sqrt(2 pi) times evaluation on the section gamma = 0; otherwise it does not.
    _constrained = np.zeros_like(_PsiN)  # the kernel layout is [N-1+k, el, L-1+n]: keep only the entry (-s', -s)
    _constrained[_N - 1 - _s, :, _L - 1 - _sp] = _PsiN[_N - 1 - _s, :, _L - 1 - _sp]
    for _name, _K, _expect in (("constrained", _constrained, "agree"), ("unconstrained", _PsiN, "differ")):
        _on_section = np.sqrt(2 * np.pi) * lifted_conv_literal(_f_samples, _K, _s, "G", _N)[0]
        framework_rows.append(
            dict(
                paper="Cor. V.4",
                check=f"{_name} kernel: $\\sqrt{{2\\pi}}(\\Psi\\star\\Lambda_s f)\\circ s$ vs $\\mathcal{{C}}_{{{_s}\\to{_sp}}}f$ (should {_expect})",
                error=rel_err(_on_section, transforms.inverse(s2conv.lifted_convolution(_f, _K, _s, _sp), _L, _sp)),
                expect=_expect,
            )
        )

    _fig = Figure(figsize=(12, 3.2), layout="constrained")
    _panels = [
        (f"$\\mathcal{{C}}_{{{_s}\\to{_sp}}}$: entry $({-_sp},{-_s})$, Eq. (44)", [("entry", (-_sp, -_s))]),
        (f"$\\mathcal{{C}}_{{{_s}\\to G}}$: column $k={-_s}$, Eq. (43)", [("column", -_s)]),
        (f"$\\mathcal{{C}}_{{G\\to{_sp}}}$: row $n={-_sp}$, Eq. (45)", [("row", -_sp)]),
        ("$\\mathcal{C}_{G\\to G}$: whole block, Eq. (46)", [("all", None)]),
    ]
    for _i, (_title, _read) in enumerate(_panels):
        kernel_block_plot(_fig.add_subplot(1, 4, _i + 1), _Psi, _el, _read if _el >= max(abs(_s), abs(_sp)) else (), _title)
    _caption = mo.md(f"$\\lvert\\Psi^{{\\ell}}_{{nk}}\\rvert$ of a random kernel at $\\ell={_el}$, with the part read by each operator outlined.")
    mo.vstack([mo.as_html(_fig), _caption, check_table(framework_rows)])
    return (framework_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. The kernels of the classical convolutions (Sec. VI, Table II, Fig. 5)

    Each convolution of the literature is one of the four operators applied to a particular kernel:

    * **A spherical kernel lifted like a signal**, $\Lambda_s\psi$, occupies the column $k=-s$, with $(\Lambda_s\psi)^{\ell}_{n,-s}=(-1)^s\sqrt{8\pi^2/(2\ell+1)}\,{}_s\psi_{\ell n}$ (Lemma IV.7(c)).  It gives the directional convolutions (Proposition VI.1) and, when $\psi$ is axisymmetric, the axisymmetric convolutions (Proposition VI.3).
    * **The flipped kernel** $\widetilde\Psi(g)=\Psi^*(g^{-1})$ has blocks $\Psi^{\ell\dagger}$, so $\widetilde{\Lambda_s\psi}$ occupies the row $n=-s$ and gives Haar synthesis (Proposition VI.2).
    * **The left-convolution kernel** $\Psi_h(g)=h^*(g^{-1}\eta)$ is invariant under left multiplication by the stabiliser, $\Psi_h(R_z(\chi)g)=\Psi_h(g)$.  It has the single row $n=0$, with $(\Psi_h)^{\ell}_{0k}=2\pi\sqrt{4\pi/(2\ell+1)}\,h^*_{\ell k}$ (Proposition VI.4 and the accompanying notes).
    * **The canonical kernel** of a multiplier $c_\ell$ from spin $s$ to spin $s'$ has the single entry $(\Psi_T)^{\ell}_{-s',-s}=(-1)^{s+s'}c^*_{\ell}$ (Theorem V.5(a)).  It gives every spin-changing convolution and the operators of mathematical physics (Propositions VI.5, VI.8).
    * **The identity entry** $\Psi^{\ell}_{-s,-s}=1$ reproduces the lift and the projection themselves, $\mathcal{C}_{s\to G}=\Lambda_s$ and $\mathcal{C}_{G\to s}=\Lambda_s^\dagger$ (Corollary V.6).  It also gives $\mathcal{C}_{s\to s}=\mathrm{Id}$, whose kernel is a distribution, not a square-integrable function (Theorem V.5(b)).

    The module `s2conv.kernels` constructs each of them.  It stores a kernel as a function on $\mathrm{SO}(3)$, by its Wigner coefficients in the layout of `s2fft`, `kernel[N-1+k, el, L-1+n]` $=\Psi^{\ell}_{nk}$, where the azimuthal band-limit $N$ bounds the second index $k$; the figures show the block $\Psi^{\ell}$ at one degree.  The check table then samples three of the kernels as functions on $\mathrm{SO}(3)$ and Wigner transforms them with `s2fft`, confirming the stated blocks.  The spherical kernels there have orders $|m|<5$, so that the flipped and left-convolution kernels, whose dependence on $\gamma$ carries those orders, are sampled exactly at the azimuthal band-limit $N=5$.
    """)
    return


@app.function
def block_from_flmn(flmn):
    """The block Psi[l, L-1+n, L-1+k] of a kernel stored by s2conv as the Wigner coefficients flmn[N-1+k, l, L-1+n]
    of the function Psi(g), for the figures and for the closed forms written out in theorem_v2."""
    N, L = azimuthal_bandlimit(flmn), flmn.shape[1]
    Psi = np.zeros((L, 2 * L - 1, 2 * L - 1), np.complex128)
    Psi[:, :, L - N : L + N - 1] = np.transpose(flmn, (1, 2, 0))
    return Psi


@app.cell
def _(L, seed):
    _rng = np.random.default_rng(seed + 2)
    _el = 4
    _m = m_axis(L)
    _psi2 = generate_flm(_rng, L, spin=2)
    _gallery = [
        ("axisymmetric\n$\\Lambda_0\\psi$, $\\psi_{\\ell m}\\propto\\delta_{m0}$", kernels.lifted(generate_flm(_rng, L) * (_m == 0)[None], 0), [("entry", (0, 0))]),
        ("left (Driscoll–Healy)\n$\\Psi_h$", kernels.left(generate_flm(_rng, L)), [("entry", (0, 0))]),
        ("directional\n$\\Lambda_0\\psi$", kernels.lifted(generate_flm(_rng, L), 0), [("column", 0)]),
        ("directional spin-2\n$\\Lambda_2\\psi$", kernels.lifted(_psi2, 2), [("column", -2)]),
        ("Haar synthesis, spin 2\n$\\widetilde{\\Lambda_2\\psi}$", kernels.flipped(kernels.lifted(_psi2, 2)), [("row", -2)]),
        ("spin $2\\to1$ multiplier\n$\\Psi_T$", kernels.multiplier(np.ones(L), 2, 1), [("entry", (-1, -2))]),
        (
            "steerable, $s=1$, $N=3$\n$\\sum_{|s'|<3}\\Psi_{T,s'}$",
            kernels.steerable({_q: _rng.standard_normal(L) + 1j for _q in range(-2, 3)}, 1),
            [("column", -1)],
        ),
        ("group convolution\nany $\\Psi$", random_kernel(L, _rng), [("all", None)]),
        ("lift $\\Lambda_{2}$ / projection $\\Lambda_2^\\dagger$\nidentity entry", kernels.identity(L, 2), [("entry", (-2, -2))]),
        ("$E$ from $Q+\\mathrm{i}U$\n$\\mathcal{C}_{2\\to0}$, $c_\\ell=-1/2$", kernels.multiplier(-0.5 * np.ones(L), 2, 0), [("entry", (0, -2))]),
    ]
    _fig = Figure(figsize=(13, 5.6), layout="constrained")
    for _i, (_title, _K, _read) in enumerate(_gallery):
        kernel_block_plot(_fig.add_subplot(2, 5, _i + 1), _K, _el, _read, _title)
    _fig.suptitle(f"Wigner blocks $|\\Psi^{{\\ell}}_{{nk}}|$ at $\\ell={_el}$; outlined: the part read by the operator (Fig. 5)", fontsize=10)
    _fig
    return


@app.cell
def _(seed):
    _rng = np.random.default_rng(seed + 3)
    _L, _N = L_CHECK, N_CHECK
    _th, _ph = transforms.sphere_angles(_L)
    _gam = lifting.gamma_samples(_N)
    _orders = (np.abs(m_axis(_L)) < _N)[None]  # orders |m| < N: the kernels below are sampled exactly at N
    kernel_rows = []
    for _s in (0, 2):
        _psi = generate_flm(_rng, _L, spin=_s) * _orders
        _sampled = lifting.lift_samples(transforms.inverse(_psi, _L, _s), _s, _N)
        kernel_rows.append(
            dict(
                paper="Lemma IV.7(c)",
                check=f"$\\Lambda_{{{_s}}}\\psi$ sampled on $\\mathrm{{SO}}(3)$, Wigner transformed vs the column $k={-_s}$",
                error=rel_err(transforms.wigner_forward(_sampled, _L, _N), kernels.lifted(_psi, _s, _N)),
            )
        )
        # tilde Phi(g) = Phi^*(g^{-1}); g^{-1} has Euler angles (pi - gamma, beta, -pi - alpha)
        _vals = evaluate(_psi, _s, _th[None, :], (np.pi - _gam)[:, None])
        _flipped = (-1) ** _s / np.sqrt(2 * np.pi) * np.exp(-1j * _s * _ph)[None, None, :] * np.conj(_vals)[:, :, None]
        kernel_rows.append(
            dict(
                paper="Thm V.2, Prop. VI.2",
                check=f"$\\widetilde{{\\Lambda_{{{_s}}}\\psi}}(g)=(\\Lambda_{{{_s}}}\\psi)^*(g^{{-1}})$ sampled vs the blocks $\\Psi^{{\\ell\\dagger}}$ (row $n={-_s}$)",
                error=rel_err(transforms.wigner_forward(_flipped, _L, _N), kernels.flipped(kernels.lifted(_psi, _s), _N)),
            )
        )
    _h = generate_flm(_rng, _L) * _orders
    _hv = evaluate(_h, 0, _th[None, :], (np.pi - _gam)[:, None])  # h(beta, pi - gamma) = h(g^{-1} eta)
    _Kh = np.broadcast_to(np.conj(_hv)[:, :, None], (2 * _N - 1, _L, 2 * _L - 1))
    kernel_rows.append(
        dict(
            paper="Prop. VI.4, notes Sec. III",
            check="$\\Psi_h(g)=h^*(g^{-1}\\eta)$ sampled vs the row $(\\Psi_h)^{\\ell}_{0k}=2\\pi\\sqrt{4\\pi/(2\\ell+1)}\\,h^*_{\\ell k}$",
            error=rel_err(transforms.wigner_forward(_Kh, _L, _N), kernels.left(_h, _N)),
        )
    )
    _f = generate_flm(_rng, _L, spin=2)
    _F = generate_flmn(_rng, _L, 4)
    kernel_rows.append(
        dict(
            paper="Cor. V.6",
            check="identity entry: $\\mathcal{C}_{2\\to G}=\\Lambda_2$, $\\mathcal{C}_{G\\to2}=\\Lambda_2^\\dagger$, $\\mathcal{C}_{2\\to2}=\\mathrm{Id}$",
            error=max(
                rel_err(s2conv.lifted_convolution(_f, kernels.identity(_L, 2), 2, "G", N_out=3), lifting.lift(_f, 2, 3)),
                rel_err(s2conv.lifted_convolution(_F, kernels.identity(_L, 2), "G", 2), lifting.project(_F, 2)),
                rel_err(s2conv.lifted_convolution(_f, kernels.identity(_L, 2), 2, 2), _f),
            ),
        )
    )
    check_table(kernel_rows)
    return (kernel_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Test fields and kernels

    The demonstrations use the following fields and kernels:

    * a scalar field $f$ made of copies of an elongated bar, placed across the sphere at random orientations $\gamma_k$, $f=\sum_k\mathcal{R}_{(\varphi_k,\theta_k,\gamma_k)}b$;
    * a smooth random scalar field;
    * the spin-2 field obtained from the smooth field by two applications of the spin raising operator $\eth$ (Sec. 10), itself computed as a spin-changing lifted convolution.

    The directional kernels are steerable, with azimuthal orders $|m|<5$.  The bar $b$ is an elongated Gaussian at the North pole along the direction $\gamma=0$, which the rotation $\mathcal{R}_{(\varphi,\theta,\gamma)}$ carries to the direction making the angle $\gamma$ with $e_\theta$, measured towards $e_\varphi$.  The ridge kernel $\psi$ is a zero-mean ridge detector with the same geometry.  The kernels are drawn at $(\theta,\varphi)=(\pi/2,0)$, at the centre of the maps.
    """)
    return


@app.function
def place_kernel(psi_lm, theta, phi, gamma=0.0):
    """R_rho psi for rho = (phi, theta, gamma): the kernel centred on (theta, phi) and turned by gamma."""
    return rotate_flm(psi_lm, (phi, theta, gamma))


@app.function
def oriented_kernel(L, sig_long, sig_short, N, ridge=False):
    """An elongated kernel at the North pole with its long axis along the direction gamma = 0, truncated to
    azimuthal orders |m| < N so that it is steerable; ridge=True makes the profile across it zero-mean."""
    L2 = 2 * L
    th, ph = transforms.sphere_angles(L2)
    T, P = np.meshgrid(th, ph, indexing="ij")
    x, y = T * np.cos(P), T * np.sin(P)
    across = np.exp(-0.5 * (y / sig_short) ** 2)
    if ridge:
        across = across - np.exp(-0.5 * (y / (2.5 * sig_short)) ** 2) / 2.5
    g = np.exp(-0.5 * (x / sig_long) ** 2) * across * (T < np.pi / 2)
    glm = transforms.forward(g, L2)[:L, L2 - L : L2 + L - 1] * (np.abs(m_axis(L)) < N)[None]
    return glm / np.sqrt(np.sum(np.abs(glm) ** 2))


@app.function
def bars_field(bar_lm, rng, n_bars=16):
    """A scalar field made of copies of a bar at jittered Fibonacci points with random orientations."""
    k = np.arange(n_bars) + 0.5
    z = 1 - 2 * k / n_bars
    azimuths = np.mod(np.pi * (1 + 5**0.5) * k, 2 * np.pi)
    f = np.zeros_like(bar_lm)
    for zi, ai in zip(z, azimuths):
        theta = float(np.clip(np.arccos(zi) + rng.normal(0, 0.05), 0.05, np.pi - 0.05))
        f = f + place_kernel(bar_lm, theta, ai + rng.normal(0, 0.1), rng.uniform(0, np.pi))
    return f


@app.function
def smooth_field(L, rng, ell0=5.0, el_min=0):
    """A real random scalar field with a Gaussian power spectrum of width ell0, normalised to unit norm."""
    el = np.arange(L)
    f = generate_flm(rng, L, reality=True) * (np.exp(-0.5 * (el / ell0) ** 2) * (el >= el_min))[:, None]
    return f / np.sqrt(np.sum(np.abs(f) ** 2))


@app.function
def gaussian_axisymmetric(L, sigma, s=0):
    """Axisymmetric spin-s kernel psi_{l0} = sqrt((2l+1)/(4 pi)) exp(-l(l+1) sigma^2 / 2), for which the axisymmetric
    convolution (Eq. 14) multiplies by the Gaussian beam exp(-l(l+1) sigma^2 / 2)."""
    el = np.arange(L)
    psi = np.zeros((L, 2 * L - 1), np.complex128)
    psi[:, L - 1] = np.sqrt((2 * el + 1) / FOUR_PI) * np.exp(-0.5 * el * (el + 1) * sigma**2)
    return psi * lm_mask(L, s)


@app.function
def spin_from_scalar(f_lm, s):
    """A spin-s field from a scalar by |s| applications of eth (s > 0) or eth-bar (s < 0), each a spin-changing
    lifted convolution C_{q -> q+-1} with the canonical kernel of its multiplier."""
    L = f_lm.shape[0]
    out, q = f_lm, 0
    while q != s:
        step = 1 if s > q else -1
        c = physics.eth_multiplier(L, q) if step == 1 else physics.ethbar_multiplier(L, q)
        out = s2conv.lifted_convolution(out, kernels.multiplier(c, q, q + step), q, q + step)
        q += step
    return out


@app.function
def upsample(flm, L_plot):
    """Zero-pad harmonic coefficients to band-limit L_plot, i.e. exact band-limited interpolation."""
    L = flm.shape[0]
    if L_plot <= L:
        return flm
    out = np.zeros((L_plot, 2 * L_plot - 1), np.complex128)
    out[:L, L_plot - L : L_plot + L - 1] = flm
    return out


@app.function
def field_grid(flm, s=0, L_plot=128):
    """Samples of a band-limited spin field on the MW grid of band-limit L_plot, with the grid."""
    L_plot = max(L_plot, flm.shape[0])
    return transforms.inverse(upsample(flm, L_plot), L_plot, s), transforms.sphere_angles(L_plot)


@app.function
def mollweide_grid(ax, data, thetas, phis, cmap="RdBu_r", vmax=None, symmetric=True, title=None):
    """Draw MW samples data[t, p] on a Mollweide axis, with longitude 0 at the centre."""
    lon = np.mod(phis + np.pi, 2 * np.pi) - np.pi
    order = np.argsort(lon)
    lon = np.append(lon[order], np.pi)
    data = np.concatenate([data[:, order], data[:, order][:, :1]], axis=1)
    if vmax is None:
        vmax = np.max(np.abs(data)) if symmetric else np.max(data)
    image = ax.pcolormesh(lon, np.pi / 2 - thetas, data, shading="gouraud", cmap=cmap,
                          vmin=-vmax if symmetric else 0.0, vmax=vmax, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9)
    return image


@app.function
def mollweide(ax, flm, s=0, part="real", cmap=None, L_plot=128, vmax=None, title=None):
    """Mollweide map of the real part, imaginary part or modulus of a band-limited spin field."""
    f, (th, ph) = field_grid(flm, s, L_plot)
    data = {"real": f.real, "imag": f.imag, "abs": np.abs(f)}[part]
    symmetric = part != "abs"
    return mollweide_grid(ax, data, th, ph, cmap or ("RdBu_r" if symmetric else "YlGnBu"), vmax, symmetric, title)


@app.function
def orthographic(ax, flm, s=0, part="real", cmap=None, lat0=0.35, lon0=0.6, glyph_flm=None, glyph_s=None,
                 n_glyph=420, glyph_len=0.07, glyph_color="k", vmax=None, title=None, L_plot=128, npix=260):
    """One hemisphere in orthographic projection.  Colour shows the field; for spin fields, glyphs with |s|-fold
    symmetry show the value in the local frame (e_theta, e_phi): an arrow for spin 1, a headless stick for spin 2
    and three spokes for spin 3, at the angle arg(f)/s from e_theta towards e_phi (Eq. 11 and Fig. 1).  The glyphs
    may be taken from another field (glyph_flm, glyph_s)."""
    out = unit_vector(np.pi / 2 - lat0, lon0)
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    V = np.stack([east, np.cross(out, east), out], axis=1)  # screen right, screen up, towards the viewer
    u = np.linspace(-1, 1, npix)
    U, W = np.meshgrid(u, u)
    inside = U**2 + W**2 < 1
    th, ph = angles_of(np.stack([U, W, np.sqrt(np.clip(1 - U**2 - W**2, 0, None))], axis=-1) @ V.T)
    f, (tg, _pg) = field_grid(flm, s, L_plot)
    vals = {"real": f.real, "imag": f.imag, "abs": np.abs(f)}[part]
    nt, npp = vals.shape
    ti = np.interp(th, tg, np.arange(nt))
    t0 = np.clip(np.floor(ti).astype(int), 0, nt - 2)
    tw = ti - t0
    pf = ph / (2 * np.pi) * npp
    p0 = np.floor(pf).astype(int) % npp
    pw = pf - np.floor(pf)
    p1 = (p0 + 1) % npp
    img = (1 - tw) * ((1 - pw) * vals[t0, p0] + pw * vals[t0, p1]) + tw * ((1 - pw) * vals[t0 + 1, p0] + pw * vals[t0 + 1, p1])
    img = np.where(inside, img, np.nan)
    symmetric = part != "abs"
    vmax = np.nanmax(np.abs(img)) if vmax is None else vmax
    image = ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap or ("RdBu_r" if symmetric else "YlGnBu"),
                      vmin=-vmax if symmetric else 0.0, vmax=vmax, interpolation="bilinear")
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=0.8, color="0.3"))
    gflm, gs = (flm, s) if glyph_flm is None else (glyph_flm, glyph_s)
    if gs:
        k = np.arange(n_glyph) + 0.5
        zc = 1 - 2 * k / n_glyph
        az = np.pi * (1 + 5**0.5) * k
        pts = np.stack([np.sqrt(1 - zc**2) * np.cos(az), np.sqrt(1 - zc**2) * np.sin(az), zc], -1)
        pts = pts[pts @ out > 0.2]
        gt, gp = angles_of(pts)
        z = evaluate(gflm, gs, gt, gp)
        size = glyph_len * (0.35 + 0.65 * np.abs(z) / np.max(np.abs(z)))
        e_t = np.stack([np.cos(gt) * np.cos(gp), np.cos(gt) * np.sin(gp), -np.sin(gt)], -1)
        e_p = np.stack([-np.sin(gp), np.cos(gp), np.zeros_like(gp)], -1)
        n_fold = abs(gs)
        segments = []
        for j in range(1 if n_fold == 2 else n_fold):
            angle = np.angle(z) / gs + 2 * np.pi * j / n_fold
            d = (np.cos(angle)[:, None] * e_t + np.sin(angle)[:, None] * e_p) * size[:, None]
            if n_fold == 1:
                a, b = (pts - 0.8 * d) @ V, 1.6 * d @ V
                ax.quiver(a[:, 0], a[:, 1], b[:, 0], b[:, 1], angles="xy", scale_units="xy", scale=1,
                          color=glyph_color, width=0.006, headwidth=4, headlength=4.5, headaxislength=4)
            else:
                a = ((pts - d) if n_fold == 2 else pts) @ V
                b = (pts + d) @ V
                segments += [np.stack([p, q])[:, :2] for p, q in zip(a, b)]
        if segments:
            ax.add_collection(LineCollection(segments, colors=glyph_color, linewidths=1.1, capstyle="round"))
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return image


@app.function
def so3_slice(flmn, gamma, L_plot=128):
    """F(alpha, beta, gamma) at fixed gamma on an MW grid, from the fibre decomposition of Corollary IV.12:
    F(alpha, beta, gamma) = sum_s (2 pi)^{-1/2} e^{-i s gamma} (Lambda_s^dagger F)(beta, alpha)."""
    total, grid = 0.0, None
    for s in range(-(azimuthal_bandlimit(flmn) - 1), azimuthal_bandlimit(flmn)):
        comp, grid = field_grid(lifting.project(flmn, s), s, L_plot)
        total = total + np.exp(-1j * s * gamma) * comp / np.sqrt(2 * np.pi)
    return total, grid


@app.cell
def _(L, seed):
    _rng = np.random.default_rng(seed + 4)
    N_DIR = 5  # azimuthal band-limit of the directional kernels: orders |m| < 5
    bar_lm = oriented_kernel(L, sig_long=3.75 * 2.6 / L, sig_short=2.6 / L, N=N_DIR)
    ridge_lm = oriented_kernel(L, sig_long=2.5 * 2.6 / L, sig_short=2.6 / L, N=N_DIR, ridge=True)
    f_bars = bars_field(bar_lm, _rng)
    f_smooth = smooth_field(L, _rng, ell0=5.0, el_min=1)
    p_spin2 = spin_from_scalar(f_smooth, 2)
    p_spin2 = p_spin2 / np.sqrt(np.sum(np.abs(p_spin2) ** 2))

    _fig = Figure(figsize=(13, 6.2), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), f_bars, title="bars field $f$")
    mollweide(_fig.add_subplot(2, 3, 2, projection="mollweide"), place_kernel(bar_lm, np.pi / 2, 0.0), title="bar $b$ at $(\\pi/2, 0)$, $\\gamma=0$")
    mollweide(_fig.add_subplot(2, 3, 3, projection="mollweide"), place_kernel(ridge_lm, np.pi / 2, 0.0), title="ridge kernel $\\psi$ at $(\\pi/2, 0)$, $\\gamma=0$")
    orthographic(_fig.add_subplot(2, 3, 4), f_smooth, 0, title="smooth scalar field")
    orthographic(_fig.add_subplot(2, 3, 5), p_spin2, 2, part="abs", title="spin-2 field $\\eth^2 f$: $|{}_2f|$ and sticks")
    orthographic(_fig.add_subplot(2, 3, 6), f_bars, 0, lat0=0.2, lon0=2.4, title="bars field, orthographic view")
    _fig
    return N_DIR, bar_lm, f_bars, f_smooth, p_spin2, ridge_lm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Directional convolution and steerability (Definitions III.3, III.5; Propositions VI.1, VI.6)

    The directional convolution $(f\circledast\psi)(\rho)=\langle f,\mathcal{R}_\rho\psi\rangle$ is a function on $\mathrm{SO}(3)$ with Wigner coefficients $\frac{8\pi^2}{2\ell+1}f_{\ell m}\psi^*_{\ell n}$ (Eq. 18).  It is the lifted convolution with no projection and the lifted kernel, $f\circledast\psi=\mathcal{C}_{0\to G}[\Lambda_0\psi]f$, exactly and with no residual constant (Proposition VI.1).

    Since the kernel is steerable, with orders $|n|<N$, the output is a tuple of $2N-1$ spin fields, $\mathcal{C}_{s\to G}[\Psi]=\sum_{|s'|<N}\Lambda_{s'}\,\mathcal{C}_{s\to s'}[\Psi]$ (Proposition VI.6).  The response at *any* orientation is therefore a phase-weighted sum of spin-changing convolutions:

    $$(f\circledast\psi)(\alpha,\beta,\gamma)=\sum_{|s'|<N}\tfrac{1}{\sqrt{2\pi}}\,e^{-\mathrm{i}s'\gamma}\,\big(\mathcal{C}_{0\to s'}[\Lambda_0\psi]f\big)(\beta,\alpha).$$

    Move the slider to steer the ridge detector.  The spin-2 component ${}_2g=\mathcal{C}_{0\to2}[\Lambda_0\psi]f$ is itself an equivariant orientation field: the spin-$\pm2$ part of the response is $2(2\pi)^{-1/2}\lvert{}_2g\rvert\cos\big(2\gamma-\arg{}_2g\big)$, largest at $\gamma=\arg({}_2g)/2$, so that ${}_2g$ drawn as headless sticks lies along the bars.
    """)
    return


@app.cell(hide_code=True)
def _():
    gamma_ui = mo.ui.slider(0, 175, step=5, value=30, show_value=True, label="kernel orientation $\\gamma$ (degrees)")
    gamma_ui
    return (gamma_ui,)


@app.cell
def _(N_DIR, f_bars, ridge_lm):
    F_dir = s2conv.lifted_convolution(f_bars, kernels.lifted(ridge_lm, 0), 0, "G", N_out=N_DIR)  # f (*) psi on SO(3)
    dir_spins = {q: s2conv.lifted_convolution(f_bars, kernels.lifted(ridge_lm, 0), 0, q) for q in range(-(N_DIR - 1), N_DIR)}
    return F_dir, dir_spins


@app.cell
def _(F_dir, dir_spins, f_bars, gamma_ui, ridge_lm):
    _g = np.deg2rad(gamma_ui.value)
    _slice, (_th, _ph) = so3_slice(F_dir, _g, 128)
    _fig = Figure(figsize=(13, 6.4), layout="constrained")
    mollweide(_fig.add_subplot(2, 2, 1, projection="mollweide"), f_bars, title="input $f$")
    mollweide(
        _fig.add_subplot(2, 2, 2, projection="mollweide"),
        place_kernel(ridge_lm, np.pi / 2, 0.0, _g),
        title=f"kernel $\\mathcal{{R}}_\\rho\\psi$ at $(\\pi/2,0)$, $\\gamma$ = {gamma_ui.value}°",
    )
    mollweide_grid(
        _fig.add_subplot(2, 2, 3, projection="mollweide"),
        _slice.real,
        _th,
        _ph,
        title=f"steered response $(f\\circledast\\psi)(\\alpha,\\beta,\\gamma)$ at $\\gamma$ = {gamma_ui.value}°",
    )
    orthographic(
        _fig.add_subplot(2, 2, 4),
        f_bars,
        0,
        lat0=0.2,
        lon0=2.4,
        glyph_flm=dir_spins[2],
        glyph_s=2,
        glyph_len=0.06,
        n_glyph=900,
        title="orientation field: spin-2 component $\\mathcal{C}_{0\\to2}[\\Lambda_0\\psi]f$ over $f$",
    )
    _fig
    return


@app.cell
def _(F_dir, L, N_DIR, dir_spins, f_bars, ridge_lm, seed):
    _el = np.arange(L)
    _expected = np.zeros_like(F_dir)
    for _n in range(-(N_DIR - 1), N_DIR):
        _expected[N_DIR - 1 + _n] = (EIGHT_PI2 / (2 * _el + 1))[:, None] * f_bars * np.conj(ridge_lm[:, L - 1 + _n])[:, None]
    _samples = transforms.wigner_inverse(F_dir, L, N_DIR)
    _steered = [sum(np.exp(-1j * _q * _gm) * transforms.inverse(_c, L, _q) for _q, _c in dir_spins.items()) / np.sqrt(2 * np.pi) for _gm in lifting.gamma_samples(N_DIR)]
    directional_rows = [
        dict(paper="Prop. VI.1, Eq. (18)", check="$\\mathcal{C}_{0\\to G}[\\Lambda_0\\psi]f$ and `classical.directional_convolution` vs $\\frac{8\\pi^2}{2\\ell+1}f_{\\ell m}\\psi^*_{\\ell n}$ (bars, ridge)",
             error=max(rel_err(F_dir, _expected), rel_err(classical.directional_convolution(f_bars, ridge_lm, 0, N_DIR), _expected))),
        dict(paper="Prop. VI.6", check="tuple of spin-changing convolutions, steered, vs inverse Wigner transform at the $\\gamma$ samples", error=rel_err(np.array(_steered), _samples)),
    ]
    _rng = np.random.default_rng(seed + 5)
    for _s in (0, 2, -1):
        _f = generate_flm(_rng, L_CHECK, spin=_s)
        _psi = generate_flm(_rng, L_CHECK, spin=_s)
        _out = s2conv.lifted_convolution(_f, kernels.lifted(_psi, _s), _s, "G")
        _rots = [random_rotation(_rng) for _ in range(4)]
        directional_rows.append(
            dict(
                paper="Defs III.3, III.5, Prop. VI.1",
                check=f"$s={_s}$: $\\mathcal{{C}}_{{{_s}\\to G}}[\\Lambda_{{{_s}}}\\psi]f$ at random $\\rho$ vs $\\langle f,\\mathcal{{R}}_\\rho\\psi\\rangle$ by quadrature, kernel rotated in real space",
                error=rel_err([evaluate_so3(_out, *_r) for _r in _rots], [inner_with_rotated(_f, _psi, _s, _r) for _r in _rots]),
            )
        )
    check_table(directional_rows)
    return (directional_rows,)


@app.function
def gl_quadrature(L):
    """Gauss-Legendre nodes in cos(theta) and uniform longitudes: exact on the sphere up to band-limit 2L + 1."""
    x, w = np.polynomial.legendre.leggauss(L + 1)
    n_phi = 2 * L + 1
    T, P = np.meshgrid(np.arccos(x), 2 * np.pi * np.arange(n_phi) / n_phi, indexing="ij")
    return T, P, w[:, None] * (2 * np.pi / n_phi) * np.ones_like(T)


@app.function
def rotated_values(psi_lm, s, rotation, theta, phi):
    """(R_rho psi)(omega) = e^{-i s vartheta} psi(R_rho^{-1} omega) in real space (Eq. 12), vartheta the third
    Euler angle of R_rho^{-1} R_omega."""
    a, b, c = euler_angles(rotation_matrix(*rotation).T @ section(theta, phi))
    return np.exp(-1j * s * c) * evaluate(psi_lm, s, b, a)


@app.function
def inner_with_rotated(f_lm, psi_lm, s, rotation, bilinear=False):
    """<f, R_rho psi> = int f (R_rho psi)^* d mu by exact quadrature, with the kernel rotated in real space
    (Definitions III.1, III.3, III.5-III.7); bilinear=True drops the conjugation, as in Definition III.8."""
    T, P, W = gl_quadrature(f_lm.shape[0])
    k = rotated_values(psi_lm, s, rotation, T, P)
    return np.sum(W * evaluate(f_lm, s, T, P) * (k if bilinear else np.conj(k)))


@app.function
def left_convolution_quadrature(f_lm, h_lm, theta, phi):
    """(f *_L h)(omega) = int_{SO(3)} f(rho eta) h(R_rho^{-1} omega) d mu(rho) (Definition III.2) by exact quadrature:
    Gauss-Legendre in cos(beta), uniform in alpha and gamma, with h evaluated at the rotated points."""
    L = f_lm.shape[0]
    x, w = np.polynomial.legendre.leggauss(L + 1)
    n = 2 * L + 1
    ang = 2 * np.pi * np.arange(n) / n
    A, B, G = np.meshgrid(ang, np.arccos(x), ang, indexing="ij")
    weights = (2 * np.pi / n) ** 2 * w[None, :, None]
    R = rotation_matrix(A, B, G)
    f_rho_eta = evaluate(f_lm, 0, B, A)
    out = []
    for t, p in zip(np.atleast_1d(theta), np.atleast_1d(phi)):
        tt, pp = angles_of(np.einsum("...ji,j->...i", R, unit_vector(t, p)))
        out.append(np.sum(weights * f_rho_eta * evaluate(h_lm, 0, tt, pp)))
    return np.array(out)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Axisymmetric and left convolutions (Definitions III.1, III.2; Propositions VI.3, VI.4)

    **Axisymmetric convolution.**  For an axisymmetric kernel, $(f\odot\psi)(\omega)=\langle f,\mathcal{R}_\omega\psi\rangle$ keeps its output on the sphere, with $(f\odot\psi)_{\ell m}=\sqrt{4\pi/(2\ell+1)}\,f_{\ell m}\psi^*_{\ell0}$ (Eq. 14).  In the framework it is $(2\pi)^{-1/2}\,\mathcal{C}_{0\to0}[\Lambda_0\psi]$.  The factor $(2\pi)^{-1/2}$ is the difference between evaluating a Mackey function on the section and projecting it with the adjoint (Proposition VI.3).

    **Left convolution of Driscoll & Healy.**  $(f\star_{\mathrm{L}}h)(\omega)=\int_{\mathrm{SO}(3)}f(\rho\eta)\,h(R_\rho^{-1}\omega)\,\mathrm{d}\mu(\rho)$ equals $\mathcal{C}_{0\to0}[\Psi_h]$ with no residual constant, and its harmonic form is $2\pi\sqrt{4\pi/(2\ell+1)}\,f_{\ell m}h_{\ell0}$ (Eq. 16).  The $2\pi$ is the volume of the fibre, and only the zonal part $h_{\ell0}$ of an arbitrary kernel enters, since $\mathcal{C}_{0\to0}$ reads the single entry $(\Psi_h)^{\ell}_{00}$ (Proposition VI.4).

    Below, the bars field is smoothed by a Gaussian axisymmetric kernel.  It is also left-convolved with the *directional* bar $b$, and the result is identical to the left convolution with the zonal part of $b$ alone.
    """)
    return


@app.cell
def _(L, bar_lm, f_bars, seed):
    _m = m_axis(L)
    _el = np.arange(L)
    _psi_ax = gaussian_axisymmetric(L, np.deg2rad(4.0))
    f_axisym = s2conv.lifted_convolution(f_bars, kernels.lifted(_psi_ax, 0), 0, 0) / np.sqrt(2 * np.pi)
    f_left = s2conv.lifted_convolution(f_bars, kernels.left(bar_lm), 0, 0)
    _f_left_zonal = s2conv.lifted_convolution(f_bars, kernels.left(bar_lm * (_m == 0)[None]), 0, 0)

    _fig = Figure(figsize=(13, 3.3), layout="constrained")
    mollweide(_fig.add_subplot(1, 3, 1, projection="mollweide"), f_bars, title="input $f$")
    mollweide(_fig.add_subplot(1, 3, 2, projection="mollweide"), f_axisym, title="$f\\odot\\psi$, Gaussian beam $4°$")
    mollweide(_fig.add_subplot(1, 3, 3, projection="mollweide"), f_left, title="$f\\star_{\\mathrm{L}}b$ with the directional bar $b$")

    _eq14 = np.sqrt(FOUR_PI / (2 * _el + 1))[:, None] * f_bars * np.conj(_psi_ax[:, L - 1])[:, None]
    _eq16 = 2 * np.pi * np.sqrt(FOUR_PI / (2 * _el + 1))[:, None] * f_bars * bar_lm[:, L - 1][:, None]
    axisym_rows = [
        dict(paper="Prop. VI.3, Eq. (14)", check="$(2\\pi)^{-1/2}\\mathcal{C}_{0\\to0}[\\Lambda_0\\psi]f$ and `classical.axisymmetric_convolution` vs $\\sqrt{4\\pi/(2\\ell+1)}f_{\\ell m}\\psi^*_{\\ell0}$",
             error=max(rel_err(f_axisym, _eq14), rel_err(classical.axisymmetric_convolution(f_bars, _psi_ax), _eq14))),
        dict(paper="Prop. VI.4, Eq. (16)", check="$\\mathcal{C}_{0\\to0}[\\Psi_b]f$ and `classical.left_convolution` vs $2\\pi\\sqrt{4\\pi/(2\\ell+1)}f_{\\ell m}b_{\\ell0}$",
             error=max(rel_err(f_left, _eq16), rel_err(classical.left_convolution(f_bars, bar_lm), _eq16))),
        dict(paper="Prop. VI.4", check="only the zonal part enters: $f\\star_{\\mathrm{L}}b$ vs $f\\star_{\\mathrm{L}}b_{\\text{zonal}}$", error=rel_err(f_left, _f_left_zonal)),
    ]
    _rng = np.random.default_rng(seed + 6)
    _f = generate_flm(_rng, L_CHECK)
    _psi = generate_flm(_rng, L_CHECK) * (m_axis(L_CHECK) == 0)[None]
    _h = generate_flm(_rng, L_CHECK)
    _pts_t, _pts_p = np.arccos(_rng.uniform(-0.9, 0.9, 3)), _rng.uniform(0, 2 * np.pi, 3)
    _ax_out = s2conv.lifted_convolution(_f, kernels.lifted(_psi, 0), 0, 0) / np.sqrt(2 * np.pi)
    axisym_rows.append(
        dict(paper="Def. III.1, Prop. VI.3", check="framework vs $\\langle f,\\mathcal{R}_\\omega\\psi\\rangle$ by quadrature at random $\\omega$",
             error=rel_err(evaluate(_ax_out, 0, _pts_t, _pts_p), [inner_with_rotated(_f, _psi, 0, (_p, _t, 0.0)) for _t, _p in zip(_pts_t, _pts_p)]))
    )
    axisym_rows.append(
        dict(paper="Def. III.2, Prop. VI.4", check="framework vs $\\int f(\\rho\\eta)h(R_\\rho^{-1}\\omega)\\,\\mathrm{d}\\mu(\\rho)$ by quadrature on $\\mathrm{SO}(3)$",
             error=rel_err(evaluate(s2conv.lifted_convolution(_f, kernels.left(_h), 0, 0), 0, _pts_t, _pts_p), left_convolution_quadrature(_f, _h, _pts_t, _pts_p)))
    )
    mo.vstack([mo.as_html(_fig), check_table(axisym_rows)])
    return axisym_rows, f_axisym, f_left


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Spin convolutions, synthesis and a directional spin wavelet frame (Definitions III.5, III.6, III.10; Propositions VI.1–VI.3)

    The directional spin wavelet transform of McEwen *et al.* consists of three lifted convolutions:

    * wavelet coefficients $W^{\psi^{(j)}}={}_sf\circledast{}_s\psi^{(j)}=\mathcal{C}_{s\to G}[\Lambda_s\psi^{(j)}]\,{}_sf$ (Eq. 23);
    * scaling coefficients $W^{\Phi}={}_sf\odot{}_s\Phi=(2\pi)^{-1/2}\mathcal{C}_{s\to0}[\Lambda_s\Phi]\,{}_sf$, a *scalar* field (Eq. 25);
    * the synthesis (Eq. 26), whose directional part is Haar synthesis $W\circledast^\dagger\psi=\mathcal{C}_{G\to s}[\widetilde{\Lambda_s\psi}]W$ with $(W\circledast^\dagger\psi)_{\ell m}=\sum_nW^{\ell}_{mn}\psi_{\ell n}$ (Eq. 32).  Haar synthesis is the adjoint of analysis (Proposition VI.2).

    The admissibility condition (Eq. 27) states that the analysis operators form a Parseval frame, so that synthesis inverts analysis exactly.  Below, a spin-2 field is analysed with a scaling function and four steerable directional wavelets.  These are built from telescoping Gaussian windows, $\kappa_\ell^2+\sum_j\zeta_{j\ell}^2=1$, with orders $n\in\{0,\pm2\}$.  The field is then reconstructed.
    """)
    return


@app.function
def directional_wavelets(L, s, ell0=2.0, n_scales=4):
    """A spin-s scaling function and steerable directional wavelets with orders n = 0, +-2 satisfying the
    admissibility condition (27) for l >= |s| >= 2, from telescoping Gaussian windows kappa^2 + sum_j zeta_j^2 = 1."""
    el = np.arange(L)
    windows = [np.exp(-((el / (ell0 * 2.0**j)) ** 2)) for j in range(n_scales)] + [np.ones(L)]
    kappa = np.sqrt(windows[0])
    zetas = [np.sqrt(np.clip(windows[j + 1] - windows[j], 0, None)) for j in range(n_scales)]
    eta = {-2: 1 / np.sqrt(6), 0: 2 / np.sqrt(6), 2: 1 / np.sqrt(6)}
    scaling = np.zeros((L, 2 * L - 1), np.complex128)
    scaling[:, L - 1] = np.sqrt((2 * el + 1) / FOUR_PI) * kappa
    wavelets = []
    for zeta in zetas:
        psi = np.zeros((L, 2 * L - 1), np.complex128)
        for n, weight in eta.items():
            psi[:, L - 1 + n] = np.sqrt((2 * el + 1) / EIGHT_PI2) * zeta * weight
        wavelets.append(psi * lm_mask(L, s))
    return scaling * lm_mask(L, s), wavelets


@app.cell
def _(L, p_spin2, seed):
    _s = 2
    _el = np.arange(L)
    _Phi, _psis = directional_wavelets(L, _s)
    W_scaling = s2conv.lifted_convolution(p_spin2, kernels.lifted(_Phi, _s), _s, 0) / np.sqrt(2 * np.pi)
    W_wavelets = [s2conv.lifted_convolution(p_spin2, kernels.lifted(_psi, _s), _s, "G", N_out=3) for _psi in _psis]
    _recon = s2conv.lifted_convolution(W_scaling, kernels.flipped(kernels.lifted(_Phi, _s)), 0, _s) / np.sqrt(2 * np.pi) + sum(
        s2conv.lifted_convolution(_W, kernels.flipped(kernels.lifted(_psi, _s)), "G", _s) for _psi, _W in zip(_psis, W_wavelets)
    )
    _recon_classical = classical.axisymmetric_synthesis(W_scaling, _Phi, _s) + sum(
        classical.directional_synthesis(_W, _psi, _s) for _psi, _W in zip(_psis, W_wavelets)
    )
    _admissible = (FOUR_PI / (2 * _el + 1)) * np.abs(_Phi[:, L - 1]) ** 2 + (EIGHT_PI2 / (2 * _el + 1)) * sum(np.sum(np.abs(_p) ** 2, axis=1) for _p in _psis)
    _eq23 = np.stack([(EIGHT_PI2 / (2 * _el + 1))[:, None] * p_spin2 * np.conj(_psis[0][:, L - 1 + _n])[:, None] for _n in (-2, -1, 0, 1, 2)])
    _eq25 = np.sqrt(FOUR_PI / (2 * _el + 1))[:, None] * p_spin2 * np.conj(_Phi[:, L - 1])[:, None]

    _fig = Figure(figsize=(13, 6.2), layout="constrained")
    orthographic(_fig.add_subplot(2, 3, 1), p_spin2, 2, part="abs", title="spin-2 input: $|{}_2f|$ and sticks")
    mollweide(_fig.add_subplot(2, 3, 2, projection="mollweide"), W_scaling, title="scaling coefficients $W^{\\Phi}$ (scalar)")
    for _j in range(4):
        _slice, (_th, _ph) = so3_slice(W_wavelets[_j], 0.0, 96)
        mollweide_grid(_fig.add_subplot(2, 3, 3 + _j, projection="mollweide"), _slice.real, _th, _ph, title=f"wavelet coefficients $W^{{\\psi^{{({_j})}}}}(\\alpha,\\beta,0)$")
    _fig.suptitle("Directional spin-2 wavelet analysis by lifted convolutions", fontsize=10)

    spin_rows = [
        dict(paper="Eq. (27)", check="admissibility $\\frac{4\\pi}{2\\ell+1}|\\Phi_{\\ell0}|^2+\\frac{8\\pi^2}{2\\ell+1}\\sum_{j,n}|\\psi^{(j)}_{\\ell n}|^2=1$, $\\ell\\ge2$", error=rel_err(_admissible[2:], np.ones(L - 2))),
        dict(paper="Eq. (26), Props VI.2, VI.3", check="synthesis of the wavelet and scaling coefficients reconstructs the spin-2 field (lifted convolutions and `classical` syntheses)",
             error=max(rel_err(_recon, p_spin2), rel_err(_recon_classical, p_spin2))),
        dict(paper="Eq. (23)", check="$\\mathcal{C}_{2\\to G}[\\Lambda_2\\psi^{(0)}]$ and `classical.directional_convolution` vs $\\frac{8\\pi^2}{2\\ell+1}\\,{}_2f_{\\ell m}\\,{}_2\\psi^*_{\\ell n}$",
             error=max(rel_err(W_wavelets[0], _eq23), rel_err(classical.directional_convolution(p_spin2, _psis[0], _s, 3), _eq23))),
        dict(paper="Eq. (25)", check="$(2\\pi)^{-1/2}\\mathcal{C}_{2\\to0}[\\Lambda_2\\Phi]$ and `classical.axisymmetric_convolution` vs $\\sqrt{4\\pi/(2\\ell+1)}\\,{}_2f_{\\ell m}\\,{}_2\\Phi^*_{\\ell0}$",
             error=max(rel_err(W_scaling, _eq25), rel_err(classical.axisymmetric_convolution(p_spin2, _Phi, _s), _eq25))),
    ]
    _rng = np.random.default_rng(seed + 8)
    for _s2 in (0, 2):
        _psi = generate_flm(_rng, L_CHECK, spin=_s2)
        _W = generate_flmn(_rng, L_CHECK, L_CHECK)
        _f = generate_flm(_rng, L_CHECK, spin=_s2)
        _syn = s2conv.lifted_convolution(_W, kernels.flipped(kernels.lifted(_psi, _s2)), "G", _s2)
        _ip_so3 = np.sum(((2 * np.arange(L_CHECK) + 1) / EIGHT_PI2)[None, :, None] * s2conv.lifted_convolution(_f, kernels.lifted(_psi, _s2), _s2, "G") * np.conj(_W))
        _eq32 = np.einsum("nlm,ln->lm", _W, _psi) * lm_mask(L_CHECK, _s2)
        spin_rows.append(dict(paper="Prop. VI.2, Eq. (32)", check=f"$s={_s2}$: $\\mathcal{{C}}_{{G\\to{_s2}}}[\\widetilde{{\\Lambda_{{{_s2}}}\\psi}}]W$ and `classical.directional_synthesis` vs $\\sum_nW^{{\\ell}}_{{mn}}\\psi_{{\\ell n}}$",
                              error=max(rel_err(_syn, _eq32), rel_err(classical.directional_synthesis(_W, _psi, _s2), _eq32))))
        spin_rows.append(dict(paper="Prop. VI.2", check=f"$s={_s2}$: synthesis is the adjoint, $\\langle f\\circledast\\psi,W\\rangle=\\langle f,W\\circledast^\\dagger\\psi\\rangle$",
                              error=abs(_ip_so3 - np.sum(_f * np.conj(_syn))) / abs(_ip_so3)))
    _f = generate_flm(_rng, L_CHECK, spin=2)
    _Phi2 = generate_flm(_rng, L_CHECK, spin=2) * (m_axis(L_CHECK) == 0)[None]
    _pts_t, _pts_p = np.arccos(_rng.uniform(-0.9, 0.9, 3)), _rng.uniform(0, 2 * np.pi, 3)
    spin_rows.append(dict(paper="Def. III.6, Prop. VI.3", check="$(2\\pi)^{-1/2}\\mathcal{C}_{2\\to0}[\\Lambda_2\\Phi]$ vs $\\langle{}_2f,\\mathcal{R}_\\omega{}_2\\Phi\\rangle$ by quadrature with the spin phase",
                          error=rel_err(evaluate(s2conv.lifted_convolution(_f, kernels.lifted(_Phi2, 2), 2, 0) / np.sqrt(2 * np.pi), 0, _pts_t, _pts_p),
                                        [inner_with_rotated(_f, _Phi2, 2, (_p, _t, 0.0)) for _t, _p in zip(_pts_t, _pts_p)])))
    mo.vstack([mo.as_html(_fig), check_table(spin_rows)])
    return W_scaling, W_wavelets, spin_rows


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Convolution on the rotation group and composition (Definition III.4; Propositions V.3, VI.7)

    The group convolution $(\Psi\star F)(\rho)=\langle F,\pi_G(\rho)\Psi\rangle$ is $\mathcal{C}_{G\to G}[\Psi]$ (Proposition VI.7).  It is the $\mathrm{SO}(3)$ layer of Cohen *et al.*, the correlation of Kostelec & Rockmore and, with channel mixing, the generalised convolution of Cobb *et al.*

    The four families are closed under composition, and composing multiplies kernel blocks, $\Psi_3^{\ell}=\Psi_2^{\ell}\Psi_1^{\ell}$ (Eq. 47).  Through an intermediate spin-$s'$ field only the index $-s'$ survives, so the multipliers of spin-changing convolutions simply multiply (Proposition V.3).  A chain of spin-changing layers therefore needs no intermediate transforms.
    """)
    return


@app.cell
def _(seed):
    _rng = np.random.default_rng(seed + 9)
    _L, _N = L_CHECK, N_CHECK
    _Psi1, _Psi2 = random_kernel(_L, _rng, _N), random_kernel(_L, _rng, _N)  # orders |n|, |k| < N
    _Psi3 = kernels.compose(_Psi2, _Psi1)  # the block product Psi_2^l Psi_1^l
    _F = generate_flmn(_rng, _L, _N)
    _f = generate_flm(_rng, _L, spin=1)
    _out = s2conv.lifted_convolution(_F, _Psi1, "G", "G", N_out=_N)
    _samples = transforms.wigner_inverse(_out, _L, _N)
    _gam, (_th, _ph) = lifting.gamma_samples(_N), transforms.sphere_angles(_L)
    _errs = []
    for _j, _t, _p in ((0, 2, 5), (3, 7, 1), (8, 4, 12)):
        _PsiRot = rotate_flmn(_Psi1, (_ph[_p], _th[_t], _gam[_j]))  # pi_G(rho) Psi, in the layout of F
        _value = np.sum(((2 * np.arange(_L) + 1) / EIGHT_PI2)[None, :, None] * _F * np.conj(_PsiRot))
        _errs.append(abs(_samples[_j, _t, _p] - _value) / abs(_value))
    _psi = generate_flm(_rng, _L, spin=1)
    group_rows = [
        dict(paper="Def. III.4, Eq. (20)", check="$\\mathcal{C}_{G\\to G}[\\Psi]F$ on the grid vs $\\langle F,\\pi_G(\\rho)\\Psi\\rangle$ by Parseval with left translation", error=max(_errs)),
        dict(paper="Prop. V.3, Eq. (47)", check="$\\mathcal{C}_{G\\to G}[\\Psi_2]\\,\\mathcal{C}_{G\\to G}[\\Psi_1]=\\mathcal{C}_{G\\to G}[\\Psi_2\\Psi_1]$",
             error=rel_err(s2conv.lifted_convolution(_out, _Psi2, "G", "G", N_out=_N), s2conv.lifted_convolution(_F, _Psi3, "G", "G", N_out=_N))),
        dict(paper="Prop. V.3", check="$\\mathcal{C}_{G\\to G}[\\Psi_2]\\,\\mathcal{C}_{1\\to G}[\\Lambda_1\\psi]=\\mathcal{C}_{1\\to G}[\\Psi_2\\,\\Lambda_1\\psi]$ (directional then group)",
             error=rel_err(s2conv.lifted_convolution(s2conv.lifted_convolution(_f, kernels.lifted(_psi, 1), 1, "G"), _Psi2, "G", "G"),
                           s2conv.lifted_convolution(_f, kernels.compose(_Psi2, kernels.lifted(_psi, 1)), 1, "G"))),
        dict(paper="Prop. V.3", check="$\\mathcal{C}_{-2\\to0}[\\Psi_2]\\,\\mathcal{C}_{1\\to-2}[\\Psi_1]$: the multipliers multiply, $c^{(3)}_\\ell=c^{(2)}_\\ell c^{(1)}_\\ell$",
             error=rel_err(s2conv.lifted_convolution(s2conv.lifted_convolution(_f, _Psi1, 1, -2), _Psi2, -2, 0), lifted.spin_multiplier(_f, kernels.to_multiplier(_Psi2, -2, 0) * kernels.to_multiplier(_Psi1, 1, -2), 1, 0))),
        dict(paper="Sec. VI D", check="a lifted input reduces $\\mathcal{C}_{G\\to G}$ to $\\mathcal{C}_{s\\to G}$: $\\mathcal{C}_{G\\to G}[\\Psi]\\Lambda_1f=\\mathcal{C}_{1\\to G}[\\Psi]f$",
             error=rel_err(s2conv.lifted_convolution(lifting.lift(_f, 1, _N), _Psi1, "G", "G", N_out=_N), s2conv.lifted_convolution(_f, _Psi1, 1, "G", N_out=_N))),
    ]
    check_table(group_rows)
    return (group_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Spin-changing convolutions (Proposition VI.5, Theorem V.5, Corollary V.7)

    Every bounded rotation-equivariant map from spin $s$ to spin $s'$ is a multiplier, ${}_{s'}f_{\ell m}=c_\ell\,{}_sf_{\ell m}$, and it is the lifted convolution $\mathcal{C}_{s\to s'}$ with the canonical kernel $(\Psi_T)^{\ell}_{-s',-s}=(-1)^{s+s'}c^*_\ell$ (Theorem V.5).  In real space its kernel is radial apart from the frame phases forced by the two spins (Corollary V.7, Eq. 56).  Writing $s(x)^{-1}s(y)=R_z(a)R_y(b)R_z(c)$,

    $$K(x,y)=e^{\mathrm{i}s'a}\,K(b)\,e^{\mathrm{i}sc},\qquad K(b)=\sum_\ell\frac{2\ell+1}{4\pi}\,c_\ell\,d^{\ell}_{s's}(b).$$

    This is the form of the spin-weighted convolution of Tessore *et al.* in the translated canonical angles; their literal printed angles, $\alpha=a+\pi$, give $(-1)^{s'}K_\ell$ instead.  Directional kernel content cannot survive between two *fixed* spins.  The same family contains the spin-weighted convolutions of Esteves *et al.* and the spin needlets of Geller & Marinucci.

    Choose the spins and the width of a Gaussian multiplier $c_\ell=e^{-\ell(\ell+1)\sigma^2/2}$.  The spin-$s$ input is generated from the smooth scalar field with $\eth$ or $\bar\eth$.  The figure shows the input, the output, the radial profile $K(b)$ and the two-point kernel $K(x,\cdot)$ for $x$ at the centre of the map.
    """)
    return


@app.cell(hide_code=True)
def _():
    sc_s_ui = mo.ui.slider(-3, 3, value=2, show_value=True, label="input spin $s$")
    sc_sp_ui = mo.ui.slider(-3, 3, value=1, show_value=True, label="output spin $s'$")
    sc_width_ui = mo.ui.slider(2, 20, step=1, value=6, show_value=True, label="width $\\sigma$ (degrees)")
    mo.hstack([sc_s_ui, sc_sp_ui, sc_width_ui], justify="start", gap=2)
    return sc_s_ui, sc_sp_ui, sc_width_ui


@app.function
def radial_profile(c, s, sp, b):
    """K(b) = sum_l (2l+1)/(4 pi) c_l d^l_{s' s}(b), the radial part of the kernel of a spin s -> s' multiplier
    (Corollary V.7, using (-1)^{s+s'} d^l_{-s',-s} = d^l_{s' s}), evaluated through Eq. (B25)."""
    L = len(c)
    Delta = delta_matrices(L)
    k = m_axis(L)
    G = np.einsum("l,lk->k", np.asarray(c) * (2 * np.arange(L) + 1) / FOUR_PI, Delta[:, :, L - 1 + sp] * Delta[:, :, L - 1 + s])
    return (1j) ** (sp - s) * (np.exp(-1j * np.multiply.outer(b, k)) @ G)


@app.function
def two_point_kernel(c, s, sp, x_theta, x_phi, y_theta, y_phi, literal_angles=False):
    """K(x, y) = e^{i s' a} K(b) e^{i s c} with s(x)^{-1} s(y) = Rz(a) Ry(b) Rz(c) (Corollary V.7, Eq. A10);
    literal_angles=True uses the printed first angle a + pi of Tessore et al."""
    a, b, cc = euler_angles(section(x_theta, x_phi).T @ section(y_theta, y_phi))
    if literal_angles:
        a = a + np.pi
    return np.exp(1j * sp * a) * radial_profile(c, s, sp, b) * np.exp(1j * s * cc)


@app.function
def two_point_quadrature(c, s, sp, f_lm, theta, phi, literal_angles=False):
    """(T f)(x) = int K(x, y) f(y) d mu(y) by exact quadrature with the two-point kernel above."""
    T, P, W = gl_quadrature(f_lm.shape[0])
    fy = evaluate(f_lm, s, T, P)
    return np.array([np.sum(W * two_point_kernel(c, s, sp, t, p, T, P, literal_angles) * fy)
                     for t, p in zip(np.atleast_1d(theta), np.atleast_1d(phi))])


@app.function
def needlet_window(xi, B=2.0):
    """The needlet window b(xi) of Marinucci et al.: supported on [1/B, B], with sum_j b(xi / B^j)^2 = 1 for xi >= 1."""
    u = np.linspace(-1, 1, 20001)
    bump = np.zeros_like(u)
    bump[1:-1] = np.exp(-1.0 / (1.0 - u[1:-1] ** 2))
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (bump[1:] + bump[:-1]) * np.diff(u))])
    cumulative /= cumulative[-1]

    def plateau(t):
        t = np.asarray(t, float)
        middle = np.interp(1 - 2 * B / (B - 1) * (t - 1 / B), u, cumulative)
        return np.where(t <= 1 / B, 1.0, np.where(t >= 1, 0.0, middle))

    return np.sqrt(np.clip(plateau(np.asarray(xi, float) / B) - plateau(xi), 0.0, None))


@app.cell
def _(L, f_smooth, sc_s_ui, sc_sp_ui, sc_width_ui):
    _s, _sp = sc_s_ui.value, sc_sp_ui.value
    _el = np.arange(L)
    _sigma = np.deg2rad(sc_width_ui.value)
    _c = np.exp(-0.5 * _el * (_el + 1) * _sigma**2)
    _fin = spin_from_scalar(f_smooth, _s)
    _fout = s2conv.lifted_convolution(_fin, kernels.multiplier(_c, _s, _sp), _s, _sp)

    _fig = Figure(figsize=(13, 6.4), layout="constrained")
    orthographic(_fig.add_subplot(2, 3, 1), _fin, _s, part="real" if _s == 0 else "abs", title=f"input, spin {_s}")
    orthographic(_fig.add_subplot(2, 3, 2), _fout, _sp, part="real" if _sp == 0 else "abs", title=f"$\\mathcal{{C}}_{{{_s}\\to{_sp}}}[\\Psi_T]f$, spin {_sp}")
    _axp = _fig.add_subplot(2, 3, 3)
    _b = np.linspace(0, np.pi, 400)
    _prof = radial_profile(_c, _s, _sp, _b)
    _axp.plot(np.rad2deg(_b), _prof.real, color="#1f4e79", label="$K(b)$")
    _axp.axhline(0, color="0.6", lw=0.6)
    _axp.set_xlabel("separation $b$ (degrees)")
    _axp.set_title("radial profile of the equivariant kernel (Cor. V.7)", fontsize=9)
    _axp.set_xlim(0, 60)
    _th, _ph = transforms.sphere_angles(96)
    _T, _P = np.meshgrid(_th, _ph, indexing="ij")
    _K = two_point_kernel(_c, _s, _sp, np.pi / 2, 0.0, _T, _P)
    mollweide_grid(_fig.add_subplot(2, 3, 4, projection="mollweide"), np.abs(_K), _th, _ph, cmap="YlGnBu", symmetric=False, title="$|K(x,y)|$: radial")
    mollweide_grid(_fig.add_subplot(2, 3, 5, projection="mollweide"), _K.real, _th, _ph, title="$\\mathrm{Re}\\,K(x,y)$: frame phases $e^{\\mathrm{i}s'a}e^{\\mathrm{i}sc}$")
    _axc = _fig.add_subplot(2, 3, 6)
    _axc.plot(_el, _c.real, color="#1f4e79")
    _axc.set_xlabel("degree $\\ell$")
    _axc.set_title("multiplier $c_\\ell$", fontsize=9)
    _fig
    return


@app.cell
def _(sc_s_ui, sc_sp_ui, seed):
    _s, _sp = sc_s_ui.value, sc_sp_ui.value
    _rng = np.random.default_rng(seed + 10)
    _L = L_CHECK
    _el = np.arange(_L)
    _c = _rng.standard_normal(_L) + 1j * _rng.standard_normal(_L)
    _f = generate_flm(_rng, _L, spin=_s)
    _pts_t, _pts_p = np.arccos(_rng.uniform(-0.9, 0.9, 3)), _rng.uniform(0, 2 * np.pi, 3)
    _out = s2conv.lifted_convolution(_f, kernels.multiplier(_c, _s, _sp), _s, _sp)
    _at_pts = evaluate(_out, _sp, _pts_t, _pts_p)
    spinchange_rows = [
        dict(paper="Thm V.5, Prop. VI.5", check=f"$\\mathcal{{C}}_{{{_s}\\to{_sp}}}[\\Psi_T]$ vs the multiplier $c_\\ell\\,{{}}_sf_{{\\ell m}}$", error=rel_err(_out, _c[:, None] * _f * lm_mask(_L, max(abs(_s), abs(_sp))))),
        dict(paper="Cor. V.7, Eq. (A10)", check="framework vs quadrature with the radial kernel $e^{\\mathrm{i}s'a}K(b)e^{\\mathrm{i}sc}$ (Tessore et al., canonical angles)",
             error=rel_err(_at_pts, two_point_quadrature(_c, _s, _sp, _f, _pts_t, _pts_p))),
        dict(paper="Prop. VI.5(c)", check="Tessore et al., literal angles $a+\\pi$: quadrature equals $(-1)^{s'}$ times the canonical result",
             error=rel_err(two_point_quadrature(_c, _s, _sp, _f, _pts_t, _pts_p, literal_angles=True), (-1) ** _sp * _at_pts)),
    ]
    # Esteves et al. (2020): Eq. (13) with c_l = k_l, as the canonical kernel and as a lifted spherical kernel
    _psi = np.zeros((_L, 2 * _L - 1), np.complex128)
    _psi[:, _L - 1 - _sp] = (-1) ** _sp * np.sqrt((2 * _el + 1) / EIGHT_PI2) * np.conj(_c)
    spinchange_rows.append(dict(paper="Prop. VI.5(a), notes Sec. III", check="Esteves et al. Eq. (13) as a lifted spherical kernel with ${}_s\\psi_{\\ell,-s'}=(-1)^{s'}\\sqrt{(2\\ell+1)/8\\pi^2}\\,k^*_\\ell$",
                                error=rel_err(s2conv.lifted_convolution(_f, kernels.lifted(_psi * lm_mask(_L, _s), _s), _s, _sp), _out)))
    _k = generate_flm(_rng, _L, spin=_s)
    _reference = np.conj(_k[:, _L - 1 + _sp])[:, None] * _f * lm_mask(_L, max(abs(_s), abs(_sp)))
    _kprime = ((-1.0) ** m_axis(_L))[None, :] * np.sqrt((2 * _el + 1) / EIGHT_PI2)[:, None] * _k[:, ::-1] * lm_mask(_L, _s)
    spinchange_rows.append(dict(paper="Prop. VI.5(a), notes Eq. (4)", check="Esteves et al. Eq. (15) $g^{s'}_{\\ell m}=f^{s}_{\\ell m}k^{s*}_{\\ell s'}$ via the reparameterised kernel $k'_{\\ell n}=(-1)^n\\sqrt{(2\\ell+1)/8\\pi^2}\\,k_{\\ell,-n}$",
                                error=rel_err(s2conv.lifted_convolution(_f, kernels.lifted(_kprime, _s), _s, _sp), _reference)))
    spinchange_rows.append(dict(paper="notes Sec. III", check="lifting the unmodified Esteves kernel instead gives a different operator (should differ)",
                                error=rel_err(s2conv.lifted_convolution(_f, kernels.lifted(_k, _s), _s, _sp), _reference), expect="differ"))
    # Geller & Marinucci spin needlets: c_jl = b(sqrt(e_ls) / B^j) with sum_j c_jl^2 = 1 for l > |s|
    _eig = np.sqrt(np.clip((_el - abs(_s)) * (_el + abs(_s) + 1.0), 0, None))
    _needlets = [needlet_window(_eig / 2.0**_j) for _j in range(6)]
    _keep = _el > abs(_s)
    spinchange_rows.append(dict(paper="Prop. VI.5(b)", check="spin needlets: $\\sum_j\\mathcal{C}_{s\\to s}[b_j]\\,\\mathcal{C}_{s\\to s}[b_j]=\\mathrm{Id}$ for $\\ell>|s|$ (composition)",
                                error=rel_err(sum(s2conv.lifted_convolution(s2conv.lifted_convolution(_f, kernels.multiplier(_b, _s, _s), _s, _s), kernels.multiplier(_b, _s, _s), _s, _s) for _b in _needlets), _f * _keep[:, None])))
    spinchange_rows.append(dict(paper="Prop. VI.5(b)", check="mixed needlets: $\\sum_j\\mathcal{C}_{0\\to s}[b_j]\\,\\mathcal{C}_{s\\to0}[b_j]=\\mathrm{Id}$ for $\\ell>|s|$",
                                error=rel_err(sum(s2conv.lifted_convolution(s2conv.lifted_convolution(_f, kernels.multiplier(_b, _s, 0), _s, 0), kernels.multiplier(_b, 0, _s), 0, _s) for _b in _needlets), _f * _keep[:, None])))
    check_table(spinchange_rows)
    return (spinchange_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10. Operators of mathematical physics (Proposition VI.8)

    The spin raising and lowering operators $\eth$ and $\bar\eth$ are spin-changing multipliers, $c_\ell=\sqrt{(\ell-s)(\ell+s+1)}$ and $-\sqrt{(\ell+s)(\ell-s+1)}$, with distributional kernels.  The first figure reproduces the construction of Fig. 1: a scalar field and $\eth f$, $\eth^2f$, $\eth^3f$, each obtained by a lifted convolution and drawn with glyphs of 1-, 2- and 3-fold symmetry.

    The $E$/$B$ decomposition of a spin-$\pm2$ polarisation field, $a_E=-\tfrac12(a_2+a_{-2})$ and $a_B=\tfrac{\mathrm{i}}2(a_2-a_{-2})$, is a sum of a $\mathcal{C}_{2\to0}$ and a $\mathcal{C}_{-2\to0}$ operator.  The spherical Kaiser–Squires relation between the convergence $\kappa$ and the shear $\gamma$ of weak lensing is the multiplier $\mathcal{C}_{0\to2}$ with $\mathcal{D}_\ell=-\sqrt{(\ell+2)(\ell-1)/(\ell(\ell+1))}$.  By Proposition V.3 it is the ratio of the multipliers of $\tfrac12\eth^2$ and $\tfrac12\bar\eth\eth$.
    """)
    return


@app.cell
def _(f_smooth):
    _fields = [(spin_from_scalar(f_smooth, _q), _q) for _q in range(4)]
    _fig = Figure(figsize=(13, 3.6), layout="constrained")
    for _i, (_x, _q) in enumerate(_fields):
        orthographic(_fig.add_subplot(1, 4, _i + 1), _x, _q, part="real" if _q == 0 else "abs",
                     title=["$f$ (spin 0)", "$\\eth f$ (spin 1)", "$\\eth^2 f$ (spin 2)", "$\\eth^3 f$ (spin 3)"][_i])
    _fig.suptitle("Spin fields generated from a scalar field by lifted convolutions $\\mathcal{C}_{s\\to s+1}$ (compare Fig. 1)", fontsize=10)
    _fig
    return


@app.cell
def _(L, seed):
    _rng = np.random.default_rng(seed + 11)
    _el = np.arange(L)
    _half = 0.5 * np.ones(L)
    _aE = smooth_field(L, _rng, 5.0, el_min=2)
    _aB = 0.6 * smooth_field(L, _rng, 5.0, el_min=2)
    _a2, _am2 = -(_aE + 1j * _aB), -(_aE - 1j * _aB)
    _P, _Pbar = transforms.inverse(_a2, L, 2), transforms.inverse(_am2, L, -2)
    _p2, _pm2 = transforms.forward(_P, L, 2), transforms.forward(_Pbar, L, -2)
    _E = s2conv.lifted_convolution(_p2, kernels.multiplier(-_half, 2, 0), 2, 0) + s2conv.lifted_convolution(_pm2, kernels.multiplier(-_half, -2, 0), -2, 0)
    _B = s2conv.lifted_convolution(_p2, kernels.multiplier(1j * _half, 2, 0), 2, 0) + s2conv.lifted_convolution(_pm2, kernels.multiplier(-1j * _half, -2, 0), -2, 0)
    _kappa = smooth_field(L, _rng, 5.0, el_min=2)
    _D = physics.kaiser_squires_multiplier(L)
    _shear = s2conv.lifted_convolution(_kappa, kernels.multiplier(_D, 0, 2), 0, 2)
    _Dinv = np.where(_el >= 2, 1 / np.where(_D == 0, 1, _D), 0)
    _zs = np.array([np.sqrt(np.prod(np.arange(_l - 1, _l + 3, dtype=float))) if _l >= 2 else 0.0 for _l in range(L)])

    _fig = Figure(figsize=(13, 3.5), layout="constrained")
    orthographic(_fig.add_subplot(1, 4, 1), _a2, 2, part="abs", title="polarisation $Q+\\mathrm{i}U$ (spin 2)")
    orthographic(_fig.add_subplot(1, 4, 2), _E, 0, title="$E$ recovered by $\\mathcal{C}_{2\\to0}+\\mathcal{C}_{-2\\to0}$")
    orthographic(_fig.add_subplot(1, 4, 3), _B, 0, title="$B$ recovered by $\\mathcal{C}_{2\\to0}+\\mathcal{C}_{-2\\to0}$")
    orthographic(_fig.add_subplot(1, 4, 4), _kappa, 0, glyph_flm=_shear, glyph_s=2, title="convergence $\\kappa$ with the shear $\\mathcal{C}_{0\\to2}[\\mathcal{D}]\\kappa$")

    _pts_t, _pts_p = np.arccos(_rng.uniform(-0.9, 0.9, 5)), _rng.uniform(0, 2 * np.pi, 5)
    _hstep = 1e-4
    physics_rows = []
    for _s in (0, 1, -2):
        _f = generate_flm(_rng, L_CHECK, spin=_s)
        _vals = lambda _t, _p, _f=_f, _s=_s: evaluate(_f, _s, _t, _p)
        _dt = (-_vals(_pts_t + 2 * _hstep, _pts_p) + 8 * _vals(_pts_t + _hstep, _pts_p) - 8 * _vals(_pts_t - _hstep, _pts_p) + _vals(_pts_t - 2 * _hstep, _pts_p)) / (12 * _hstep)
        _dp = (-_vals(_pts_t, _pts_p + 2 * _hstep) + 8 * _vals(_pts_t, _pts_p + _hstep) - 8 * _vals(_pts_t, _pts_p - _hstep) + _vals(_pts_t, _pts_p - 2 * _hstep)) / (12 * _hstep)
        _f0 = _vals(_pts_t, _pts_p)
        _eth_fd = -(_dt - _s / np.tan(_pts_t) * _f0 + 1j / np.sin(_pts_t) * _dp)
        _ethbar_fd = -(_dt + _s / np.tan(_pts_t) * _f0 - 1j / np.sin(_pts_t) * _dp)
        _eth = evaluate(physics.eth(_f, _s), _s + 1, _pts_t, _pts_p)
        _ethbar = evaluate(physics.ethbar(_f, _s), _s - 1, _pts_t, _pts_p)
        physics_rows.append(dict(paper="Prop. VI.8(a), Eq. (3)", check=f"$\\eth$ and $\\bar\\eth$ on spin {_s} vs fourth-order finite differences of the differential operators (3)",
                                 error=max(rel_err(_eth, _eth_fd), rel_err(_ethbar, _ethbar_fd))))
    physics_rows += [
        dict(paper="Prop. VI.8(b)", check="$Q-\\mathrm{i}U$ is the conjugate of $Q+\\mathrm{i}U$ for real $E$, $B$", error=rel_err(_Pbar, np.conj(_P))),
        dict(paper="Prop. VI.8(b)", check="$E$ and $B$ recovered by sums of $\\mathcal{C}_{2\\to0}$ and $\\mathcal{C}_{-2\\to0}$, and by `physics.eb_decomposition`",
             error=max(rel_err(_E, _aE), rel_err(_B, _aB), *(rel_err(_x, _y) for _x, _y in zip(physics.eb_decomposition(_p2, _pm2), (_aE, _aB))))),
        dict(paper="Prop. VI.8(b)", check="$\\bar\\eth^2$ on spin 2 and $\\eth^2$ on spin $-2$: $c_\\ell=\\sqrt{(\\ell+2)!/(\\ell-2)!}$ (composition)",
             error=max(rel_err((physics.ethbar_multiplier(L, 1) * physics.ethbar_multiplier(L, 2))[2:], _zs[2:]), rel_err((physics.eth_multiplier(L, -1) * physics.eth_multiplier(L, -2))[2:], _zs[2:]))),
        dict(paper="Props VI.8(c), V.3", check="Kaiser–Squires $\\mathcal{D}_\\ell$ = multiplier of $\\frac12\\eth^2$ over that of $\\frac12\\bar\\eth\\eth$",
             error=rel_err((physics.eth_multiplier(L, 0) * physics.eth_multiplier(L, 1))[2:] / (physics.ethbar_multiplier(L, 1) * physics.eth_multiplier(L, 0))[2:], _D[2:])),
        dict(paper="Prop. VI.8(c)", check="$\\mathcal{C}_{2\\to0}[\\mathcal{D}^{-1}]\\,\\mathcal{C}_{0\\to2}[\\mathcal{D}]\\kappa=\\kappa$, and with `physics.kaiser_squires` and its inverse",
             error=max(rel_err(s2conv.lifted_convolution(_shear, kernels.multiplier(_Dinv, 2, 0), 2, 0), _kappa),
                       rel_err(physics.kaiser_squires(_kappa), _shear), rel_err(physics.kaiser_squires_inverse(_shear), _kappa))),
    ]
    mo.vstack([mo.as_html(_fig), check_table(physics_rows)])
    return (physics_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 11. Beyond the framework: section evaluation and harmonic-diagonal operators (Sec. VII)

    Anisotropic constructions that keep a directional kernel's output on the sphere are not lifted convolutions in general.  They are either **evaluations on a section** $\sigma(x)=s(x)R_z(\chi_\sigma(x))$ of the lifted convolution, which are sums of spin components in a chosen frame (Proposition VII.1, Eq. 63),

    $$E_\sigma[\Psi]\,{}_sf=\sum_{|s'|<N}\frac{e^{-\mathrm{i}s'\chi_\sigma}}{\sqrt{2\pi}}\,\mathcal{C}_{s\to s'}[\Psi]\,{}_sf,$$

    or **harmonic-diagonal multipliers** that depend on the order $m$ (Proposition VII.2):

    * the *standard correlation* (Definition III.7) is $E_{\sigma_0}[\Lambda_0\psi]$ on the coordinate section, i.e. the directional convolution at $\gamma=0$.  It coincides with the expansion of Wiaux *et al.* and has an $\mathcal{O}(L^3)$ grid algorithm by factorisation of rotations (Eqs. B26, B27);
    * the *commutative anisotropic convolution* of Sadeghi *et al.* (Definition III.8) is the bilinear version on the section $\chi_\sigma=\pi-\varphi$;
    * *harmonic multiplication* and *sifting* (Definition III.9) multiply by $g_{\ell m}$ and $g^*_{\ell m}$.

    Standard correlation and sifting meet the three criteria of Roddy & McEwen, namely a directional kernel, spherical output and $\mathcal{O}(L^3)$ cost; the cited algorithm for the commutative anisotropic convolution costs $\mathcal{O}(L^3\log L)$.  Sifting keeps only the orders $m$ present in the filter, here $|m|<5$, hence the banded appearance of its output below; the section evaluations inherit the coordinate singularities of the section at the poles (Remark IV.1).  None is fully rotation-equivariant in general (Sec. 12).  The last panel shows the equivariant alternative: the spin-2 component of the steerable directional convolution retains the orientation, as a spin field.
    """)
    return


@app.function
@functools.partial(jax.jit, static_argnames=("chi",))
def section_at(components, theta, phi, chi=None):
    """E_sigma[Psi] f = sum_{s'} e^{-i s' chi_sigma} (2 pi)^{-1/2} C_{s -> s'}[Psi] f (Eq. 63) at arbitrary points, from
    the spin components of s2conv.section.section_components, stacked over s' = -(N-1), ..., N-1; chi is the function
    chi_sigma(theta, phi), or None for the coordinate section.  JIT compiled, as the equivariance tests call it often."""
    N = azimuthal_bandlimit(components)
    total = 0.0
    for sp, comp in zip(range(-(N - 1), N), components):
        values = evaluate(comp, sp, theta, phi)
        total = total + (values if chi is None else jnp.exp(-1j * sp * chi(theta, phi)) * values)
    return total


@app.function
def section_on_grid(components, L_plot=128, chi=None):
    """E_sigma[Psi] f on the MW grid of band-limit L_plot, with one inverse spin transform per component."""
    th, ph = transforms.sphere_angles(L_plot)
    T, P = np.meshgrid(th, ph, indexing="ij")
    N = azimuthal_bandlimit(components)
    total = 0.0
    for sp, comp in zip(range(-(N - 1), N), components):
        values = field_grid(comp, sp, L_plot)[0]
        total = total + (values if chi is None else np.exp(-1j * sp * chi(T, P)) * values)
    return total, (th, ph)


@app.function
def chi_sadeghi(theta, phi):
    """The section twist chi_sigma = pi - phi of the commutative anisotropic convolution."""
    return np.pi - phi + 0.0 * theta


@app.cell
def _(L, N_DIR, dir_spins, f_bars, ridge_lm, seed):
    _el = np.arange(L)
    _comps = s2conv.section.section_components(f_bars, kernels.lifted(ridge_lm, 0), 0, N_DIR)
    _std, (_th, _ph) = section_on_grid(_comps, 128)
    _sad, _ = section_on_grid(s2conv.section.section_components(f_bars, kernels.lifted(conjugate(ridge_lm, 0), 0), 0, N_DIR), 128, chi_sadeghi)
    _sift = field_grid(diagonal.sifting_convolution(f_bars, ridge_lm), 0, 128)[0]
    _fig = Figure(figsize=(13, 6.4), layout="constrained")
    mollweide_grid(_fig.add_subplot(2, 2, 1, projection="mollweide"), _std.real, _th, _ph, title="standard correlation $E_{\\sigma_0}[\\Lambda_0\\psi]f$ (polar-equivariant)")
    mollweide_grid(_fig.add_subplot(2, 2, 2, projection="mollweide"), _sad.real, _th, _ph, title="commutative anisotropic $\\psi\\odot_{\\mathrm{S}}f$, $\\chi_\\sigma=\\pi-\\varphi$")
    mollweide_grid(_fig.add_subplot(2, 2, 3, projection="mollweide"), _sift.real, _th, _ph, title="sifting $f\\circledcirc\\psi$ (real part; polar-equivariant)")
    mollweide(_fig.add_subplot(2, 2, 4, projection="mollweide"), dir_spins[2], 2, part="abs", title="equivariant: $|\\mathcal{C}_{0\\to2}[\\Lambda_0\\psi]f|$, the spin-2 orientation energy")
    _fig.suptitle("Four ways to keep the output of a directional kernel on the sphere", fontsize=10)

    # standard correlation four ways at the demonstration band-limit
    _dirG = transforms.wigner_inverse(s2conv.lifted_convolution(f_bars, kernels.lifted(ridge_lm, 0), 0, "G", N_out=N_DIR), L, N_DIR)[0]
    _std_L = s2conv.section.standard_correlation(f_bars, ridge_lm, N_DIR)
    _wiaux = sum(transforms.inverse(((-1) ** _n * np.sqrt(FOUR_PI / (2 * _el + 1)) * np.conj(ridge_lm[:, L - 1 - _n]))[:, None] * f_bars * lm_mask(L, _n), L, _n) for _n in range(-(N_DIR - 1), N_DIR))
    _H = s2conv.section.standard_correlation_fourier(f_bars, ridge_lm)
    _thL, _phL = transforms.sphere_angles(L)
    _TL, _PL = np.meshgrid(_thL, _phL, indexing="ij")
    beyond_rows = [
        dict(paper="Prop. VII.1, Def. III.7", check="standard correlation: section evaluation (Eq. 63), `section.standard_correlation`, vs directional convolution at $\\gamma=0$", error=rel_err(_std_L, _dirG)),
        dict(paper="Sec. VII A (Wiaux et al.)", check="standard correlation: $\\sum_n\\sum_{\\ell m}(-1)^n\\sqrt{4\\pi/(2\\ell+1)}\\,\\psi^*_{\\ell,-n}f_{\\ell m}\\,{}_nY_{\\ell m}$", error=rel_err(_wiaux, _dirG)),
        dict(paper="App. B 8, Eqs. (B26), (B27)", check="standard correlation: $\\mathcal{O}(L^3)$ grid algorithm by factorisation of rotations, `section.standard_correlation_fourier`", error=rel_err(evaluate_fourier(_H, _TL, _PL), _dirG)),
    ]
    _rng = np.random.default_rng(seed + 12)
    _f, _h, _g = generate_flm(_rng, L_CHECK), generate_flm(_rng, L_CHECK), generate_flm(_rng, L_CHECK)
    _pts_t, _pts_p = np.arccos(_rng.uniform(-0.9, 0.9, 3)), _rng.uniform(0, 2 * np.pi, 3)
    _std_c = s2conv.section.section_components(_f, kernels.lifted(_h, 0), 0, L_CHECK)
    beyond_rows.append(dict(paper="Def. III.7", check="standard correlation (general kernel) vs $\\langle f,\\mathcal{R}_{(\\varphi,\\theta,0)}\\psi\\rangle$ by quadrature",
                            error=rel_err(section_at(_std_c, _pts_t, _pts_p), [inner_with_rotated(_f, _h, 0, (_p, _t, 0.0)) for _t, _p in zip(_pts_t, _pts_p)])))
    _sad_fh = s2conv.section.section_components(_f, kernels.lifted(conjugate(_h, 0), 0), 0, L_CHECK)
    _sad_hf = s2conv.section.section_components(_h, kernels.lifted(conjugate(_f, 0), 0), 0, L_CHECK)
    beyond_rows.append(dict(paper="Def. III.8, Prop. VII.1", check="commutative anisotropic: section $\\chi_\\sigma=\\pi-\\varphi$ vs $\\int(\\mathcal{R}_{(\\varphi,\\theta,\\pi-\\varphi)}h)f$ by quadrature",
                            error=rel_err(section_at(_sad_fh, _pts_t, _pts_p, chi_sadeghi), [inner_with_rotated(_f, _h, 0, (_p, _t, np.pi - _p), bilinear=True) for _t, _p in zip(_pts_t, _pts_p)])))
    beyond_rows.append(dict(paper="Def. III.8 (Sadeghi et al.)", check="commutativity $h\\odot_{\\mathrm{S}}f=f\\odot_{\\mathrm{S}}h$",
                            error=rel_err(section_at(_sad_fh, _pts_t, _pts_p, chi_sadeghi), section_at(_sad_hf, _pts_t, _pts_p, chi_sadeghi))))
    _th12, _ph12 = transforms.sphere_angles(L_CHECK)
    _rev = (-np.arange(2 * L_CHECK - 1)) % (2 * L_CHECK - 1)
    beyond_rows.append(dict(paper="Def. III.9", check="sifting: $(f\\circledcirc g)(\\theta,\\varphi)=[(g\\circledcirc f)(\\theta,-\\varphi)]^*$",
                            error=rel_err(transforms.inverse(diagonal.sifting_convolution(_f, _g), L_CHECK), np.conj(transforms.inverse(diagonal.sifting_convolution(_g, _f), L_CHECK)[:, _rev]))))
    _g_iso = np.repeat(_g[:, L_CHECK - 1 : L_CHECK], 2 * L_CHECK - 1, axis=1) * lm_mask(L_CHECK)
    beyond_rows.append(dict(paper="Prop. VII.2", check="an $m$-independent multiplier $g_{\\ell m}=g_{\\ell0}$ is $\\mathcal{C}_{0\\to0}[\\Psi_T]$ with $c_\\ell=g_{\\ell0}$",
                            error=rel_err(diagonal.harmonic_multiplication(_f, _g_iso), s2conv.lifted_convolution(_f, kernels.multiplier(_g[:, L_CHECK - 1], 0, 0), 0, 0))))
    mo.vstack([mo.as_html(_fig), check_table(beyond_rows)])
    return (beyond_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 12. Equivariance (Theorem V.2, Propositions VII.1, VII.2)

    Every lifted convolution commutes with rotations, $\mathcal{C}_{s\to s'}[\Psi]\,\pi_s(g)=\pi_{s'}(g)\,\mathcal{C}_{s\to s'}[\Psi]$ (Theorem V.2).  Following Cobb *et al.* (2021) and Ocampo *et al.* (2023), we measure the equivariance of an operator $T$ by the mean relative error

    $$\epsilon(T)=\frac{1}{N_fN_g}\sum_{i=1}^{N_f}\sum_{j=1}^{N_g}\frac{\lVert T(\mathcal{R}_{g_j}f_i)-\mathcal{R}_{g_j}(Tf_i)\rVert}{\lVert\mathcal{R}_{g_j}(Tf_i)\rVert}$$

    over $N_f=10$ random input fields, whose harmonic coefficients are drawn from the standard normal distribution, and $N_g=10$ Haar-random rotations (a single rotation for the polar tests).  The norm is that of $L^2$ on the sphere or on the rotation group.  We normalise by $\lVert\mathcal{R}_gTf\rVert=\lVert Tf\rVert$, which does not depend on the rotation; Cobb *et al.* and Ocampo *et al.* normalise by $\lVert T\mathcal{R}_gf\rVert$, which gives the same error for an equivariant operator.

    For outputs on the sphere the comparison is made in **real space**: $T(\mathcal{R}f)(\omega)$ against $(\mathcal{R}\,Tf)(\omega)=e^{-\mathrm{i}s'\vartheta}(Tf)(R^{-1}\omega)$, including the spin phase of Eq. (12), on a Gauss–Legendre quadrature grid that integrates the squared error of a band-limited output exactly.  Outputs on the rotation group are rotated by left translation of their Wigner coefficients, $(\pi(g)F)^{\ell}_{mn}=\sum_kD^{\ell}_{mk}(g)F^{\ell}_{kn}$, and their norms follow from Parseval's relation for Eq. (7).  The section-evaluated and diagonal operators of Sec. VII return a scalar field in the coordinate frame.  Standard correlation and the diagonal operators are guaranteed only polar equivariance, while the commutative anisotropic convolution keeps a polar rotation $R_z(a)$ exactly when $e^{\mathrm{i}ta}=1$ for every active output spin $t$, so that a kernel with even orders only keeps the half turn (Proposition VII.1).
    """)
    return


@app.function
def sphere_quadrature(L):
    """Gauss-Legendre nodes in cos(theta) and uniform longitudes, flattened, with their weights.  The 2L x 4L nodes
    integrate the squared modulus of a spin field of band-limit L exactly, and the section evaluations, which are not
    band-limited, accurately; they avoid the poles."""
    x, w = np.polynomial.legendre.leggauss(2 * L)
    n = 4 * L
    theta, phi = np.meshgrid(np.arccos(x), 2 * np.pi * np.arange(n) / n, indexing="ij")
    return theta.ravel(), phi.ravel(), np.repeat(w * 2 * np.pi / n, n)


@app.function
def l2_rel_err(a, b, weights):
    """||a - b|| / ||b|| in the L2 norm defined by the quadrature (or Parseval) weights."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.sum(weights * np.abs(a - b) ** 2) / max(np.sum(weights * np.abs(b) ** 2), 1e-300)))


@app.function
def equivariance_error(apply, fields, s_out, rotations, group_input=False):
    """Mean relative equivariance error of a map T with output on the sphere: the mean over the fields f and rotations
    g of ||T(R_g f) - R_g(T f)|| / ||R_g(T f)||, where R_g(T f)(omega) = e^{-i s' vartheta} (T f)(R_g^{-1} omega) is
    evaluated in real space (Eq. 12) and the L2 norms by quadrature.  apply(f) returns a function of (theta, phi);
    group_input marks inputs on SO(3), which are rotated by left translation."""
    L = fields[0].shape[-2]
    theta, phi, weights = sphere_quadrature(L)
    maps = [apply(f) for f in fields]
    errors = []
    for rotation in rotations:
        a, b, c = euler_angles(rotation_matrix(*rotation).T @ section(theta, phi))
        for f, Tf in zip(fields, maps):
            lhs = apply(rotate_flmn_jit(f, rotation) if group_input else rotate_flm_jit(f, rotation))(theta, phi)
            errors.append(l2_rel_err(lhs, np.exp(-1j * s_out * c) * Tf(b, a), weights))
    return float(np.mean(errors))


@app.function
def equivariance_error_group(apply, inputs, rotations, group_input=False):
    """Mean relative equivariance error of a map T with output on SO(3): the mean over the inputs x and rotations g of
    ||T(pi(g) x) - pi(g)(T x)|| / ||pi(g)(T x)||, where the left translation pi(g) F(rho) = F(g^{-1} rho) acts on the
    first Wigner index and ||F||^2 = sum_l (2l+1)/(8 pi^2) sum_{mn} |F^l_{mn}|^2 by Parseval's relation for Eq. (7)."""
    L = inputs[0].shape[-2]
    weights = ((2 * np.arange(L) + 1) / EIGHT_PI2)[None, :, None]
    outputs = [apply(x) for x in inputs]
    errors = []
    for rotation in rotations:
        for x, Tx in zip(inputs, outputs):
            lhs = apply(rotate_flmn_jit(x, rotation) if group_input else rotate_flm_jit(x, rotation))
            errors.append(l2_rel_err(lhs, rotate_flmn_jit(Tx, rotation), weights))
    return float(np.mean(errors))


@app.function
def on_sphere(Psi, s, sp):
    """The map f -> C_{s -> s'}[Psi] f as a function of (theta, phi), for equivariance tests."""
    return lambda f: functools.partial(evaluate_jit, lifted_convolution_jit(f, Psi, s, sp), sp)


@app.function
def section_map(Psi, s, N, chi=None):
    """The section evaluation f -> E_sigma[Psi] f as a function of (theta, phi), a scalar in the coordinate frame."""
    return lambda f: functools.partial(section_at, section_components_jit(f, Psi, s, N), chi=chi)


@app.cell
def _(seed):
    _rng = np.random.default_rng(seed + 13)
    _L = L_CHECK
    _m = m_axis(_L)
    _rot = [random_rotation(_rng) for _ in range(N_EQ_ROTATIONS)]
    _polar = [(1.1, 0.0, 0.0)]
    _half_turn = [(np.pi, 0.0, 0.0)]
    _f0 = [generate_flm(_rng, _L) for _ in range(N_EQ_FIELDS)]
    _f2 = [generate_flm(_rng, _L, spin=2) for _ in range(N_EQ_FIELDS)]
    _fm1 = [generate_flm(_rng, _L, spin=-1) for _ in range(N_EQ_FIELDS)]
    _psi = generate_flm(_rng, _L)
    _psi2 = generate_flm(_rng, _L, spin=2)
    _F = [generate_flmn(_rng, _L, _L) for _ in range(N_EQ_FIELDS)]
    _Psi = random_kernel(_L, _rng)
    _even = conjugate(_psi * (_m % 2 == 0)[None], 0)
    _g_zonal = np.repeat(_psi[:, _L - 1 : _L], 2 * _L - 1, axis=1)  # an m-independent multiplier g_lm = psi_l0

    def _row(name, error, expect="agree", rotation=f"{N_EQ_ROTATIONS} random rotations"):
        return dict(paper=name, check=f"{rotation}: {'equivariant' if expect == 'agree' else 'not equivariant'}", error=error, expect=expect)

    equivariance_rows = [
        _row("$\\mathcal{C}_{0\\to0}[\\Psi]$", equivariance_error(on_sphere(_Psi, 0, 0), _f0, 0, _rot)),
        _row("$\\mathcal{C}_{2\\to1}[\\Psi]$", equivariance_error(on_sphere(_Psi, 2, 1), _f2, 1, _rot)),
        _row("$\\mathcal{C}_{-1\\to2}[\\Psi]$", equivariance_error(on_sphere(_Psi, -1, 2), _fm1, 2, _rot)),
        _row("axisymmetric $(2\\pi)^{-1/2}\\mathcal{C}_{0\\to0}[\\Lambda_0\\psi_{\\text{zonal}}]$", equivariance_error(on_sphere(kernels.lifted(_psi * (_m == 0)[None], 0), 0, 0), _f0, 0, _rot)),
        _row("left $\\mathcal{C}_{0\\to0}[\\Psi_h]$", equivariance_error(on_sphere(kernels.left(_psi), 0, 0), _f0, 0, _rot)),
        _row("spin axisymmetric $\\mathcal{C}_{2\\to0}[\\Lambda_2\\Phi]$", equivariance_error(on_sphere(kernels.lifted(_psi2 * (_m == 0)[None], 2), 2, 0), _f2, 0, _rot)),
        _row("$\\eth$: $\\mathcal{C}_{2\\to3}$", equivariance_error(on_sphere(kernels.multiplier(physics.eth_multiplier(_L, 2), 2, 3), 2, 3), _f2, 3, _rot)),
        _row("Kaiser–Squires $\\mathcal{C}_{0\\to2}$", equivariance_error(on_sphere(kernels.multiplier(physics.kaiser_squires_multiplier(_L), 0, 2), 0, 2), _f0, 2, _rot)),
        _row("$E$ from $Q\\pm\\mathrm{i}U$", equivariance_error(lambda _p: (lambda _t, _ph: evaluate(s2conv.lifted_convolution(_p, kernels.multiplier(-0.5 * np.ones(_L), 2, 0), 2, 0)
                                                                                        + s2conv.lifted_convolution(conjugate(_p, 2), kernels.multiplier(-0.5 * np.ones(_L), -2, 0), -2, 0), 0, _t, _ph)), _f2, 0, _rot)),
        _row("directional $\\mathcal{C}_{0\\to G}[\\Lambda_0\\psi]$", equivariance_error_group(lambda _x: lifted_convolution_jit(_x, kernels.lifted(_psi, 0), 0, "G"), _f0, _rot)),
        _row("directional spin $\\mathcal{C}_{2\\to G}[\\Lambda_2\\psi]$", equivariance_error_group(lambda _x: lifted_convolution_jit(_x, kernels.lifted(_psi2, 2), 2, "G"), _f2, _rot)),
        _row("group $\\mathcal{C}_{G\\to G}[\\Psi]$", equivariance_error_group(lambda _x: lifted_convolution_jit(_x, _Psi, "G", "G"), _F, _rot, group_input=True)),
    ]
    # Haar synthesis has its input on the group: rotate the input by left translation, compare on the sphere.
    _synthesis = kernels.flipped(kernels.lifted(_psi2, 2))
    equivariance_rows += [
        _row("Haar synthesis $\\mathcal{C}_{G\\to2}[\\widetilde{\\Lambda_2\\psi}]$", equivariance_error(on_sphere(_synthesis, "G", 2), _F, 2, _rot, group_input=True)),
        _row("standard correlation $E_{\\sigma_0}[\\Lambda_0\\psi]$", equivariance_error(section_map(kernels.lifted(_psi, 0), 0, _L), _f0, 0, _rot), "differ"),
        _row("standard correlation $E_{\\sigma_0}[\\Lambda_0\\psi]$", equivariance_error(section_map(kernels.lifted(_psi, 0), 0, _L), _f0, 0, _polar), rotation="polar rotation $R_z(1.1)$"),
        _row("standard correlation, zonal kernel", equivariance_error(section_map(kernels.lifted(_psi * (_m == 0)[None], 0), 0, _L), _f0, 0, _rot)),
        _row("commutative anisotropic $\\odot_{\\mathrm{S}}$", equivariance_error(section_map(kernels.lifted(conjugate(_psi, 0), 0), 0, _L, chi_sadeghi), _f0, 0, _rot), "differ"),
        _row("commutative anisotropic $\\odot_{\\mathrm{S}}$", equivariance_error(section_map(kernels.lifted(conjugate(_psi, 0), 0), 0, _L, chi_sadeghi), _f0, 0, _polar), "differ", rotation="polar rotation $R_z(1.1)$"),
        _row("commutative anisotropic, even orders only", equivariance_error(section_map(kernels.lifted(_even, 0), 0, _L, chi_sadeghi), _f0, 0, _half_turn), rotation="half turn $R_z(\\pi)$"),
        _row("sifting $\\circledcirc$", equivariance_error(lambda _x: (lambda _t, _p: evaluate(diagonal.sifting_convolution(_x, _psi), 0, _t, _p)), _f0, 0, _rot), "differ"),
        _row("sifting $\\circledcirc$", equivariance_error(lambda _x: (lambda _t, _p: evaluate(diagonal.sifting_convolution(_x, _psi), 0, _t, _p)), _f0, 0, _polar), rotation="polar rotation $R_z(1.1)$"),
        _row("harmonic multiplication, $g_{\\ell m}=g_{\\ell0}$", equivariance_error(lambda _x: (lambda _t, _p: evaluate(diagonal.harmonic_multiplication(_x, _g_zonal), 0, _t, _p)), _f0, 0, _rot)),
    ]
    check_table(equivariance_rows, first="Operator")
    return (equivariance_rows,)


@app.cell(hide_code=True)
def _():
    eq_op_ui = mo.ui.dropdown(
        options=["spin 0 → 2 multiplier (Kaiser–Squires)", "standard correlation", "commutative anisotropic", "sifting"],
        value="standard correlation",
        label="operator",
    )
    mo.vstack([mo.md("**Equivariance at a glance.**  Rotate the bars field, apply the operator, and compare with the rotated output."), eq_op_ui])
    return (eq_op_ui,)


@app.cell
def _(L, N_DIR, eq_op_ui, f_bars, ridge_lm):
    _rot = (0.5, 1.1, -0.4)
    _choice = eq_op_ui.value
    if _choice.startswith("spin"):
        _apply, _sp = on_sphere(kernels.multiplier(physics.kaiser_squires_multiplier(L), 0, 2), 0, 2), 2
    elif _choice == "standard correlation":
        _apply, _sp = section_map(kernels.lifted(ridge_lm, 0), 0, N_DIR), 0
    elif _choice == "commutative anisotropic":
        _apply, _sp = section_map(kernels.lifted(conjugate(ridge_lm, 0), 0), 0, N_DIR, chi_sadeghi), 0
    else:
        _apply, _sp = (lambda _x: (lambda _t, _p: evaluate(diagonal.sifting_convolution(_x, ridge_lm), 0, _t, _p))), 0
    _th, _ph = transforms.sphere_angles(64)
    _T, _P = np.meshgrid(_th, _ph, indexing="ij")
    _lhs = _apply(rotate_flm(f_bars, _rot))(_T, _P)
    _a, _b, _c = euler_angles(rotation_matrix(*_rot).T @ section(_T, _P))
    _rhs = np.exp(-1j * _sp * _c) * _apply(f_bars)(_b, _a)
    _vmax = np.max(np.abs(_rhs.real))
    _fig = Figure(figsize=(13, 3.3), layout="constrained")
    mollweide_grid(_fig.add_subplot(1, 3, 1, projection="mollweide"), _lhs.real, _th, _ph, vmax=_vmax, title="$T(\\mathcal{R}f)$")
    mollweide_grid(_fig.add_subplot(1, 3, 2, projection="mollweide"), _rhs.real, _th, _ph, vmax=_vmax, title="$\\mathcal{R}(Tf)$, with the spin phase")
    mollweide_grid(_fig.add_subplot(1, 3, 3, projection="mollweide"), np.abs(_lhs - _rhs), _th, _ph, cmap="Greys", vmax=_vmax, symmetric=False,
                   title=f"$|T\\mathcal{{R}}f-\\mathcal{{R}}Tf|$, relative $L^2$ error {l2_rel_err(_lhs, _rhs, np.sin(_T)):.1e}")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 13. Real fields

    The operators apply to any band-limited field, and the fields of geophysics and cosmology are natural inputs.  This section applies them to four fields, stored as harmonic coefficients at band-limit $64$ in `notebooks/data` and truncated to the demonstration band-limit $L$, the geophysical fields with a Gaussian taper $e^{-\ell(\ell+1)/L^2}$ that keeps the ringing at coastlines and jets small (`notebooks/data/README.md` gives their sources and licences and how they are made):

    * the Earth's topography and bathymetry, a scalar field (ETOPO 2022, NOAA National Centers for Environmental Information);
    * the near-surface wind in January, averaged over 1991–2020, a spin-1 field (NCEP/NCAR reanalysis, Kalnay *et al.* 1996);
    * the ocean surface currents in 2025, a spin-1 field (GODAS, Behringer & Xue 2004);
    * a simulated weak-lensing convergence field $\kappa$ and its shear, a scalar and a spin-2 field.

    A tangent vector field with components $v_\theta$ and $v_\varphi$ is the spin-1 field ${}_1f=v_\theta+\mathrm{i}v_\varphi$, so that a wind with northward and eastward components $(v,u)$ is ${}_1f=-v+\mathrm{i}u$ and is drawn with arrows.  The section has figures only and no checks.
    """)
    return


@app.function
def example_field(name, L, taper=True):
    """Harmonic coefficients of an example field of Sec. 13, truncated to band-limit L, and its spin.  The file is read
    from notebooks/data, or downloaded from the repository on GitHub when the notebook runs without it.  With taper,
    the coefficients are multiplied by the Gaussian exp(-l(l+1) / L^2), which falls to 0.37 at l = L and keeps the
    ringing of the truncation at sharp features (coastlines, jets) small."""
    path = Path(mo.notebook_dir() or ".") / "data" / f"{name}.npz"
    if path.exists():
        data = np.load(path)
    else:
        url = f"https://raw.githubusercontent.com/astro-informatics/s2conv/main/notebooks/data/{name}.npz"
        with urllib.request.urlopen(url) as response:
            data = np.load(io.BytesIO(response.read()))
    flm, L_data = data["flm"].astype(np.complex128), int(data["L"])
    if L > L_data:
        raise ValueError(f"the example fields are stored at band-limit {L_data}, below L = {L}")
    el = np.arange(L)
    window = np.exp(-el * (el + 1) / L**2) if taper else np.ones(L)
    return flm[:L, L_data - L : L_data + L - 1] * window[:, None], int(data["spin"])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Topography.**  An axisymmetric Gaussian convolution smooths the topography.  The steerable ridge detector $\psi$ of Sec. 4 then finds mountain ranges, in the land elevation $f_+=\max(f,0)$, since the step of several kilometres at the continental margins would otherwise dominate its response.  Its directional convolution with the land elevation, computed as the tuple of spin-changing convolutions $\mathcal{C}_{0\to s'}[\Lambda_0\psi]f$, $|s'|<5$ (Proposition VI.6), gives the response at every orientation $\gamma$.  Its largest value over $\gamma$ marks ridges of the detector's width, and the spin-2 component ${}_2g=\mathcal{C}_{0\to2}[\Lambda_0\psi]f$, drawn as sticks, gives their direction.  The bars field of Sec. 4 is shown with the same analysis for comparison.
    """)
    return


@app.cell
def _(L, N_DIR, f_bars, ridge_lm):
    def _ridges(_f):
        """Largest ridge response over the orientations gamma, and the spin-2 orientation field."""
        _comps = {_q: s2conv.lifted_convolution(_f, kernels.lifted(ridge_lm, 0), 0, _q) for _q in range(-(N_DIR - 1), N_DIR)}
        _grids = {_q: np.asarray(field_grid(_c, _q)[0]) for _q, _c in _comps.items()}
        _resp = np.max([sum(np.exp(-1j * _q * _g) * _x for _q, _x in _grids.items()).real
                        for _g in np.linspace(0, np.pi, 36, endpoint=False)], axis=0) / np.sqrt(2 * np.pi)
        return np.clip(_resp, 0, None), _comps[2]

    _topo, _ = example_field("earth_topography", L)
    _smooth = classical.axisymmetric_convolution(_topo, gaussian_axisymmetric(L, 4 / L), 0)
    _grid, _ = field_grid(_topo, 0, 2 * L)  # clip at sea level on a finer grid, to keep the aliasing small
    _land = np.asarray(transforms.forward(np.maximum(np.asarray(_grid).real, 0), 2 * L))[:L, L : 3 * L - 1]
    _resp, _orient = _ridges(_land)
    _resp_bars, _orient_bars = _ridges(f_bars)
    _th, _ph = transforms.sphere_angles(128)

    _fig = Figure(figsize=(13, 6.6), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), _topo, cmap="BrBG_r", title="topography and bathymetry $f$ (km)")
    mollweide(_fig.add_subplot(2, 3, 2, projection="mollweide"), _smooth, cmap="BrBG_r", title="smoothed by an axisymmetric Gaussian convolution")
    mollweide_grid(_fig.add_subplot(2, 3, 3, projection="mollweide"), _resp, _th, _ph, cmap="YlOrBr", symmetric=False,
                   title="ridge strength $\\max_\\gamma\\,(f_+\\circledast\\psi)(\\cdot,\\gamma)$ of the land")
    orthographic(_fig.add_subplot(2, 3, 4), _topo, 0, cmap="BrBG_r", lat0=np.radians(30), lon0=np.radians(85), glyph_flm=_orient, glyph_s=2,
                 title="Himalaya, with the orientation field ${}_2g$")
    orthographic(_fig.add_subplot(2, 3, 5), _topo, 0, cmap="BrBG_r", lat0=np.radians(-15), lon0=np.radians(-65), glyph_flm=_orient, glyph_s=2,
                 title="Andes, with ${}_2g$")
    orthographic(_fig.add_subplot(2, 3, 6), f_bars, 0, lat0=0.2, lon0=2.4, glyph_flm=_orient_bars, glyph_s=2,
                 title="bars field of Sec. 4, with its ${}_2g$")
    _fig.suptitle("Directional analysis of the Earth's topography by lifted convolutions", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Wind and ocean currents.**  The spin lowering operator is the lifted convolution $\bar\eth=\mathcal{C}_{1\to0}$ with the multiplier of Sec. 10, and for a spin-1 field it gives the divergence and the vorticity (the radial component of the curl) together: $\bar\eth\,{}_1f=-(\nabla\cdot v+\mathrm{i}\,\hat r\cdot\nabla\times v)$, divided by the Earth's radius for the physical units.  Since $\eth\bar\eth$ is the Laplacian on spin-1 fields, applying $\eth\,\nabla^{-2}=\mathcal{C}_{0\to1}$ to the real and imaginary parts separately splits the field into its curl-free and divergence-free parts, the Helmholtz decomposition, which sum to the field exactly.  A spin-preserving Gaussian multiplier $\mathcal{C}_{1\to1}$ smooths a vector field without mixing its components with the local frame.
    """)
    return


@app.cell
def _(L):
    def _helmholtz(_f):
        """div + i curl of a spin-1 field in 1/s, as coefficients of its real and imaginary parts, and the curl-free
        and divergence-free parts of the field, from eth-bar f = -R (div + i curl) and f = eth lap^{-1} eth-bar f."""
        _g = -physics.ethbar(_f, 1) / R_EARTH
        _el = np.arange(_f.shape[0])
        _inv_lap = np.where(_el > 0, -1 / np.maximum(_el * (_el + 1), 1), 0.0)[:, None]
        _g_div, _g_curl = (_g + conjugate(_g, 0)) / 2, (_g - conjugate(_g, 0)) / 2
        return _g_div, _g_curl, -R_EARTH * physics.eth(_inv_lap * _g_div, 0), -R_EARTH * physics.eth(_inv_lap * _g_curl, 0)

    _wind, _ = example_field("wind", L)
    _div, _curl, _wind_div, _wind_rot = _helmholtz(_wind)
    _view = dict(lat0=np.radians(20), lon0=np.radians(-150))

    _fig = Figure(figsize=(13, 6.6), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), _wind, 1, part="abs", title="January near-surface wind speed $|{}_1f|$")
    orthographic(_fig.add_subplot(2, 3, 2), _wind, 1, part="abs", **_view, title="the wind ${}_1f$ over the Pacific")
    mollweide(_fig.add_subplot(2, 3, 3, projection="mollweide"), _curl, part="imag", title="vorticity $-\\mathrm{Im}\\,\\bar\\eth\\,{}_1f$")
    mollweide(_fig.add_subplot(2, 3, 4, projection="mollweide"), _div, part="real", title="divergence $-\\mathrm{Re}\\,\\bar\\eth\\,{}_1f$")
    orthographic(_fig.add_subplot(2, 3, 5), _wind_rot, 1, part="abs", **_view, title="divergence-free part")
    orthographic(_fig.add_subplot(2, 3, 6), _wind_div, 1, part="abs", **_view, title="curl-free part")
    _fig.suptitle("Vorticity, divergence and Helmholtz decomposition of the wind by spin-changing lifted convolutions", fontsize=10)

    _cur, _ = example_field("ocean_currents", L)
    _, _cur_curl, _, _ = _helmholtz(_cur)
    _el = np.arange(L)
    _cur_smooth = s2conv.lifted_convolution(_cur, kernels.multiplier(np.exp(-0.5 * _el * (_el + 1) * (3 / L) ** 2), 1, 1), 1, 1)
    _view = dict(lat0=np.radians(5), lon0=np.radians(-140))
    _fig2 = Figure(figsize=(13, 3.5), layout="constrained")
    mollweide(_fig2.add_subplot(1, 4, 1, projection="mollweide"), _cur, 1, part="abs", title="ocean surface current speed, 2025")
    orthographic(_fig2.add_subplot(1, 4, 2), _cur, 1, part="abs", **_view, title="currents of the equatorial Pacific")
    orthographic(_fig2.add_subplot(1, 4, 3), _cur_smooth, 1, part="abs", **_view, title="smoothed by a Gaussian $\\mathcal{C}_{1\\to1}$")
    mollweide(_fig2.add_subplot(1, 4, 4, projection="mollweide"), _cur_curl, part="imag", title="vorticity of the currents")
    mo.vstack([mo.as_html(_fig), mo.as_html(_fig2)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Weak gravitational lensing.**  The convergence $\kappa$ is a shifted lognormal field with the Limber power spectrum of the linear matter power spectrum (Eisenstein & Hu 1998) for sources at redshift 1.  Its shear is the spin-2 field $\gamma=\mathcal{C}_{0\to2}[\mathcal{D}]\kappa$ of Sec. 10, and the shear measured from galaxy shapes carries shape noise, here white with a dispersion of $0.26$ per component and a density of galaxies set by the slider.  The spherical Kaiser–Squires estimator $\mathcal{C}_{2\to0}[\mathcal{D}^{-1}]$ (Kaiser & Squires 1993; Wallis *et al.* 2022) recovers $\kappa_E+\mathrm{i}\kappa_B$ from the noisy shear: $\kappa_E$ estimates the convergence, and $\kappa_B$, which lensing does not produce, is pure noise.  All maps are smoothed by the same axisymmetric Gaussian convolution.
    """)
    return


@app.cell(hide_code=True)
def _():
    ngal_ui = mo.ui.slider(steps=[0.1, 0.3, 1, 3, 10, 30], value=1, show_value=True, label="galaxies per arcmin$^2$")
    ngal_ui
    return (ngal_ui,)


@app.cell
def _(L, ngal_ui, seed):
    _kappa, _ = example_field("convergence", L, taper=False)
    _gamma = physics.kaiser_squires(_kappa)
    _rng = np.random.default_rng(seed + 14)
    _N = 0.26**2 / (ngal_ui.value / (np.pi / 10800) ** 2)  # noise power of each of the E and B modes
    _noise = np.sqrt(_N) * (_rng.standard_normal((L, 2 * L - 1)) + 1j * _rng.standard_normal((L, 2 * L - 1))) * lm_mask(L, 2)
    _gamma_obs = _gamma + _noise
    _k = physics.kaiser_squires_inverse(_gamma_obs)
    _beam = gaussian_axisymmetric(L, 3 / L)
    _smooth = lambda _x: classical.axisymmetric_convolution(_x, _beam, 0)
    _kE, _kB, _kT = _smooth((_k + conjugate(_k, 0)) / 2), _smooth((_k - conjugate(_k, 0)) / 2j), _smooth(_kappa)
    _r = np.real(np.vdot(_kT, _kE)) / np.sqrt(np.real(np.vdot(_kT, _kT) * np.vdot(_kE, _kE)))
    _vmax = float(np.max(np.abs(field_grid(_kT)[0])))

    _fig = Figure(figsize=(13, 6.6), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), _kappa, title="simulated convergence $\\kappa$")
    orthographic(_fig.add_subplot(2, 3, 2), _kappa, 0, glyph_flm=_gamma, glyph_s=2, title="$\\kappa$ with its shear $\\gamma=\\mathcal{C}_{0\\to2}[\\mathcal{D}]\\kappa$")
    orthographic(_fig.add_subplot(2, 3, 3), _kappa, 0, glyph_flm=_gamma_obs, glyph_s=2, title="$\\kappa$ with the noisy shear")
    mollweide(_fig.add_subplot(2, 3, 4, projection="mollweide"), _kT, vmax=_vmax, title="$\\kappa$, smoothed")
    mollweide(_fig.add_subplot(2, 3, 5, projection="mollweide"), _kE, vmax=_vmax, title=f"Kaiser–Squires $\\kappa_E$, smoothed: correlation {_r:.2f} with $\\kappa$")
    mollweide(_fig.add_subplot(2, 3, 6, projection="mollweide"), _kB, vmax=_vmax, title="Kaiser–Squires $\\kappa_B$, smoothed: noise")
    _fig.suptitle("Mass mapping from a noisy shear field by lifted convolutions $\\mathcal{C}_{0\\to2}$ and $\\mathcal{C}_{2\\to0}$", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 14. Summary

    The table collects every check in the notebook.  Each row of Tables I and II of the article, and each construction of Sec. VII, is covered above by one or more checks.
    """)
    return


@app.cell
def _(axisym_rows, beyond_rows, conventions_rows, directional_rows, equivariance_rows, framework_rows, group_rows, kernel_rows, physics_rows, spin_rows, spinchange_rows):
    _sections = [
        ("1. Conventions (Sec. II, Sec. IV)", conventions_rows),
        ("2. Generalised lifted convolution (Sec. V)", framework_rows),
        ("3. Kernels (Table II, Fig. 5)", kernel_rows),
        ("5. Directional convolution, steerability", directional_rows),
        ("6. Axisymmetric and left convolutions", axisym_rows),
        ("7. Spin convolutions, synthesis, wavelets", spin_rows),
        ("8. Group convolution, composition", group_rows),
        ("9. Spin-changing convolutions", spinchange_rows),
        ("10. Operators of mathematical physics", physics_rows),
        ("11. Section evaluation, sifting (Sec. VII)", beyond_rows),
        ("12. Equivariance", equivariance_rows),
    ]
    _lines = ["| Section | Checks | Passed | Worst error of the agreeing checks |", "|:--|--:|--:|--:|"]
    _total = _passed = 0
    for _name, _rows in _sections:
        _ok = [(_r["error"] < TOL) == (_r.get("expect", "agree") == "agree") for _r in _rows]
        _agree = [_r["error"] for _r in _rows if _r.get("expect", "agree") == "agree"]
        _total += len(_rows)
        _passed += sum(_ok)
        _lines.append(f"| {_name} | {len(_rows)} | {sum(_ok)} | {max(_agree):.1e} |")
    _lines.append(f"| **All** | **{_total}** | **{_passed}** | |")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 15. Cost (Sec. VIII, Table III)

    The operators cost what the part of the kernel they read implies: $\mathcal{O}(L^3)$ for $\mathcal{C}_{s\to s'}$, which needs one forward and one inverse spin transform; $\mathcal{O}(NL^3)$ for $\mathcal{C}_{s\to G}$ with a steerable kernel; and $\mathcal{O}(NL^3+N^2L^2)$ for a group convolution with input and kernel restricted to orders $|n|<N$, which becomes $\mathcal{O}(L^4)$ when $N=L$.  The JAX backend keeps $N<8$, so the timing below fixes $N\le5$: every operator then scales as $L^3$, with a prefactor set by the part of the kernel it reads.  Each transform is compiled on its first call, which the timing excludes.  The timing is optional, since it runs transforms up to $L=128$.
    """)
    return


@app.cell(hide_code=True)
def _():
    timing_ui = mo.ui.run_button(label="Run the timing")
    timing_ui
    return (timing_ui,)


@app.cell
def _(timing_ui):
    mo.stop(not timing_ui.value, mo.md("*Press the button to time the four operators.*"))
    _rng = np.random.default_rng(0)
    _Ls = [16, 32, 64, 128]
    _N = N_CHECK
    _names = ["$\\mathcal{C}_{2\\to1}$", "$\\mathcal{C}_{2\\to G}$, $N=3$", f"$\\mathcal{{C}}_{{2\\to G}}$, $N={_N}$", f"$\\mathcal{{C}}_{{G\\to G}}$, $N={_N}$"]
    _times = {_name: [] for _name in _names}
    for _Lt in _Ls:
        _f = transforms.inverse(generate_flm(_rng, _Lt, spin=2), _Lt, 2)
        _c = np.ones(_Lt)
        _psi = generate_flm(_rng, _Lt, spin=2)
        _psi3 = _psi * (np.abs(m_axis(_Lt)) < 3)[None]
        _psiN = _psi * (np.abs(m_axis(_Lt)) < _N)[None]
        _K = random_kernel(_Lt, _rng, _N)
        _F = transforms.wigner_inverse(generate_flmn(_rng, _Lt, _N), _Lt, _N)
        _jobs = {
            _names[0]: lambda: transforms.inverse(s2conv.lifted_convolution(transforms.forward(_f, _Lt, 2), kernels.multiplier(_c, 2, 1), 2, 1), _Lt, 1),
            _names[1]: lambda: transforms.wigner_inverse(s2conv.lifted_convolution(transforms.forward(_f, _Lt, 2), kernels.lifted(_psi3, 2), 2, "G", N_out=3), _Lt, 3),
            _names[2]: lambda: transforms.wigner_inverse(s2conv.lifted_convolution(transforms.forward(_f, _Lt, 2), kernels.lifted(_psiN, 2), 2, "G", N_out=_N), _Lt, _N),
            _names[3]: lambda: transforms.wigner_inverse(s2conv.lifted_convolution(transforms.wigner_forward(_F, _Lt, _N), _K, "G", "G", N_out=_N), _Lt, _N),
        }
        for _name, _job in _jobs.items():
            _job()
            _t0 = time.perf_counter()
            _job()
            _times[_name].append(time.perf_counter() - _t0)
    _fig = Figure(figsize=(6.5, 4.2), layout="constrained")
    _ax = _fig.add_subplot()
    for _name, _ts in _times.items():
        _ax.loglog(_Ls, _ts, "o-", label=_name)
    _Larr = np.array(_Ls, float)
    _ax.loglog(_Larr, _times[_names[3]][-1] * (_Larr / _Larr[-1]) ** 3, ":", color="0.6", label="$\\propto L^3$")
    _ax.set_xticks(_Ls, [str(_Lt) for _Lt in _Ls])
    _ax.minorticks_off()
    _ax.set_xlabel("band-limit $L$")
    _ax.set_ylabel("time (s)")
    _ax.legend(fontsize=8)
    _fig
    return


if __name__ == "__main__":
    app.run()
