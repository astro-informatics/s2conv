# Lifted convolutions of real spin fields: a companion to the notebook lifted_convolutions.py, built on s2conv.
# Install s2conv with the notebook extras from the repository root, pip install -e ".[notebooks]", then run
#     marimo edit notebooks/real_fields.py   (or: marimo run notebooks/real_fields.py)
# The fields are read from notebooks/data, which make_example_fields.py builds from the public data.

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Lifted convolutions of real spin fields")

with app.setup:
    # Imports and constants shared by every cell and function below.
    import functools
    import io
    import os
    import urllib.request
    from pathlib import Path

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # allocate GPU memory as needed

    import jax

    jax.config.update("jax_enable_x64", True)  # must precede the import of s2fft and s2conv

    import marimo as mo
    import matplotlib
    import matplotlib.ticker
    import numpy as np
    import s2conv
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    from s2conv import kernels, transforms
    from s2conv.convolutions import physics
    from s2conv.utils.evaluation import evaluate
    from s2conv.utils.indexing import conjugate, lm_mask, m_axis
    from s2conv.utils.rotation import rotate_flm

    R_EARTH = 6.3712e6  # mean radius of the Earth (m)
    GAMMA = 9.798  # mean normal gravity on the Earth's surface (m s^-2)
    R_CORE = 3.485e6  # radius of the core-mantle boundary (m)
    ARCSEC = 180 * 3600 / np.pi  # arcseconds per radian
    matplotlib.rcParams["figure.dpi"] = 72  # marimo renders at twice this: keeps the largest figures within its output limit
    # print mode, switched on by export_real_fields_figures.py: figures at the text width of the article, with at most
    # three panels per row, each as wide as its column, 8 pt text and the panel letters beneath the colour bars, and
    # the figures, their layouts and the captions collected for export
    PRINT = dict(on=False, width=6.5, figures={}, layouts={}, captions={})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Lifted convolutions of real spin fields

    This notebook accompanies the article *Unifying spherical convolutions through lifting* (J. D. McEwen, 2026, forthcoming) and the companion notebook `lifted_convolutions.py`, which verifies numerically the operators of the article as implemented in [`s2conv`](https://github.com/astro-informatics/s2conv).  Here we apply the generalised lifted spherical convolution to fields measured on the Earth and on the sky, in order to illustrate what the construction affords in practice: operators that map between fields of different spin exactly and efficiently, and that respect the geometry of each field.

    **Spin fields.**  Many of the fields of geophysics and cosmology are not scalars.  A wind, an ocean current, the deflection of the vertical and the horizontal magnetic field are tangent vectors, while the polarisation of the cosmic microwave background, the shear of weak gravitational lensing and the horizontal part of the gravity-gradient tensor describe a headless direction at each point.  The components of such a field are defined only with respect to a local frame $(e_\theta,e_\varphi)$, in which $e_\theta$ points south and $e_\varphi$ east, and which turns from point to point and is undefined at the poles of the coordinates.  A field whose components are multiplied by $e^{\mathrm{i}s\chi}$ when the frame turns by $\chi$ is a spin-$s$ field, so that a vector with components $(v_\theta,v_\varphi)$ is the spin-1 field ${}_1f=v_\theta+\mathrm{i}v_\varphi$ and a polarisation with Stokes parameters $(Q,U)$ is the spin-2 field ${}_2f=Q+\mathrm{i}U$.  Treating the components as independent scalar fields, for example by smoothing $v_\theta$ and $v_\varphi$, or $Q$ and $U$, separately, combines values expressed in different frames, and the result then depends on the choice of coordinates and is incorrect near their poles.

    **Lifted convolutions.**  A spin-$s$ field is lifted to the rotation group, convolved there with a kernel $\Psi$ and projected back to the sphere with spin $s'$, $\mathcal{C}_{s\to s'}[\Psi]=\Lambda_{s'}^\dagger(\Psi\star\Lambda_s\,\cdot\,)$.  In harmonic space the operator multiplies each coefficient ${}_sf_{\ell m}$ by a single Wigner coefficient of the kernel, and every bounded rotation-equivariant linear map between spin fields is of this form.  In each of the examples we adopt the lifted convolutions whose input and output spins are those of the physical quantities involved, as summarised in the following table.

    | Field | Spin | Lifted convolutions |
    |:--|:-:|:--|
    | Topography | 0 | slope $\mathcal{C}_{0\to1}$; ridges and their orientation $\mathcal{C}_{0\to s'}$, $\lvert s'\rvert<5$ |
    | Gravity (geoid) | 0 | gravity anomaly $\mathcal{C}_{0\to0}$; deflection of the vertical $\mathcal{C}_{0\to1}$; gravity gradients $\mathcal{C}_{0\to2}$; their inverses |
    | Geomagnetic field | 0, 1 | radial and horizontal field $\mathcal{C}_{0\to0}$, $\mathcal{C}_{0\to1}$; continuation to the core $\mathcal{C}_{0\to0}$, $\mathcal{C}_{1\to1}$ |
    | Near-surface wind | 1 | vorticity and divergence $\mathcal{C}_{1\to0}$; Helmholtz decomposition $\mathcal{C}_{0\to1}$; smoothing $\mathcal{C}_{1\to1}$ |
    | Ocean surface currents | 1 | stream function and upwelling $\mathcal{C}_{1\to0}$ |
    | CMB polarisation | 2 | $E$ and $B$ modes $\mathcal{C}_{\pm2\to0}$; polarisation from $E$ and $B$ $\mathcal{C}_{0\to\pm2}$ |
    | Weak-lensing shear | 0, 2 | shear from convergence $\mathcal{C}_{0\to2}$; convergence from shear (Kaiser–Squires) $\mathcal{C}_{2\to0}$ |

    For the wind we also compare the spin-preserving smoothing $\mathcal{C}_{1\to1}$ with the smoothing of the components as if they were scalar fields, which shows the errors that the spin formalism avoids.

    We consider here the lifted convolution itself, which, since its kernels may be distributions, also captures differential operators such as $\eth$ and $\bar\eth$ and their compositions.  Convolution is, however, often an integral component of other types of analysis, such as wavelet transforms on the sphere, convolutional neural networks and Fourier neural operators, which build on the operators illustrated here and which we do not consider.

    **Data.**  The fields are stored as harmonic coefficients in `notebooks/data`, whose README gives their sources, licences and preparation, and are truncated to the band-limit $L$ set below.  The fields of the Earth are multiplied by the taper $e^{-\ell(\ell+1)/L^2}$, which falls to $0.37$ at $\ell=L$ and limits the ringing of the truncation at coastlines and jets.

    **Reading the figures.**  Maps of the whole sphere use the Mollweide projection, centred on the meridian of longitude zero with north at the top; the fields of the Earth are in geographic coordinates and the Planck maps in Galactic coordinates.  Views of one hemisphere use the orthographic projection, centred on the point given in each caption, with local north at the top.  A signed quantity is shown as colour on a diverging scale, red for positive and blue for negative values and white for zero, symmetric about zero; a non-negative quantity, such as the modulus of a spin field, is shown on a sequential scale from light to dark.  A colour bar beneath each panel gives the quantity or its units.  Spin fields are drawn in the tangent plane with the glyphs of Fig. 1 of the article: a spin-1 field ${}_1f$ as an arrow pointing along the vector, at the angle $\arg({}_1f)$ from $e_\theta$ towards $e_\varphi$, and a spin-2 field ${}_2f$ as a headless stick at the angle $\arg({}_2f)/2$, which a half turn leaves unchanged.  The glyphs are placed at quasi-uniform (Fibonacci) points, and their lengths grow linearly with the modulus of the field, from the fraction of the largest length stated in each caption up to the largest length at the largest modulus.  The colour and the glyphs of a panel may show different fields, as each caption states.  Coastlines, from Natural Earth, are drawn as black lines.
    """)
    return


@app.cell(hide_code=True)
def _():
    L_ui = mo.ui.slider(32, 128, step=16, value=64, show_value=True, label="band-limit $L$")
    mo.vstack([mo.md(f"**Controls.**  The `s2fft` transforms run on the JAX device `{jax.devices()[0]}`."), L_ui])
    return (L_ui,)


@app.cell
def _(L_ui):
    L = L_ui.value
    return (L,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Overview

    Fig. 1 gathers the fields of the examples, which span spins 0, 1 and 2 and range from the relief of the Earth to the polarisation of the oldest light in the Universe.  A field of spin 0 assigns a number to each point, a field of spin 1 an arrow, and a field of spin 2 a headless stick, i.e. a direction without a sense.  Each field is introduced briefly in the caption, and in detail in its own section, where the lifted convolutions that act on it are described.
    """)
    return


@app.cell
def _(L):
    _g4, _g8 = gaussian(L, 4 / L), gaussian(L, 8 / L)
    _topo, _ = example_field("earth_topography", L)
    _N, _ = example_field("geoid", L)
    _eps = physics.eth(_N, 0) / R_EARTH * ARCSEC  # C_{0->1}
    _v, _ = example_field("geomagnetic_potential", L, taper=False)
    _b_h = physics.eth(_v, 0) / 1e3  # C_{0->1}, muT
    _wind, _ = example_field("wind", L)
    _cur, _ = example_field("ocean_currents", L)
    _t, _ = example_field("cmb_planck_temperature", L, taper=False)
    _p, _ = example_field("cmb_planck_polarisation", L, taper=False)
    _kappa, _ = example_field("convergence", L, taper=False)
    _gamma = physics.kaiser_squires(_kappa)  # C_{0->2}
    _m = "mollweide"
    _fig, _axes = panels("01_overview", 14, 6.2, [[_m] * 4, [_m] * 3], print_rows=[[_m] * 3, [_m] * 2, [_m] * 2])
    mollweide(_axes[0], _topo, cmap="BrBG_r", coast=True, title="(a) topography (spin 0)", label="km")
    mollweide(_axes[1], _eps, 1, part="abs", cmap="YlOrBr", coast=True, title="(b) deflection of the vertical (spin 1)", label="arcsec")
    mollweide_glyphs(_axes[1], _eps, 1, n_glyph=500)
    mollweide(_axes[2], _b_h, 1, part="abs", coast=True, title="(c) horizontal geomagnetic field (spin 1)", label="$\\mu$T")
    mollweide_glyphs(_axes[2], _b_h, 1, n_glyph=500)
    mollweide(_axes[3], _wind, 1, part="abs", coast=True, title="(d) near-surface wind (spin 1)", label="m s$^{-1}$")
    mollweide_glyphs(_axes[3], _wind, 1, n_glyph=500)
    mollweide(_axes[4], _cur, 1, part="abs", coast=True, title="(e) ocean surface currents (spin 1)", label="m s$^{-1}$")
    mollweide_glyphs(_axes[4], _cur, 1, n_glyph=900, floor=0.0)
    mollweide(_axes[5], _t, title="(f) CMB temperature (spin 0) and polarisation (spin 2)", label="$\\mu$K")
    mollweide_glyphs(_axes[5], apply(_p, _g4, 2, 2), 2, n_glyph=500)
    mollweide(_axes[6], _kappa * _g8[:, None], title="(g) lensing convergence (spin 0) and shear (spin 2)", label="dimensionless")
    mollweide_glyphs(_axes[6], _gamma * _g8[:, None], 2, n_glyph=500)
    mo.vstack([mo.as_html(_fig), caption("01_overview", r"""
    **Figure 1.** Fields of spin 0, 1 and 2 on the Earth and on the sky, the examples of this notebook.  Top, from left to right: (a) the height of the land and the depth of the ocean floor, a scalar field, shown as colour in km, brown above and green below sea level; (b) the deflection of the vertical, the small angle, of a few arcseconds, by which the direction of gravity, i.e. of a plumb line, differs from the vertical of an idealised smooth Earth because of the uneven distribution of mass within the Earth, a spin-1 field, with its size shown as colour and its direction drawn as arrows; (c) the horizontal part of the Earth's magnetic field in 2025, the field that aligns a compass needle, a spin-1 field, with its strength shown as colour in μT and its direction drawn as arrows; (d) the wind near the surface in January, averaged over thirty years, a spin-1 field, with the wind speed shown as colour and the direction in which the wind blows drawn as arrows.  Bottom: (e) the currents at the surface of the ocean in 2025, a spin-1 field, with their speed shown as colour and the direction in which the water flows drawn as arrows, whose lengths are proportional to the speed; (f) the cosmic microwave background, the light emitted by the early Universe, as measured by the Planck satellite, with the variations of its temperature, a scalar field, shown as colour in μK, and the direction of its polarisation, a spin-2 field, drawn as headless sticks, in coordinates in which the plane of our Galaxy lies along the equator of the map; and (g) a simulation of the weak gravitational lensing of distant galaxies by the matter in the Universe, with the projected density of matter, the convergence, a scalar field, shown as colour, red where it is higher and blue where it is lower than the mean, and the shear, the direction in which the images of the galaxies are stretched, a spin-2 field, drawn as headless sticks, which circle the concentrations of mass.  The fields of (f) and (g) are smoothed so that the sticks follow their large-scale patterns, and, unless stated otherwise, the lengths of the arrows and sticks grow with the size of the field.  The fields are described in detail, with the lifted convolutions that act on them, in Figs. 2 (topography), 3 (gravity), 4 (magnetic field), 5 (wind), 7 (ocean currents), 8 (CMB) and 9 (lensing).
    """, print_wording=[("drawn as arrows; (d) the wind", "drawn as arrows.  Middle: (d) the wind"),
                        ("drawn as arrows.  Bottom: (e) the currents", "drawn as arrows; (e) the currents"),
                        ("proportional to the speed; (f) the cosmic", "proportional to the speed.  Bottom: (f) the cosmic")])])
    return


@app.function
def example_field(name, L, taper=True):
    """Harmonic coefficients of an example field at band-limit L, and its spin.  The file is read from notebooks/data,
    or downloaded from the repository on GitHub when the notebook runs without it.  A field stored at a lower band-limit
    is padded with zeros.  With taper, the coefficients are multiplied by exp(-l(l+1) / L^2)."""
    data = np.load(data_file(f"{name}.npz"))
    L_data, spin = int(data["L"]), int(data["spin"])
    stored = data["flm"].astype(np.complex128)
    if data["real"]:  # only the orders m >= 0 of a real scalar field are stored
        m = np.arange(1, L_data)
        stored = np.concatenate([(-1.0) ** m[::-1] * np.conj(stored[:, :0:-1]), stored], axis=1)
    flm = np.zeros((L, 2 * L - 1), np.complex128)
    n = min(L, L_data)
    flm[:n, L - n : L + n - 1] = stored[:n, L_data - n : L_data + n - 1]
    el = np.arange(L)
    window = np.exp(-el * (el + 1) / L**2) if taper else np.ones(L)
    return flm * window[:, None], spin


@app.function
def data_file(name):
    """A file of notebooks/data, as a path or, without the repository, as bytes downloaded from GitHub."""
    path = Path(mo.notebook_dir() or ".") / "data" / name
    if path.exists():
        return path
    url = f"https://raw.githubusercontent.com/astro-informatics/s2conv/main/notebooks/data/{name}"
    with urllib.request.urlopen(url) as response:
        return io.BytesIO(response.read())


@app.function
@functools.cache
def coastline():
    """Longitudes and latitudes (radians) of the Natural Earth coastlines, with NaN between the lines."""
    data = np.load(data_file("coastlines.npz"))
    return np.radians(data["lon"].astype(float)), np.radians(data["lat"].astype(float))


@app.function
def multiplier(L, c):
    """A harmonic multiplier c_l as an array [L], from a function of the degrees l."""
    return np.asarray(c(np.arange(L, dtype=float)), dtype=complex)


@app.function
def gaussian(L, sigma):
    """The Gaussian beam exp(-l(l+1) sigma^2 / 2) of an axisymmetric smoothing of angular width sigma."""
    return multiplier(L, lambda el: np.exp(-0.5 * el * (el + 1) * sigma**2))


@app.function
def inverse_laplacian(L):
    """The multiplier -1 / (l(l+1)) of the inverse Laplacian on the unit sphere, zero for l = 0."""
    return multiplier(L, lambda el: np.where(el > 0, -1 / np.maximum(el * (el + 1), 1), 0.0))


@app.function
def apply(flm, c, s_in, s_out):
    """The lifted convolution C_{s_in -> s_out} whose kernel is the harmonic multiplier c_l."""
    return s2conv.lifted_convolution(flm, kernels.multiplier(c, s_in, s_out), s_in, s_out)


@app.function
def real_part(flm):
    """Coefficients of the real part of a scalar field."""
    return (flm + conjugate(flm, 0)) / 2


@app.function
def imag_part(flm):
    """Coefficients of the imaginary part of a scalar field."""
    return (flm - conjugate(flm, 0)) / 2j


@app.function
def helmholtz(f_lm, radius):
    """For a spin-1 field f = v_theta + i v_phi on a sphere of the given radius: the divergence and the vorticity (the
    radial component of the curl), from eth-bar f = -radius (div v + i curl v), the curl-free and divergence-free parts
    of the field, f = eth lap^{-1} eth-bar f, and the stream function psi (curl v = lap psi), all by lifted
    convolutions C_{1->0} and C_{0->1}."""
    L = f_lm.shape[0]
    g = -physics.ethbar(f_lm, 1) / radius  # C_{1->0}: div + i curl
    div, curl = real_part(g), imag_part(g)
    inv = inverse_laplacian(L)[:, None]
    f_div = -radius * physics.eth(inv * div, 0)  # C_{0->1}
    f_rot = -radius * physics.eth(inv * 1j * curl, 0)
    return dict(div=div, curl=curl, f_div=f_div, f_rot=f_rot, psi=radius**2 * inv * curl)


@app.function
def componentwise(f_lm, s, c):
    """Apply the multiplier c_l to the real and imaginary parts of a spin-s field as if they were scalar fields, the
    frame-dependent operation that the spin formalism replaces."""
    L = f_lm.shape[0]
    f = np.asarray(transforms.inverse(f_lm, L, s))
    parts = [np.asarray(transforms.inverse(np.asarray(transforms.forward(x, L)) * c[:, None], L)).real for x in (f.real, f.imag)]
    return transforms.forward(parts[0] + 1j * parts[1], L, s)


@app.function
def l2_rel(a, b):
    """Relative L2 difference of two sets of harmonic coefficients, i.e. of the fields."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


@app.function
def robust_vmax(flm, s=0, part="real", q=99.5):
    """A colour limit at the q-th percentile of the absolute values of a field, so that a few extreme values (such as
    those over the Greenland ice sheet) do not set the colour scale."""
    f = field_grid(flm, s)[0]
    return float(np.percentile(np.abs({"real": f.real, "imag": f.imag, "abs": np.abs(f)}[part]), q))


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
    return np.asarray(transforms.inverse(upsample(flm, L_plot), L_plot, s)), transforms.sphere_angles(L_plot)


@app.function
def unit_vector(theta, phi):
    """Cartesian unit vectors of the points (theta, phi)."""
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


@app.function
def angles_of(v):
    """Colatitudes and longitudes of the unit vectors v[..., 3]."""
    return np.arccos(np.clip(v[..., 2], -1.0, 1.0)), np.mod(np.arctan2(v[..., 1], v[..., 0]), 2 * np.pi)


@app.function
def panels(name, width, height, rows, print_rows=None):
    """A figure with its panels in rows, and the axes of the panels in reading order.

    Each row lists the projection of each of its panels, "mollweide" for a map of the whole sphere or None for an
    orthographic view, and a row shorter than the longest is centred.  On screen the figure is width by height inches.
    In print mode it is at the text width of the article, with the rows of print_rows if given, and is collected
    under name for export, where fit_height sets its height so that every panel fills the width of its column."""
    if PRINT["on"]:
        rows = print_rows or rows
        fig = Figure(figsize=(PRINT["width"], height * PRINT["width"] / width), layout="constrained")
    else:
        fig = Figure(figsize=(width, height), layout="constrained")
    n_cols = max(len(row) for row in rows)
    axes, grid = [], None
    if not PRINT["on"] and all(len(row) == n_cols for row in rows):
        for k, projection in enumerate(p for row in rows for p in row):
            axes.append(fig.add_subplot(len(rows), n_cols, k + 1, projection=projection))
    else:  # each panel spans two columns of the grid, so that a shorter row can be centred
        grid = fig.add_gridspec(len(rows), 2 * n_cols)
        for r, row in enumerate(rows):
            offset = n_cols - len(row)
            for i, projection in enumerate(row):
                ax = fig.add_subplot(grid[r, offset + 2 * i : offset + 2 * i + 2], projection=projection)
                if PRINT["on"] and None in row and projection == "mollweide":
                    # beside a taller orthographic view, the map sits at the top of its cell, with its colour bar
                    # attached beneath it rather than at the foot of the cell
                    ax.set_anchor("N")
                    ax.colour_bar_attached = True
                axes.append(ax)
    if PRINT["on"]:
        PRINT["figures"][name] = fig
        PRINT["layouts"][name] = (grid, rows, axes)
    return fig, axes


@app.function
def fit_height(name, rounds=4):
    """Set the height of a print figure, and of its rows, so that every panel fills the width of its column: the
    largest panels that the text width allows.  To be called once the figure is drawn."""
    fig = PRINT["figures"][name]
    grid, rows, axes = PRINT["layouts"][name]
    aspect = {"mollweide": 0.5, None: 1.0}  # height over width of a panel
    for _ in range(rounds):
        fig.draw_without_rendering()
        width, height = fig.get_size_inches()
        panes = iter(axes)
        needed, allotted = [], []
        for row in rows:
            boxes = [(next(panes).get_position(original=True), projection) for projection in row]
            needed.append(max(box.width * width * aspect[projection] for box, projection in boxes))
            allotted.append(max(box.height for box, _ in boxes) * height)
        grid.set_height_ratios(needed)
        fig.set_size_inches(width, height + sum(needed) - sum(allotted))
    fig.draw_without_rendering()


@app.function
def caption(name, text, print_wording=()):
    """The caption of a figure as markdown, collected under name for export.  In print mode, each pair (old, new) of
    print_wording replaces the wording of the screen layout by that of the print layout."""
    if PRINT["on"]:
        for old, new in print_wording:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
    PRINT["captions"][name] = text
    return mo.md(text)


@app.function
def panel_title(ax, title):
    """Set a panel title above the panel; in print mode the panel has only its letter, beneath its colour bar."""
    if not PRINT["on"]:
        ax.set_title(title, fontsize=9)


@app.function
def panel_letter(title):
    """The letter of a panel, e.g. "(a)", from its title, in print mode; otherwise None."""
    if PRINT["on"] and title and title.startswith("("):
        return title[: title.index(")") + 1]
    return None


@app.function
def glyph_count(n):
    """Fewer glyphs in print mode, where the panels are about half as wide."""
    return max(int(n * 0.45), 100) if PRINT["on"] else n


@app.function
def colour_bar(ax, image, label, letter=None):
    """A horizontal colour bar beneath a panel, labelled with the quantity or its units, and with the letter of the
    panel, if given, centred beneath it."""
    if getattr(ax, "colour_bar_attached", False):
        # the size and spacing of the other print colour bars, in units of the height of a 2:1 map
        thickness = 2 * 0.8 / 25
        bar = ax.figure.colorbar(image, cax=ax.inset_axes([0.1, -0.03 - thickness, 0.8, thickness]), orientation="horizontal")
    else:
        bar = ax.figure.colorbar(image, ax=ax, orientation="horizontal", shrink=0.8 if PRINT["on"] else 0.7,
                                 aspect=25 if PRINT["on"] else 30, pad=0.03)
    bar.set_label(label if letter is None else f"{label}\n{letter}", fontsize=8)
    bar.locator = matplotlib.ticker.MaxNLocator(nbins=4 if PRINT["on"] else 6, symmetric=bool(image.norm.vmin < 0))
    bar.update_ticks()
    bar.ax.tick_params(labelsize=8 if PRINT["on"] else 7)


@app.function
def label_panel(ax, image, title, label):
    """Title a panel and give it a colour bar labelled with label: on screen the title is above the panel, and in
    print mode the letter of the panel is beneath its colour bar, or beneath the panel if it has none."""
    if title:
        panel_title(ax, title)
    letter = panel_letter(title)
    if label:
        colour_bar(ax, image, label, letter)
    elif letter:
        ax.text(0.5, -0.03, letter, transform=ax.transAxes, ha="center", va="top", fontsize=8)


@app.function
def mollweide_grid(ax, data, thetas, phis, cmap="RdBu_r", vmax=None, symmetric=True, title=None, coast=False, contours=None,
                   label=None):
    """Draw MW samples data[t, p] on a Mollweide axis, with longitude 0 at the centre, optionally with the coastlines,
    contour lines at the given levels and a colour bar with the given label."""
    lon = np.mod(phis + np.pi, 2 * np.pi) - np.pi
    order = np.argsort(lon)
    lon = np.append(lon[order], np.pi)
    data = np.concatenate([data[:, order], data[:, order][:, :1]], axis=1)
    if vmax is None:
        vmax = np.max(np.abs(data)) if symmetric else np.max(data)
    image = ax.pcolormesh(lon, np.pi / 2 - thetas, data, shading="gouraud", cmap=cmap,
                          vmin=-vmax if symmetric else 0.0, vmax=vmax, rasterized=True)
    if contours is not None:
        ax.contour(lon, np.pi / 2 - thetas, data, levels=contours, colors="k", linewidths=0.6)
    if coast:
        c_lon, c_lat = coastline()
        jump = np.abs(np.diff(c_lon, prepend=c_lon[0])) > np.pi  # break the lines where they cross the map's edge
        ax.plot(np.where(jump, np.nan, c_lon), c_lat, color="0.15", lw=0.3 if PRINT["on"] else 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    label_panel(ax, image, title, label)
    return image


@app.function
def mollweide(ax, flm, s=0, part="real", cmap=None, L_plot=128, vmax=None, title=None, coast=False, contours=None, label=None):
    """Mollweide map of the real part, imaginary part or modulus of a band-limited spin field."""
    f, (th, ph) = field_grid(flm, s, L_plot)
    data = {"real": f.real, "imag": f.imag, "abs": np.abs(f)}[part]
    symmetric = part != "abs"
    return mollweide_grid(ax, data, th, ph, cmap or ("RdBu_r" if symmetric else "YlGnBu"), vmax, symmetric, title, coast, contours, label)


@app.function
def mollweide_glyphs(ax, flm, s, n_glyph=700, size=0.022, color="k", lat_max=np.radians(82), floor=0.35):
    """Glyphs of a spin-s field on a Mollweide axis: arrows for spin 1 and headless sticks for spin 2, at the angle
    arg(f)/s from e_theta towards e_phi and with lengths growing with |f| from floor times the largest, at Fibonacci
    points.  The direction on
    the map is that of a short step along the glyph, mapped by the projection, so that the angle of each glyph to the
    meridian is that on the sphere."""
    n_glyph = glyph_count(n_glyph)
    k = np.arange(n_glyph) + 0.5
    theta = np.arccos(1 - 2 * k / n_glyph)
    phi = np.mod(np.pi * (1 + 5**0.5) * k, 2 * np.pi)
    keep = np.abs(np.pi / 2 - theta) < lat_max
    theta, phi = theta[keep], phi[keep]
    z = np.asarray(evaluate(flm, s, theta, phi))
    angle = np.angle(z) / s
    lon, lat = np.mod(phi + np.pi, 2 * np.pi) - np.pi, np.pi / 2 - theta
    step = 1e-3
    # give the map its final shape, 2:1, before measuring directions on the screen: until the figure is drawn the
    # axes fill their whole cell, which would stretch north-south steps and tilt the glyphs towards the vertical
    ax.apply_aspect()
    start = ax.transData.transform(np.stack([lon, lat], -1))
    end = ax.transData.transform(np.stack([lon + step * np.sin(angle) / np.cos(lat), lat - step * np.cos(angle)], -1))
    screen = np.arctan2(*(end - start)[:, ::-1].T)
    scale = 2.0 if PRINT["on"] else 1.0  # the panels are about half as wide in print
    length = scale * size * (floor + (1 - floor) * np.abs(z) / np.max(np.abs(z)))
    head = dict(headwidth=3.5, headlength=4, headaxislength=3.5) if abs(s) == 1 else dict(headwidth=0, headlength=0, headaxislength=0)
    ax.quiver(lon, lat, length * np.cos(screen), length * np.sin(screen), angles="uv", scale_units="width", scale=1,
              pivot="middle", color=color, width=scale * 0.0022, **head)


@app.function
def orthographic(ax, flm, s=0, part="real", cmap=None, lat0=0.35, lon0=0.6, glyph_flm=None, glyph_s=None,
                 n_glyph=420, glyph_len=0.07, glyph_color="k", vmax=None, title=None, L_plot=128, npix=260, coast=False,
                 symmetric=None, glyph_floor=0.35, label=None):
    """One hemisphere in orthographic projection, centred on (lat0, lon0) in radians.  Colour shows the field; for spin
    fields, glyphs with |s|-fold symmetry show the value in the local frame (e_theta, e_phi): an arrow for spin 1 and a
    headless stick for spin 2, at the angle arg(f)/s from e_theta towards e_phi.  The glyphs may be taken from another
    field (glyph_flm, glyph_s).  With coast, the coastlines are drawn, and with label a colour bar."""
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
    symmetric = (part != "abs") if symmetric is None else symmetric
    vmax = np.nanmax(np.abs(img)) if vmax is None else vmax
    image = ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap or ("RdBu_r" if symmetric else "YlGnBu"),
                      vmin=-vmax if symmetric else 0.0, vmax=vmax, interpolation="bilinear")
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=0.8, color="0.3"))
    if coast:
        c_lon, c_lat = coastline()
        xyz = unit_vector(np.pi / 2 - c_lat, c_lon) @ V
        xyz[xyz[:, 2] < 0] = np.nan  # the far side
        ax.plot(xyz[:, 0], xyz[:, 1], color="0.15", lw=0.35 if PRINT["on"] else 0.6)
    gflm, gs = (flm, s) if glyph_flm is None else (glyph_flm, glyph_s)
    if gs:
        n_glyph = glyph_count(n_glyph)
        k = np.arange(n_glyph) + 0.5
        zc = 1 - 2 * k / n_glyph
        az = np.pi * (1 + 5**0.5) * k
        pts = np.stack([np.sqrt(1 - zc**2) * np.cos(az), np.sqrt(1 - zc**2) * np.sin(az), zc], -1)
        pts = pts[pts @ out > 0.2]
        gt, gp = angles_of(pts)
        z = np.asarray(evaluate(gflm, gs, gt, gp))
        size = glyph_len * (glyph_floor + (1 - glyph_floor) * np.abs(z) / np.max(np.abs(z)))
        e_t = np.stack([np.cos(gt) * np.cos(gp), np.cos(gt) * np.sin(gp), -np.sin(gt)], -1)
        e_p = np.stack([-np.sin(gp), np.cos(gp), np.zeros_like(gp)], -1)
        angle = np.angle(z) / gs
        d = (np.cos(angle)[:, None] * e_t + np.sin(angle)[:, None] * e_p) * size[:, None]
        if abs(gs) == 1:
            a, b = (pts - 0.8 * d) @ V, 1.6 * d @ V
            ax.quiver(a[:, 0], a[:, 1], b[:, 0], b[:, 1], angles="xy", scale_units="xy", scale=1,
                      color=glyph_color, width=0.006, headwidth=4, headlength=4.5, headaxislength=4)
        else:
            a, b = (pts - d) @ V, (pts + d) @ V
            ax.add_collection(LineCollection([np.stack([p, q])[:, :2] for p, q in zip(a, b)], colors=glyph_color,
                                             linewidths=1.1, capstyle="round"))
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    label_panel(ax, image, title, label)
    return image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Topography: slopes and ridges (spin 0 → 1 → 2)

    The elevation of the land and the depth of the ocean floor, the topography and bathymetry $f$ of ETOPO 2022 (NOAA National Centers for Environmental Information), form a scalar field, but the two quantities that describe the shape of the relief are not scalars: the slope is a vector, and the orientation of a mountain range is a headless direction.

    We obtain the slope by the lifted convolution $\mathcal{C}_{0\to1}$ whose kernel is the multiplier $\sqrt{\ell(\ell+1)}$ of the spin raising operator.  Since $\eth f=-(\partial_\theta+\mathrm{i}\sin^{-1}\theta\,\partial_\varphi)f$ for a scalar field, the spin-1 field $\eth f/R_\oplus$, where $R_\oplus$ is the radius of the Earth, is the vector $-\nabla f$, which points downhill and whose length is the slope, and which therefore indicates the direction in which water flows.

    To locate mountain ranges we consider a steerable ridge detector $\psi$, a kernel elongated along one direction with a zero-mean profile across it, like a second derivative, and with azimuthal orders $|m|<5$.  Its directional convolution with the field, $f\circledast\psi$, is a function on the rotation group that gives the response of the detector at every position and every orientation $\gamma$.  Since the kernel is steerable, this function is the tuple of the nine spin fields ${}_{s'}g=\mathcal{C}_{0\to s'}[\Lambda_0\psi]f$, $|s'|<5$ (Proposition VI.6 of the article), from which the response at any orientation follows as $\sum_{s'}(2\pi)^{-1/2}e^{-\mathrm{i}s'\gamma}\,{}_{s'}g$.  The largest response over $\gamma$ measures the strength of a ridge, and the spin-2 component ${}_2g$, the part of the response that varies with orientation as $\cos2\gamma$ like that of a line, is itself the orientation of the ridges as a spin-2 field: the response is largest when the detector lies at the angle $\arg({}_2g)/2$, along the ridge.  No sampling of orientations is required, and since both outputs are rotation-equivariant, ranges are detected alike whatever their orientation and position, near the equator or near the poles.  We apply the detector to the land elevation $f_+=\max(f,0)$, since the continental margins, a step of several kilometres, would otherwise dominate its response.
    """)
    return


@app.cell
def _(L):
    _topo, _ = example_field("earth_topography", L)
    _grid, _ = field_grid(_topo, 0, 2 * L)  # clip at sea level on a finer grid, to keep the aliasing small
    _land = np.asarray(transforms.forward(np.maximum(_grid.real, 0), 2 * L))[:L, L : 3 * L - 1]
    _downhill = physics.eth(_land, 0) / (R_EARTH / 1e3) * 1e3  # C_{0->1}: -grad f_+, as a slope in m per km
    _N = 5
    _ridge = oriented_kernel(L, sig_long=2.5 * 2.6 / L, sig_short=2.6 / L, N=_N, ridge=True)
    _comps = {_q: s2conv.lifted_convolution(_land, kernels.lifted(_ridge, 0), 0, _q) for _q in range(-(_N - 1), _N)}
    _grids = {_q: field_grid(_c, _q)[0] for _q, _c in _comps.items()}
    _strength = np.max([sum(np.exp(-1j * _q * _g) * _x for _q, _x in _grids.items()).real
                        for _g in np.linspace(0, np.pi, 36, endpoint=False)], axis=0) / np.sqrt(2 * np.pi)
    _th, _ph = transforms.sphere_angles(128)

    _fig, _axes = panels("02_topography", 13, 8.4, [["mollweide"] * 3, [None] * 3])
    mollweide(_axes[0], _topo, cmap="BrBG_r", coast=True, title="(a) topography and bathymetry $f$", label="km")
    _ax = _axes[1]
    mollweide(_ax, _downhill, 1, part="abs", cmap="YlOrBr", coast=True, title="(b) slope of the land and downhill direction $\\eth f_+$", label="m/km")
    mollweide_glyphs(_ax, _downhill, 1, n_glyph=900, size=0.022, floor=0.0)
    mollweide_grid(_axes[2], np.clip(_strength, 0, None), _th, _ph, cmap="YlOrBr", symmetric=False, coast=True,
                   title="(c) ridge strength $\\max_\\gamma(f_+\\circledast\\psi)(\\cdot,\\gamma)$", label="arbitrary units")
    _view = dict(lat0=np.radians(32), lon0=np.radians(85))
    orthographic(_axes[3], _land, 0, cmap="YlOrBr", symmetric=False, **_view, glyph_flm=_downhill, glyph_s=1, coast=True, glyph_floor=0.0,
                 title="(d) Himalaya: land elevation $f_+$ and $\\eth f_+$", label="km")
    orthographic(_axes[4], _comps[2], 2, part="abs", cmap="YlOrBr", **_view, coast=True, glyph_floor=0.1,
                 title="(e) Himalaya: ridge orientation ${}_2g$", label="arbitrary units")
    orthographic(_axes[5], _comps[2], 2, part="abs", cmap="YlOrBr", lat0=np.radians(-20), lon0=np.radians(-68), coast=True, glyph_floor=0.1,
                 title="(f) Andes: ridge orientation ${}_2g$", label="arbitrary units")
    mo.vstack([mo.as_html(_fig), caption("02_topography", r"""
    **Figure 2.** The topography of the Earth, its slope and its ridges, from lifted convolutions of spin 0 to spin 1 and to spin 2.  Top, from left to right: (a) the topography and bathymetry $f$, shown as colour in km on a diverging scale, brown above and green below sea level; (b) the slope of the land $|\eth f_+|/R_\oplus$, shown as colour in metres per kilometre, with the spin-1 field $\eth f_+$ drawn as arrows pointing downhill, whose lengths are proportional to the slope; (c) the ridge strength $\max_\gamma(f_+\circledast\psi)(\cdot,\gamma)$, the largest response of the ridge detector over all orientations, shown as colour on a sequential scale in arbitrary units.  Bottom: orthographic views of (d) the Himalaya and the Tibetan Plateau, centred on 32°N, 85°E, with the land elevation $f_+$ shown as colour in km and the downhill arrows of (b); (e) the same view, with the modulus of the ridge orientation field $|{}_2g|$ shown as colour and ${}_2g$ drawn as headless sticks at the angle $\arg({}_2g)/2$, whose lengths grow in proportion to $|{}_2g|$ from a tenth of the largest length; and (f) the same for the Andes, centred on 20°S, 68°W.  The arrows run down the flanks of the plateau, the ridge strength picks out the Himalaya, the Andes, the Rocky Mountains, the highlands of East Africa and the ice sheets of Greenland and Antarctica, and where the ridge strength is large the sticks lie along the ranges, from east to west along the Himalaya and from north to south along the Andes.
    """)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Gravity: one potential and its derivatives of spin 0, 1 and 2

    The geoid is the surface of constant gravity potential that coincides with mean sea level over the oceans, and its height $N$ above the reference ellipsoid is a scalar field, which we take from the Earth Gravitational Model EGM2008 (Pavlis *et al.* 2012).  By Bruns' formula $N$ is the disturbing potential $T$, the difference between the gravity potential of the Earth and that of the ellipsoid, divided by the normal gravity $\gamma$, and the quantities that geodesy measures are derivatives of $T$ of increasing spin.  The gravity anomaly, measured by gravimeters, is the scalar field with coefficients $\Delta g_{\ell m}=\frac{\gamma}{R_\oplus}(\ell-1)N_{\ell m}$ in the spherical approximation, i.e. the lifted convolution $\mathcal{C}_{0\to0}$ with the multiplier $\frac{\gamma}{R_\oplus}(\ell-1)$.  The deflection of the vertical, the angle between the direction of gravity, i.e. the plumb line, and the normal to the ellipsoid, is measured by astronomical positioning and is the vector $-\nabla N/R_\oplus$, i.e. the spin-1 field $\varepsilon=\eth N/R_\oplus$ given by $\mathcal{C}_{0\to1}$ with the multiplier $\sqrt{\ell(\ell+1)}/R_\oplus$.  The horizontal, trace-free part of the tensor of second derivatives of $T$, measured by the gradiometer of the GOCE satellite, is the spin-2 field $\Gamma=\tfrac12\eth^2T/R_\oplus^2$ given by $\mathcal{C}_{0\to2}$.

    The classical integral formulae of physical geodesy are lifted convolutions between these spins.  The formula of Vening Meinesz, which gives the deflection of the vertical from gravity anomalies, is the spin-changing convolution $\mathcal{C}_{0\to1}$ with the multiplier $\sqrt{\ell(\ell+1)}/(\gamma(\ell-1))$.  The recovery of the geoid from deflections of the vertical, known as astrogeodetic levelling, is the lifted convolution $\mathcal{C}_{1\to0}$ that inverts $\mathcal{C}_{0\to1}$, and the recovery of the geoid from its gradients is the lifted convolution $\mathcal{C}_{2\to0}$ that inverts $\tfrac12\eth^2$, the operator that also underlies the Kaiser–Squires relation between the shear and the convergence of weak lensing.  Each of these relations is a single multiplier in harmonic space and holds to machine precision for band-limited fields, as the relative errors reported in the caption of Fig. 3 confirm.
    """)
    return


@app.cell
def _(L):
    _N, _ = example_field("geoid", L)
    _dg = apply(_N, multiplier(L, lambda el: GAMMA / R_EARTH * (el - 1) * 1e5), 0, 0)  # C_{0->0}: gravity anomaly (mGal)
    _eps = physics.eth(_N, 0) / R_EARTH * ARCSEC  # C_{0->1}: deflection of the vertical (arcsec)
    _vm = multiplier(L, lambda el: np.where(el > 1, np.sqrt(el * (el + 1)) / np.maximum(el - 1, 1) / (GAMMA * 1e5) * ARCSEC, 0))
    _eps_vm = apply(_dg, _vm, 0, 1)  # Vening Meinesz: C_{0->1} from gravity anomalies
    _grad = physics.eth(physics.eth(_N, 0), 1) * GAMMA / (2 * R_EARTH**2) * 1e9  # C_{0->2}: gradients (Eotvos)
    _N_from_eps = apply(_eps / ARCSEC * R_EARTH, inverse_laplacian(L) * physics.ethbar_multiplier(L, 1), 1, 0)  # C_{1->0}
    _inv = multiplier(L, lambda el: np.where(el > 1, 1 / np.sqrt(np.maximum((el - 1) * el * (el + 1) * (el + 2), 1)), 0))
    _N_from_grad = apply(_grad * 2 * R_EARTH**2 / (GAMMA * 1e9), _inv, 2, 0)  # C_{2->0}
    _errs = (l2_rel(_eps_vm, _eps), l2_rel(_N_from_eps, _N), l2_rel(_N_from_grad, _N))

    _fig, _axes = panels("03_gravity", 13, 8.4, [["mollweide"] * 3, [None] * 3])
    mollweide(_axes[0], _N, coast=True, title="(a) geoid height $N$", label="m")
    mollweide(_axes[1], _dg, coast=True, vmax=robust_vmax(_dg), title="(b) gravity anomaly $\\Delta g=\\mathcal{C}_{0\\to0}N$", label="mGal")
    _ax = _axes[2]
    mollweide(_ax, _eps, 1, part="abs", cmap="YlOrBr", coast=True, title="(c) deflection of the vertical $\\varepsilon=\\mathcal{C}_{0\\to1}N$", label="arcsec")
    mollweide_glyphs(_ax, _eps, 1, n_glyph=800, size=0.018)
    orthographic(_axes[3], _N, 0, lat0=np.radians(-5), lon0=np.radians(80), glyph_flm=_eps, glyph_s=1, coast=True,
                 title="(d) Indian Ocean: $N$ and $\\varepsilon$", label="m")
    orthographic(_axes[4], _grad, 2, part="abs", cmap="YlOrBr", lat0=np.radians(-20), lon0=np.radians(-75), coast=True,
                 title="(e) Andes: gravity gradients $\\Gamma=\\mathcal{C}_{0\\to2}N$", label="E")
    orthographic(_axes[5], _N_from_eps, 0, lat0=np.radians(-5), lon0=np.radians(140), coast=True,
                 title="(f) New Guinea: $N$ recovered from $\\varepsilon$ by $\\mathcal{C}_{1\\to0}$", label="m")
    mo.vstack([mo.as_html(_fig), caption("03_gravity", r"""
    **Figure 3.** The gravity field of the Earth from EGM2008, as fields of spin 0, 1 and 2 related by lifted convolutions.  Top, from left to right: (a) the geoid height $N$, shown as colour in metres on a diverging scale, red above and blue below the reference ellipsoid; (b) the gravity anomaly $\Delta g=\mathcal{C}_{0\to0}N$, shown as colour in mGal ($10^{-5}$ m s$^{-2}$) on a diverging scale saturated at the 99.5th percentile of its modulus; (c) the modulus of the deflection of the vertical $\varepsilon=\mathcal{C}_{0\to1}N$, shown as colour in arcseconds, with $\varepsilon$ drawn as arrows, which point down the slope of the geoid and whose lengths grow in proportion to $|\varepsilon|$ from 35 per cent of the largest length.  Bottom, orthographic views: (d) the Indian Ocean, centred on 5°S, 80°E, with $N$ shown as colour in metres and $\varepsilon$ drawn as arrows as in (c); (e) the Andes, centred on 20°S, 75°W, with the modulus of the gravity gradients $\Gamma=\mathcal{C}_{0\to2}N$ shown as colour in Eötvös ($10^{-9}$ s$^{-2}$) and $\Gamma$ drawn as headless sticks along the horizontal direction in which the second derivative of the disturbing potential is largest, whose lengths grow in proportion to $|\Gamma|$ from 35 per cent of the largest length; and (f) Indonesia and New Guinea, centred on 5°S, 140°E, with the geoid height recovered from the deflections of the vertical by $\mathcal{C}_{1\to0}$ shown as colour in metres.  The deflections point into the geoid low of the Indian Ocean and away from the geoid high over New Guinea, and the gradients are largest along the Andes and the adjacent trench.  The formula of Vening Meinesz, $\mathcal{C}_{0\to1}$ applied to $\Delta g$, reproduces $\varepsilon$ to a relative error of %.0e, and the geoid is recovered from $\varepsilon$ and from $\Gamma$ to relative errors of %.0e and %.0e.
    """ % _errs)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. The geomagnetic field: a potential field of spin 0 and 1, continued to the core

    Outside its sources the main magnetic field of the Earth, generated by the dynamo in the fluid outer core, is the gradient of a potential, $\mathbf{B}=-\nabla V$, which the International Geomagnetic Reference Field provides as spherical harmonic coefficients up to degree 13 (IGRF-14, epoch 2025; the model is described by Alken *et al.* 2021).  With $v=V/R_\oplus$ on the surface of the Earth, the radial component of the field is the scalar field $B_r=\mathcal{C}_{0\to0}[\ell+1]\,v$, and the horizontal component is the spin-1 field $B_\theta+\mathrm{i}B_\varphi=\eth v=\mathcal{C}_{0\to1}[\sqrt{\ell(\ell+1)}]\,v$, which is the field that aligns a compass needle.  The declination, the angle by which the needle deviates east of geographic north, is the direction of the horizontal field measured from north, and the inclination, the angle by which the field dips below the horizontal, follows from the ratio of $-B_r$ to the modulus of the horizontal field.  The magnetic equator is the line of zero inclination, and the agonic lines are those of zero declination.

    Each degree $\ell$ of a potential field of internal origin decays with radius as $r^{-(\ell+2)}$, so that the field at the boundary between the core and the mantle, at radius $r_c=3485$ km, follows from that at the surface by the multiplier $(R_\oplus/r_c)^{\ell+2}$, the so-called downward continuation.  The multiplier preserves spin, and we apply it to the radial field as $\mathcal{C}_{0\to0}$ and to the horizontal vector field as $\mathcal{C}_{1\to1}$, directly and without reference to its components.  Downward continuation amplifies the coefficients of degree 13 by $(R_\oplus/r_c)^{15}\approx8500$ and reveals the structure of the field at the top of the core: intense patches of flux at high latitudes, and patches of reversed flux beneath the South Atlantic and southern Africa, which weaken the surface field over the South Atlantic in the so-called South Atlantic Anomaly.
    """)
    return


@app.cell
def _(L):
    _v, _ = example_field("geomagnetic_potential", L, taper=False)
    _b_r = apply(_v, multiplier(L, lambda el: el + 1), 0, 0)  # C_{0->0}
    _b_h = physics.eth(_v, 0)  # C_{0->1}: B_theta + i B_phi
    _core = multiplier(L, lambda el: (R_EARTH / R_CORE) ** (el + 2))
    _b_r_core, _b_h_core = apply(_b_r, _core, 0, 0) / 1e6, apply(_b_h, _core, 1, 1) / 1e6  # C_{0->0}, C_{1->1} (mT)
    _f_h, (_th, _ph) = field_grid(_b_h, 1)
    _f_r, _ = field_grid(_b_r, 0)
    _decl = np.degrees(np.arctan2(_f_h.imag, -_f_h.real))  # east of north
    _incl = np.degrees(np.arctan2(-_f_r.real, np.abs(_f_h)))

    _m = "mollweide"
    _fig, _axes = panels("04_geomagnetism", 13, 8.4, [[_m] * 3, [_m, _m, None]])
    mollweide(_axes[0], _b_r / 1e3, coast=True, title="(a) radial field $B_r=\\mathcal{C}_{0\\to0}v$", label="$\\mu$T")
    _ax = _axes[1]
    mollweide(_ax, _b_h / 1e3, 1, part="abs", coast=True, title="(b) horizontal field $\\eth v=\\mathcal{C}_{0\\to1}v$", label="$\\mu$T")
    mollweide_glyphs(_ax, _b_h, 1, n_glyph=600)
    mollweide_grid(_axes[2], _decl, _th, _ph, vmax=30, coast=True, contours=[-20, -10, 0, 10, 20],
                   title="(c) declination", label="degrees east of north")
    mollweide_grid(_axes[3], _incl, _th, _ph, vmax=90, coast=True, contours=[0],
                   title="(d) inclination", label="degrees below the horizontal")
    mollweide(_axes[4], _b_r_core, coast=True, title="(e) $B_r$ at the core–mantle boundary, $\\mathcal{C}_{0\\to0}$", label="mT")
    orthographic(_axes[5], _b_r_core, 0, lat0=np.pi / 2 - 1e-3, lon0=0.0, glyph_flm=_b_h_core, glyph_s=1, coast=True,
                 title="(f) core–mantle boundary beneath the Arctic", label="mT")
    mo.vstack([mo.as_html(_fig), caption("04_geomagnetism", r"""
    **Figure 4.** The geomagnetic field of IGRF-14 in 2025, as fields of spin 0 and 1, and its downward continuation to the core.  Top, from left to right: (a) the radial field $B_r=\mathcal{C}_{0\to0}v$ at the surface, shown as colour in μT on a diverging scale, blue where the field points into the Earth, as in the northern hemisphere, and red where it points out; (b) the modulus of the horizontal field $\eth v=\mathcal{C}_{0\to1}v$, shown as colour in μT on a sequential scale, with $\eth v$ drawn as arrows in the direction in which a compass needle points, whose lengths grow in proportion to the modulus from 35 per cent of the largest length; (c) the declination, shown as colour in degrees east of geographic north on a diverging scale saturated at ±30°, red to the east and blue to the west, with contours every 10°, solid to the east and dashed to the west, among which the solid contours of 0° are the agonic lines.  Bottom: (d) the inclination, shown as colour in degrees below the horizontal, red where the field dips into the Earth and blue where it points out of it, with the magnetic equator as the contour of zero inclination; (e) the radial field continued by $\mathcal{C}_{0\to0}$ to the boundary between the core and the mantle, shown as colour in mT on a diverging scale as in (a); and (f) an orthographic view of the same boundary centred on the North pole, with the continued radial field shown as colour in mT and the horizontal field continued by $\mathcal{C}_{1\to1}$ drawn as arrows as in (b), with the coastlines projected radially onto the boundary for orientation.  The compass needles point towards the north magnetic pole in the Canadian Arctic, the declination is small over Europe and Africa and changes sign across the agonic lines, and at the core the flux is concentrated in patches at high latitudes, with reversed flux, red in the southern hemisphere, beneath the South Atlantic and southern Africa.
    """)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Wind: vorticity, divergence and smoothing (spin 1)

    We consider the near-surface wind in January, averaged over 1991–2020, from the NCEP/NCAR reanalysis (Kalnay *et al.* 1996).  With northward and eastward components $(v,u)$, and since $e_\theta$ points south, the wind is the spin-1 field ${}_1f=-v+\mathrm{i}u$.

    For a spin-1 field the spin lowering operator gives the divergence and the vorticity, i.e. the radial component of the curl, together, $\bar\eth\,{}_1f=-R_\oplus(\nabla\cdot\mathbf{v}+\mathrm{i}\,\hat{\mathbf{r}}\cdot\nabla\times\mathbf{v})$, so that the single lifted convolution $\mathcal{C}_{1\to0}$ yields both as the real and imaginary parts of one scalar field, exactly for the band-limited field, without finite differences and without special treatment of the poles.  Since $\eth\bar\eth$ is the Laplacian on spin-1 fields, applying the lifted convolution $\mathcal{C}_{0\to1}$ of $\eth\nabla^{-2}$ to the divergence and to the vorticity separately gives the Helmholtz decomposition of the wind into a curl-free part, the flow into and out of regions of convergence, and a divergence-free part, the rotational flow of the trade winds and the westerlies, which sum to the wind exactly.

    Smoothing the wind by a Gaussian is the spin-preserving lifted convolution $\mathcal{C}_{1\to1}$ whose kernel is the Gaussian multiplier.  A common alternative smooths the components $v_\theta$ and $v_\varphi$ as though they were two scalar fields.  Since the frame $(e_\theta,e_\varphi)$ turns around the poles of the coordinates, this averages vectors expressed in different frames, and near a pole the components vary rapidly even where the wind itself is uniform.  The poles of the coordinates are, however, a matter of convention, and to expose the dependence of each method on that convention we place one pole at the location of the strongest smoothed wind and smooth in both systems of coordinates.  The spin-1 smoothing, which is rotation-equivariant, gives the same result in either system to machine precision.  The component-wise smoothing gives a plausible result in geographic coordinates, where the strongest winds lie far from the poles, but with the pole placed in the strong flow it combines the components across the turning frame and destroys the flow near the pole, so that its result depends on the choice of coordinates.
    """)
    return


@app.cell
def _(L):
    wind, _ = example_field("wind", L)
    return (wind,)


@app.cell
def _(wind):
    _wind = wind
    _h = helmholtz(_wind, R_EARTH)
    _view = dict(lat0=np.radians(20), lon0=np.radians(-150))
    _fig, _axes = panels("05_wind", 13, 8.4, [["mollweide"] * 3, [None] * 3])
    _ax = _axes[0]
    mollweide(_ax, _wind, 1, part="abs", coast=True, title="(a) January near-surface wind ${}_1f$", label="m s$^{-1}$")
    mollweide_glyphs(_ax, _wind, 1)
    mollweide(_axes[1], _h["curl"] * 1e5, coast=True, vmax=robust_vmax(_h["curl"] * 1e5),
              title="(b) vorticity $-\\mathrm{Im}\\,\\bar\\eth\\,{}_1f/R_\\oplus$", label="$10^{-5}$ s$^{-1}$")
    mollweide(_axes[2], _h["div"] * 1e5, coast=True, vmax=robust_vmax(_h["div"] * 1e5),
              title="(c) divergence $-\\mathrm{Re}\\,\\bar\\eth\\,{}_1f/R_\\oplus$", label="$10^{-5}$ s$^{-1}$")
    orthographic(_axes[3], _wind, 1, part="abs", **_view, coast=True, title="(d) the wind over the North Pacific", label="m s$^{-1}$")
    orthographic(_axes[4], _h["f_rot"], 1, part="abs", **_view, coast=True, title="(e) divergence-free part", label="m s$^{-1}$")
    orthographic(_axes[5], _h["f_div"], 1, part="abs", **_view, coast=True, title="(f) curl-free part", label="m s$^{-1}$")
    _sum_err = l2_rel(_h["f_rot"] + _h["f_div"], _wind)
    mo.vstack([mo.as_html(_fig), caption("05_wind", r"""
    **Figure 5.** The January near-surface wind and its vorticity, divergence and Helmholtz decomposition, from the lifted convolutions $\mathcal{C}_{1\to0}$ and $\mathcal{C}_{0\to1}$.  Top, from left to right: (a) the wind speed $|{}_1f|$, shown as colour in m s$^{-1}$ on a sequential scale, with the wind ${}_1f$ drawn as arrows pointing downwind, whose lengths grow in proportion to the speed from 35 per cent of the largest length; (b) the vorticity $-\mathrm{Im}(\bar\eth\,{}_1f)/R_\oplus$ and (c) the divergence $-\mathrm{Re}(\bar\eth\,{}_1f)/R_\oplus$, each shown as colour in $10^{-5}$ s$^{-1}$ on a diverging scale saturated at the 99.5th percentile of its modulus, so that the extreme values over the steep margins of the Greenland and Antarctic ice sheets do not set the scale, red for anticlockwise rotation and for outflow and blue for clockwise rotation and for inflow.  Bottom: orthographic views of the North Pacific, centred on 20°N, 150°W, of (d) the wind, (e) its divergence-free part and (f) its curl-free part, each with its speed shown as colour in m s$^{-1}$ and drawn as arrows as in (a).  The vorticity is positive in the cyclonic flow around the Aleutian and Icelandic lows and negative in the anticyclones of the northern subtropics, and the divergence is negative along the intertropical convergence zone, into which the trade winds of both hemispheres flow.  The divergence-free part carries the trade winds and the westerlies, the curl-free part carries the flow into the convergence zone and out of the subtropical highs, and the two parts sum to the wind to a relative error of %.0e.
    """ % _sum_err)])
    return


@app.cell
def _(L, wind):
    _wind = wind
    _beam = gaussian(L, 8 / L)
    _spin = apply(_wind, _beam, 1, 1)
    _speed, (_th, _ph) = field_grid(_spin, 1)
    _t, _p = np.unravel_index(np.argmax(np.abs(_speed)), _speed.shape)
    _lat, _lon = np.pi / 2 - _th[_t], _ph[_p]  # the strongest smoothed wind
    _to_pole = lambda _x: rotate_flm(_x, (0.0, -(np.pi / 2 - _lat), -_lon))  # moves (_lat, _lon) to the North pole
    _back = lambda _x: rotate_flm(_x, (_lon, np.pi / 2 - _lat, 0.0))
    _spin_moved = _back(apply(_to_pole(_wind), _beam, 1, 1))
    _comp = componentwise(_wind, 1, _beam)
    _comp_moved = _back(componentwise(_to_pole(_wind), 1, _beam))
    _jet = dict(lat0=_lat, lon0=_lon, coast=True, n_glyph=1400, glyph_len=0.05)
    _vmax = float(np.max(np.abs(_speed)))
    _fig2, _axes = panels("06_wind_smoothing", 13, 4.6, [[None] * 4], print_rows=[[None] * 2, [None] * 2])
    orthographic(_axes[0], _spin, 1, part="abs", vmax=_vmax, **_jet, title="(a) spin-1 smoothing $\\mathcal{C}_{1\\to1}$", label="m s$^{-1}$")
    orthographic(_axes[1], _comp, 1, part="abs", vmax=_vmax, **_jet, title="(b) components as scalars,\ngeographic coordinates", label="m s$^{-1}$")
    orthographic(_axes[2], _comp_moved, 1, part="abs", vmax=_vmax, **_jet, title="(c) components as scalars,\npole at the centre", label="m s$^{-1}$")
    orthographic(_axes[3], _comp_moved - _spin, 1, part="abs", cmap="Greys", vmax=_vmax, glyph_floor=0.0, **_jet,
                 title="(d) difference of (c) and (a)", label="m s$^{-1}$")
    mo.vstack([mo.as_html(_fig2), caption("06_wind_smoothing", r"""
    **Figure 6.** Smoothing the wind as a spin-1 field and as two scalar components.  Orthographic views centred on the strongest wind after smoothing, at %.0f°%s, %.0f°%s, of the wind smoothed by a Gaussian of width $8/L$ radians, each with its speed shown as colour in m s$^{-1}$ on a common sequential scale and drawn as arrows as in Fig. 5.  From left to right: (a) the smoothing by the spin-1 convolution $\mathcal{C}_{1\to1}$, which, with the pole of the coordinates placed at the centre of the view, gives the same result to a relative error of %.0e; (b) the smoothing of the components $v_\theta$ and $v_\varphi$ as scalar fields in geographic coordinates; (c) the same smoothing of the components with the pole of the coordinates at the centre of the view; and (d) the modulus of the difference between (c) and (a), shown as colour on the same scale, from white to black, with the difference drawn as arrows whose lengths are proportional to its modulus.  The component-wise smoothing agrees closely with the spin-1 smoothing in geographic coordinates, but with the pole at the centre it destroys the flow within about the width of the smoothing of the pole, where the frame turns fastest, with a relative error of %.0e over the sphere.
    """ % (abs(np.degrees(_lat)), "N" if _lat >= 0 else "S", abs(((np.degrees(_lon) + 180) % 360) - 180), "E" if ((np.degrees(_lon) + 180) % 360) - 180 >= 0 else "W",
           l2_rel(_spin_moved, _spin), l2_rel(_comp_moved, _spin)),
        print_wording=[("From left to right: (a)", "Top, from left to right: (a)"),
                       ("in geographic coordinates; (c) the same", "in geographic coordinates.  Bottom: (c) the same")])])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Ocean currents: gyres and upwelling (spin 1)

    The surface currents of the ocean in 2025, averaged over the year at a depth of 5 m, from the Global Ocean Data Assimilation System (GODAS; Behringer & Xue 2004), form a spin-1 field like the wind, which we set to zero on land.  The lifted convolution $\mathcal{C}_{1\to0}$ again gives their vorticity and divergence.  The stream function $\psi=R_\oplus^2\nabla^{-2}(\hat{\mathbf{r}}\cdot\nabla\times\mathbf{v})$ of the divergence-free part, obtained from the vorticity by the inverse Laplacian, has as its contours the streamlines of that flow, along which the rotational part of the current runs with higher values of $\psi$ to its right; it is defined over the whole sphere, including the land, where it has no physical meaning.  The streamlines trace the subtropical gyres, which circulate clockwise in the northern hemisphere and anticlockwise in the southern hemisphere, and the Antarctic Circumpolar Current.  The divergence of the surface flow marks upwelling, where water rises from below to replace surface water that is carried away.  The trade winds drive an Ekman transport of the surface water to the right of the wind in the northern hemisphere and to its left in the southern hemisphere, i.e. away from the equator on both sides, so that the surface flow diverges along the equator, in the cold tongues of the eastern Pacific and Atlantic, and along the coasts of Peru and north-west Africa, where the winds blow along the coast.
    """)
    return


@app.cell
def _(L):
    _cur, _ = example_field("ocean_currents", L)
    _h = helmholtz(_cur, R_EARTH)
    _psi = _h["psi"] / 1e6  # 10^6 m^2/s
    _psi_grid, _ = field_grid(_psi)
    _levels = np.linspace(-1, 1, 13) * np.max(np.abs(_psi_grid.real))
    _div = _h["div"] * 1e7
    _m = "mollweide"
    _fig, _axes = panels("07_ocean_currents", 13, 8.6, [[_m, _m], [_m, None]])
    _ax = _axes[0]
    mollweide(_ax, _cur, 1, part="abs", coast=True, title="(a) surface currents ${}_1f$, 2025", label="m s$^{-1}$")
    mollweide_glyphs(_ax, _cur, 1, n_glyph=1200, size=0.02, floor=0.0)
    mollweide(_axes[1], _psi, coast=True, contours=_levels,
              title="(b) stream function $\\psi$ and streamlines", label="$10^6$ m$^2$ s$^{-1}$")
    mollweide(_axes[2], _div, coast=True, vmax=robust_vmax(_div),
              title="(c) divergence $-\\mathrm{Re}\\,\\bar\\eth\\,{}_1f/R_\\oplus$", label="$10^{-7}$ s$^{-1}$")
    orthographic(_axes[3], _div, 0, lat0=np.radians(-5), lon0=np.radians(-110), glyph_flm=_cur, glyph_s=1, coast=True, glyph_floor=0.0,
                 vmax=robust_vmax(_div), title="(d) eastern equatorial Pacific", label="$10^{-7}$ s$^{-1}$")
    mo.vstack([mo.as_html(_fig), caption("07_ocean_currents", r"""
    **Figure 7.** The surface currents of the ocean in 2025, their stream function and their divergence, from the lifted convolution $\mathcal{C}_{1\to0}$.  Top: (a) the current speed $|{}_1f|$, shown as colour in m s$^{-1}$ on a sequential scale, with the current drawn as arrows pointing downstream, whose lengths are proportional to the speed; (b) the stream function $\psi$ of the divergence-free part, shown as colour in $10^6$ m$^2$ s$^{-1}$ on a diverging scale, with twelve equally spaced contours, the streamlines, drawn solid for positive and dashed for negative values.  Bottom: (c) the divergence $-\mathrm{Re}(\bar\eth\,{}_1f)/R_\oplus$, shown as colour in $10^{-7}$ s$^{-1}$ on a diverging scale saturated at the 99.5th percentile of its modulus, red where the surface flow diverges and water upwells and blue where it converges; and (d) an orthographic view of the eastern equatorial Pacific, centred on 5°S, 110°W, with the divergence shown as colour as in (c) and the currents drawn as arrows as in (a).  The fastest currents are the westward equatorial currents of the Pacific, Atlantic and Indian Oceans; the subtropical gyres appear as maxima of $\psi$, red, in the northern hemisphere and minima, blue, in the southern hemisphere; and the divergence is concentrated along the equator in the Pacific and Atlantic and along the coast of Peru, where the equatorial currents diverge from the equator.
    """)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Polarisation of the cosmic microwave background (spin 2)

    The linear polarisation of the cosmic microwave background (CMB), described by the Stokes parameters $Q$ and $U$, is the spin-2 field $Q+\mathrm{i}U$.  Its physical content is carried by two scalar fields, the $E$ and $B$ modes, which do not depend on the frame and which separate the contributions of different physical processes: density perturbations in the early Universe generate only $E$ modes, while $B$ modes are generated by the gravitational lensing of $E$ modes and would be generated by primordial gravitational waves.  By Proposition VI.8(b) of the article, $a_E=-\tfrac12(a_2+a_{-2})$ and $a_B=\tfrac{\mathrm{i}}2(a_2-a_{-2})$, where $a_{\pm2}$ are the coefficients of $Q\pm\mathrm{i}U$, so that each mode is a sum of the lifted convolutions $\mathcal{C}_{2\to0}$ and $\mathcal{C}_{-2\to0}$.  Conversely, the polarisation follows from the two modes by the lifted convolutions $\mathcal{C}_{0\to\pm2}$, as $a_{\pm2}=-(a_E\pm\mathrm{i}a_B)$, so that the decomposition may be inverted exactly.

    We consider the Planck 2018 SMICA maps of the temperature and polarisation of the CMB (Planck Collaboration 2020), in Galactic coordinates, so that the plane of the Galaxy lies along the equator of the Mollweide maps.  We decompose the polarisation into its $E$ and $B$ modes, and synthesise the polarisation again from the two modes.  On the large scales considered here, the polarisation measured by Planck combines the signal with noise and residual Galactic emission, which is most evident along the Galactic plane and which an analysis would mask.
    """)
    return


@app.cell
def _(L):
    _t, _ = example_field("cmb_planck_temperature", L, taper=False)
    _p, _ = example_field("cmb_planck_polarisation", L, taper=False)
    _e, _b = physics.eb_decomposition(_p)  # C_{2->0} + C_{-2->0}
    _p_back, _ = physics.eb_to_polarisation(_e, _b)  # C_{0->2}
    # the round trip between maps, through the harmonic transforms: the polarisation map to the E and B maps and back
    _p_map = np.asarray(transforms.inverse(_p, L, 2))
    _e_map, _b_map = (np.asarray(transforms.inverse(_x, L)).real for _x in physics.eb_decomposition(transforms.forward(_p_map, L, 2)))
    _p_map_back = np.asarray(transforms.inverse(physics.eb_to_polarisation(transforms.forward(_e_map, L), transforms.forward(_b_map, L))[0], L, 2))
    _map_err = float(np.linalg.norm(_p_map_back - _p_map) / np.linalg.norm(_p_map))
    _beam = gaussian(L, 4 / L)
    _smooth = lambda _x: apply(_x, _beam, 2, 2)  # C_{2->2}, for the sticks
    _fig, _axes = panels("08_cmb_polarisation", 13, 7.4, [["mollweide"] * 3, ["mollweide"] * 2])
    _ax = _axes[0]
    mollweide(_ax, _t, title="(a) temperature and polarisation", label="$\\mu$K")
    mollweide_glyphs(_ax, _smooth(_p), 2)
    _ax = _axes[1]
    mollweide(_ax, _p, 2, part="abs", title="(b) polarisation $Q+\\mathrm{i}U$", label="$\\mu$K")
    mollweide_glyphs(_ax, _smooth(_p), 2)
    mollweide(_axes[2], _e, title="(c) $E$ mode, $\\mathcal{C}_{2\\to0}+\\mathcal{C}_{-2\\to0}$", label="$\\mu$K")
    mollweide(_axes[3], _b, title="(d) $B$ mode, $\\mathcal{C}_{2\\to0}+\\mathcal{C}_{-2\\to0}$", label="$\\mu$K")
    _ax = _axes[4]
    mollweide(_ax, _p_back, 2, part="abs", title="(e) polarisation from $E$ and $B$, $\\mathcal{C}_{0\\to2}$", label="$\\mu$K")
    mollweide_glyphs(_ax, _smooth(_p_back), 2)
    mo.vstack([mo.as_html(_fig), caption("08_cmb_polarisation", r"""
    **Figure 8.** The temperature and polarisation of the CMB measured by Planck, the $E$ and $B$ modes of the polarisation from the lifted convolutions $\mathcal{C}_{2\to0}+\mathcal{C}_{-2\to0}$, and the polarisation synthesised again from the two modes by $\mathcal{C}_{0\to2}$, in Galactic coordinates.  Top, from left to right: (a) the temperature anisotropy, shown as colour in μK on a diverging scale, red hotter and blue colder than the mean, with the polarisation, smoothed by the spin-2 convolution $\mathcal{C}_{2\to2}$ with a Gaussian of width $4/L$ radians so that the sticks follow its large-scale pattern, drawn as headless sticks along the direction of polarisation, at the angle $\arg(Q+\mathrm{i}U)/2$, whose lengths grow in proportion to the polarised intensity from 35 per cent of the largest length; (b) the polarised intensity $|Q+\mathrm{i}U|$, shown as colour in μK on a sequential scale, with the same sticks; (c) the $E$ mode, shown as colour in μK on a diverging scale.  Bottom: (d) the $B$ mode, shown as colour in μK on a diverging scale; (e) the polarised intensity of the polarisation synthesised from the $E$ and $B$ modes, shown as colour in μK as in (b), with its smoothed polarisation drawn as sticks as in (a).  Both modes show residual Galactic emission along the equator of the maps, and the $B$ mode, which on these scales is dominated by noise and by these residuals, has an rms of %.0f per cent of that of the $E$ mode.  The decomposition and its inverse are single multipliers in harmonic space, $-\tfrac12$ and $\pm\tfrac{\mathrm{i}}2$ one way and $-1$ and $\mp\mathrm{i}$ the other, so that the polarisation synthesised from the two modes reproduces the measured polarisation to the precision of the arithmetic.  Starting from the map of the polarisation, computing the maps of the $E$ and $B$ modes and synthesising the map of the polarisation again, including all the harmonic transforms, recovers the map to a relative $L^2$ error of %.1e, set by the rounding errors of the transforms.
    """ % (100 * np.linalg.norm(np.asarray(_b)) / np.linalg.norm(np.asarray(_e)), _map_err))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Weak gravitational lensing: shear and convergence (spin 0 ↔ 2)

    The images of distant galaxies are distorted by the gravitational deflection of their light by the matter along the line of sight.  The convergence $\kappa$, the projected density of matter weighted by the geometry of the lensing, is a scalar field, and the shear $\gamma$, the anisotropic stretching of the images, is a spin-2 field.  The two are related by the spherical Kaiser–Squires relation (Kaiser & Squires 1993; Wallis *et al.* 2022), which is the lifted convolution $\gamma=\mathcal{C}_{0\to2}[\mathcal{D}]\kappa$ with $\mathcal{D}_\ell=-\sqrt{(\ell+2)(\ell-1)/(\ell(\ell+1))}$ (Proposition VI.8(c) of the article), and whose inverse $\kappa=\mathcal{C}_{2\to0}[\mathcal{D}^{-1}]\gamma$ recovers the convergence, apart from its monopole and dipole, from the shear, on the full sphere and without the planar approximation of the original method.  Since the galaxies are stretched tangentially around concentrations of mass, the sticks of the shear circle the overdense regions and point radially away from the underdense regions.

    We simulate the convergence as a shifted lognormal field (Hilbert *et al.* 2011), which captures the skewness of the distribution of matter, with a realistic power spectrum: the Limber integral of the non-linear matter power spectrum computed by CAMB (Lewis *et al.* 2000) with HMcode (Mead *et al.* 2021) for the Planck 2018 cosmology, for a Euclid-like distribution of source galaxies with median redshift 0.9.  We compute the shear from the convergence and recover the convergence from the shear.
    """)
    return


@app.cell
def _(L):
    _kappa, _ = example_field("convergence", L, taper=False)
    _gamma = physics.kaiser_squires(_kappa)  # C_{0->2}
    _kappa_back = physics.kaiser_squires_inverse(_gamma)  # C_{2->0}
    _g = gaussian(L, 8 / L)[:, None]  # smoothing, which commutes with both, so that the sticks resolve the pattern
    _m = "mollweide"
    _fig, _axes = panels("09_lensing", 13, 8.6, [[_m, _m], [_m, None]])
    mollweide(_axes[0], _kappa, title="(a) simulated convergence $\\kappa$", label="dimensionless")
    mollweide(_axes[1], _gamma, 2, part="abs", title="(b) shear $\\gamma=\\mathcal{C}_{0\\to2}[\\mathcal{D}]\\kappa$", label="dimensionless")
    mollweide(_axes[2], _kappa_back, vmax=float(np.max(np.abs(field_grid(_kappa)[0]))),
              title="(c) convergence recovered from $\\gamma$ by $\\mathcal{C}_{2\\to0}$", label="dimensionless")
    orthographic(_axes[3], _kappa * _g, 0, glyph_flm=_gamma * _g, glyph_s=2, n_glyph=1200,
                 title="(d) smoothed $\\kappa$ and its shear", label="dimensionless")
    mo.vstack([mo.as_html(_fig), caption("09_lensing", r"""
    **Figure 9.** The convergence and shear of weak gravitational lensing, related by the lifted convolutions $\mathcal{C}_{0\to2}$ and $\mathcal{C}_{2\to0}$.  Top: (a) the simulated convergence $\kappa$, shown as colour on a diverging scale, red for overdense and blue for underdense regions; (b) the modulus of the shear $\gamma=\mathcal{C}_{0\to2}[\mathcal{D}]\kappa$, shown as colour on a sequential scale.  Bottom: (c) the convergence recovered from the shear by $\mathcal{C}_{2\to0}[\mathcal{D}^{-1}]$, shown as colour on the diverging scale of (a); and (d) an orthographic view, centred on 20°N, 34°E in the coordinates of the simulation, of the convergence smoothed by a Gaussian of width $8/L$ radians, shown as colour on a diverging scale, with the shear smoothed in the same way drawn as headless sticks along the direction in which the images are stretched, at the angle $\arg(\gamma)/2$, whose lengths grow in proportion to $|\gamma|$ from 35 per cent of the largest length.  The smoothing is a multiplier and commutes with $\mathcal{C}_{0\to2}[\mathcal{D}]$, so that the smoothed shear is the shear of the smoothed convergence.  The sticks circle the overdense regions and point away from the underdense regions, and the convergence recovered from the shear reproduces (a) to a relative error of %.0e, since the simulated convergence has no monopole or dipole.
    """ % l2_rel(_kappa_back, _kappa))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    In each example we have applied lifted convolutions $\mathcal{C}_{s\to s'}$ whose input and output spins are those of the physical quantities concerned.  Derivatives raise or lower the spin, as do their compositions, such as the formula of Vening Meinesz and the Kaiser–Squires relation; smoothing and the continuation of potential fields preserve it; the decomposition of polarisation into $E$ and $B$ modes lowers it; and the steerable ridge detector maps a scalar field to the tuple of spin fields that makes up a function on the rotation group.  Each operator is a single multiplier of the harmonic coefficients, computed exactly and efficiently for band-limited fields, invertible where the physics allows, as the round trips of the geoid, the polarisation and the convergence confirm to the precision of the arithmetic, and rotation-equivariant, so that its result does not depend on the choice of coordinates.  The comparison with the smoothing of the components of the wind as scalar fields shows the error that the spin formalism removes: a flow destroyed around the poles of the coordinates.
    """)
    return


if __name__ == "__main__":
    app.run()
