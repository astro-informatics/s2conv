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
    import numpy as np
    import s2conv
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    from s2conv import kernels, transforms
    from s2conv.convolutions import classical, physics
    from s2conv.utils.evaluation import evaluate
    from s2conv.utils.indexing import conjugate, lm_mask, m_axis
    from s2conv.utils.rotation import rotate_flm

    R_EARTH = 6.3712e6  # mean radius of the Earth (m)
    GAMMA = 9.798  # mean normal gravity on the Earth's surface (m s^-2)
    R_CORE = 3.485e6  # radius of the core-mantle boundary (m)
    ARCSEC = 180 * 3600 / np.pi  # arcseconds per radian


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Lifted convolutions of real spin fields

    This notebook applies the **generalised lifted spherical convolution** of the article *Unifying spherical convolutions through lifting*, as implemented in [`s2conv`](https://github.com/astro-informatics/s2conv), to fields measured on the Earth and on the sky.  The companion notebook `lifted_convolutions.py` verifies the operators; this one shows what they are for.

    **Spin fields.**  Many of the fields of geophysics and cosmology are not scalars.  A wind, an ocean current, a deflection of the vertical or a horizontal magnetic field is a tangent vector; the polarisation of the cosmic microwave background (CMB), the shear of weak gravitational lensing and the horizontal gravity-gradient tensor are symmetric trace-free tensors, i.e. headless directions.  Their components are defined only relative to a local frame $(e_\theta,e_\varphi)$, which turns from point to point and is undefined at the poles.  Written in that frame, a field whose components change by $e^{\mathrm{i}s\chi}$ when the frame turns by $\chi$ is a spin-$s$ field: ${}_1f=v_\theta+\mathrm{i}v_\varphi$ for a vector, ${}_2f=Q+\mathrm{i}U$ for a headless direction.  Treating the components as independent scalar fields, i.e. smoothing or filtering $v_\theta$ and $v_\varphi$ (or $Q$ and $U$) separately, mixes values expressed in different frames: the result depends on where the poles of the coordinates are, it is not rotation-equivariant, and it is wrong near the poles.

    **Lifted convolutions.**  A spin-$s$ field is lifted to the rotation group, convolved there with a kernel $\Psi$ and projected back to spin $s'$, $\mathcal{C}_{s\to s'}[\Psi]=\Lambda_{s'}^\dagger(\Psi\star\Lambda_s\,\cdot\,)$.  In harmonic space it multiplies each coefficient ${}_sf_{\ell m}$ by a single Wigner coefficient of the kernel, so that every bounded rotation-equivariant linear map between spin fields is of this form.  Each example below uses the lifted convolution with the input and output spins that the physics dictates:

    | Field | Spin | Lifted convolutions |
    |:--|:-:|:--|
    | Earth's topography | 0 | slope $\mathcal{C}_{0\to1}$, ridges and their orientation $\mathcal{C}_{0\to s'}$, $\lvert s'\rvert<5$ |
    | Earth's gravity field (geoid) | 0 | gravity anomaly $\mathcal{C}_{0\to0}$, deflection of the vertical $\mathcal{C}_{0\to1}$, gravity gradients $\mathcal{C}_{0\to2}$, and their inverses |
    | Earth's magnetic field | 0, 1 | radial and horizontal field $\mathcal{C}_{0\to0}$, $\mathcal{C}_{0\to1}$; continuation to the core $\mathcal{C}_{0\to0}$, $\mathcal{C}_{1\to1}$ |
    | Near-surface wind | 1 | vorticity and divergence $\mathcal{C}_{1\to0}$, Helmholtz decomposition $\mathcal{C}_{0\to1}$, smoothing $\mathcal{C}_{1\to1}$ |
    | Ocean surface currents | 1 | stream function and upwelling $\mathcal{C}_{1\to0}$ |
    | CMB polarisation | 2 | $E$/$B$ decomposition $\mathcal{C}_{\pm2\to0}$, pure-mode polarisation $\mathcal{C}_{0\to2}$, smoothing $\mathcal{C}_{2\to2}$ |
    | Weak-lensing shear | 2 | mass mapping (Kaiser–Squires) $\mathcal{C}_{2\to0}$ |

    Two examples (the wind and the CMB polarisation) compare the spin-$s$ smoothing $\mathcal{C}_{s\to s}$ with the smoothing of the components as scalars, to show what the spin formalism avoids.  The fields are stored as harmonic coefficients in `notebooks/data` (`notebooks/data/README.md` gives their sources and licences and how they are made) and truncated to the band-limit $L$ below; the Earth fields are tapered by $e^{-\ell(\ell+1)/L^2}$, which falls to $0.37$ at $\ell=L$ and keeps the ringing of the truncation at coastlines and jets small.  Coastlines are those of Natural Earth.
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
def mollweide_grid(ax, data, thetas, phis, cmap="RdBu_r", vmax=None, symmetric=True, title=None, coast=False, contours=None):
    """Draw MW samples data[t, p] on a Mollweide axis, with longitude 0 at the centre, optionally with the coastlines
    and contour lines at the given levels."""
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
        ax.plot(np.where(jump, np.nan, c_lon), c_lat, color="0.15", lw=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9)
    return image


@app.function
def mollweide(ax, flm, s=0, part="real", cmap=None, L_plot=128, vmax=None, title=None, coast=False, contours=None):
    """Mollweide map of the real part, imaginary part or modulus of a band-limited spin field."""
    f, (th, ph) = field_grid(flm, s, L_plot)
    data = {"real": f.real, "imag": f.imag, "abs": np.abs(f)}[part]
    symmetric = part != "abs"
    return mollweide_grid(ax, data, th, ph, cmap or ("RdBu_r" if symmetric else "YlGnBu"), vmax, symmetric, title, coast, contours)


@app.function
def mollweide_glyphs(ax, flm, s, n_glyph=700, size=0.022, color="k", lat_max=np.radians(82), floor=0.35):
    """Glyphs of a spin-s field on a Mollweide axis: arrows for spin 1 and headless sticks for spin 2, at the angle
    arg(f)/s from e_theta towards e_phi and with lengths growing with |f| from floor times the largest, at Fibonacci
    points.  The direction on
    the map is that of a short step along the glyph, mapped by the projection, so that the angle of each glyph to the
    meridian is that on the sphere."""
    k = np.arange(n_glyph) + 0.5
    theta = np.arccos(1 - 2 * k / n_glyph)
    phi = np.mod(np.pi * (1 + 5**0.5) * k, 2 * np.pi)
    keep = np.abs(np.pi / 2 - theta) < lat_max
    theta, phi = theta[keep], phi[keep]
    z = np.asarray(evaluate(flm, s, theta, phi))
    angle = np.angle(z) / s
    lon, lat = np.mod(phi + np.pi, 2 * np.pi) - np.pi, np.pi / 2 - theta
    step = 1e-3
    start = ax.transData.transform(np.stack([lon, lat], -1))
    end = ax.transData.transform(np.stack([lon + step * np.sin(angle) / np.cos(lat), lat - step * np.cos(angle)], -1))
    screen = np.arctan2(*(end - start)[:, ::-1].T)
    length = size * (floor + (1 - floor) * np.abs(z) / np.max(np.abs(z)))
    head = dict(headwidth=3.5, headlength=4, headaxislength=3.5) if abs(s) == 1 else dict(headwidth=0, headlength=0, headaxislength=0)
    ax.quiver(lon, lat, length * np.cos(screen), length * np.sin(screen), angles="uv", scale_units="width", scale=1,
              pivot="middle", color=color, width=0.0022, **head)


@app.function
def orthographic(ax, flm, s=0, part="real", cmap=None, lat0=0.35, lon0=0.6, glyph_flm=None, glyph_s=None,
                 n_glyph=420, glyph_len=0.07, glyph_color="k", vmax=None, title=None, L_plot=128, npix=260, coast=False,
                 symmetric=None, glyph_floor=0.35):
    """One hemisphere in orthographic projection, centred on (lat0, lon0) in radians.  Colour shows the field; for spin
    fields, glyphs with |s|-fold symmetry show the value in the local frame (e_theta, e_phi): an arrow for spin 1 and a
    headless stick for spin 2, at the angle arg(f)/s from e_theta towards e_phi.  The glyphs may be taken from another
    field (glyph_flm, glyph_s).  With coast, the coastlines are drawn."""
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
        ax.plot(xyz[:, 0], xyz[:, 1], color="0.15", lw=0.6)
    gflm, gs = (flm, s) if glyph_flm is None else (glyph_flm, glyph_s)
    if gs:
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
    if title:
        ax.set_title(title, fontsize=9)
    return image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The Earth's topography: slopes and ridges (spin 0 → 1 → 2)

    The topography and bathymetry $f$ (ETOPO 2022, NOAA) is a scalar field, but the two quantities that describe its shape are not.

    **Slope.**  The gradient is a vector, and the lifted convolution $\mathcal{C}_{0\to1}$ with the multiplier $\sqrt{\ell(\ell+1)}$ of the spin raising operator gives it directly as a spin-1 field: since $\eth f=-(\partial_\theta+\mathrm{i}\sin^{-1}\theta\,\partial_\varphi)f$, the field $\eth f/R_\oplus$ is the vector $-\nabla f$, pointing downhill, with the slope as its modulus.  The arrows show where water would flow.

    **Ridges.**  A mountain range has an orientation, a headless direction.  The steerable ridge detector $\psi$ (an elongated second-derivative profile with azimuthal orders $|m|<5$) is applied at every orientation $\gamma$ at once by the directional convolution $f\circledast\psi$, a function on the rotation group.  Since the kernel is steerable, this function is the tuple of the nine spin fields ${}_{s'}g=\mathcal{C}_{0\to s'}[\Lambda_0\psi]f$, $|s'|<5$ (Proposition VI.6), from which the response at any $\gamma$ follows as $\sum_{s'}(2\pi)^{-1/2}e^{-\mathrm{i}s'\gamma}\,{}_{s'}g$.  Its largest value over $\gamma$ is the ridge strength, and the spin-2 component ${}_2g$ is the orientation of the ridges as a spin-2 field: it is largest along a ridge at the angle $\arg({}_2g)/2$, drawn as sticks.  No sampling of orientations is needed, and both outputs are exactly rotation-equivariant, so that ranges running north–south and east–west, near the equator or near the poles, are detected alike.  The detector is applied to the land elevation $f_+=\max(f,0)$, since the continental margins, a step of several kilometres, would otherwise dominate its response.

    The figure shows the topography, the slope of the land with the downhill direction, and the ridge strength, and zooms onto the Himalaya and the Andes with the downhill arrows and the ridge orientation.
    """)
    return


@app.cell
def _(L):
    _topo, _ = example_field("earth_topography", L)
    _downhill = physics.eth(_topo, 0) / (R_EARTH / 1e3) * 1e3  # C_{0->1}: -grad f, as a slope in m per km
    _grid, _ = field_grid(_topo, 0, 2 * L)  # clip at sea level on a finer grid, to keep the aliasing small
    _land = np.asarray(transforms.forward(np.maximum(_grid.real, 0), 2 * L))[:L, L : 3 * L - 1]
    _land_downhill = physics.eth(_land, 0) / (R_EARTH / 1e3) * 1e3
    _N = 5
    _ridge = oriented_kernel(L, sig_long=2.5 * 2.6 / L, sig_short=2.6 / L, N=_N, ridge=True)
    _comps = {_q: s2conv.lifted_convolution(_land, kernels.lifted(_ridge, 0), 0, _q) for _q in range(-(_N - 1), _N)}
    _grids = {_q: field_grid(_c, _q)[0] for _q, _c in _comps.items()}
    _strength = np.max([sum(np.exp(-1j * _q * _g) * _x for _q, _x in _grids.items()).real
                        for _g in np.linspace(0, np.pi, 36, endpoint=False)], axis=0) / np.sqrt(2 * np.pi)
    _th, _ph = transforms.sphere_angles(128)

    _fig = Figure(figsize=(13, 7.4), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), _topo, cmap="BrBG_r", coast=True, title="topography and bathymetry $f$ (km)")
    _ax = _fig.add_subplot(2, 3, 2, projection="mollweide")
    mollweide(_ax, _land_downhill, 1, part="abs", cmap="YlOrBr", coast=True, title="slope of the land $|\\eth f_+|/R_\\oplus$ (m/km), downhill arrows")
    mollweide_glyphs(_ax, _land_downhill, 1, n_glyph=900, size=0.022, floor=0.0)
    mollweide_grid(_fig.add_subplot(2, 3, 3, projection="mollweide"), np.clip(_strength, 0, None), _th, _ph, cmap="YlOrBr", symmetric=False, coast=True,
                   title="ridge strength $\\max_\\gamma(f_+\\circledast\\psi)(\\cdot,\\gamma)$")
    _view = dict(lat0=np.radians(32), lon0=np.radians(85))
    orthographic(_fig.add_subplot(2, 3, 4), _land, 0, cmap="YlOrBr", symmetric=False, **_view, glyph_flm=_land_downhill, glyph_s=1, coast=True, glyph_floor=0.0,
                 title="Himalaya and Tibet: land elevation, downhill $\\eth f_+$")
    orthographic(_fig.add_subplot(2, 3, 5), _comps[2], 2, part="abs", cmap="YlOrBr", **_view, coast=True, glyph_floor=0.1,
                 title="ridge orientation ${}_2g=\\mathcal{C}_{0\\to2}[\\Lambda_0\\psi]f_+$")
    orthographic(_fig.add_subplot(2, 3, 6), _comps[2], 2, part="abs", cmap="YlOrBr", lat0=np.radians(-20), lon0=np.radians(-68), coast=True, glyph_floor=0.1,
                 title="Andes: ridge orientation ${}_2g$")
    _fig.suptitle("Topography: a scalar field whose slope is spin 1 and whose ridge orientation is spin 2", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The Earth's gravity field: one potential, three spins (spin 0 → 0, 1, 2 and back)

    The geoid height $N$ of the Earth Gravitational Model EGM2008 (Pavlis *et al.* 2012), the height of the equipotential surface of gravity above the reference ellipsoid, is a scalar field.  By Bruns' formula it is the disturbing potential $T=\gamma N$ divided by normal gravity, and the quantities that geodesy measures are derivatives of $T$ of increasing spin:

    * the **gravity anomaly** $\Delta g_{\ell m}=\frac{\gamma}{R}(\ell-1)N_{\ell m}$, measured by gravimeters, is a scalar, $\mathcal{C}_{0\to0}$;
    * the **deflection of the vertical**, the angle between the plumb line and the normal to the ellipsoid, is the vector $-\nabla N/R$, the spin-1 field $\varepsilon=\eth N/R=\mathcal{C}_{0\to1}[\sqrt{\ell(\ell+1)}/R]\,N$, measured by astronomical positioning;
    * the horizontal, trace-free part of the **gravity-gradient tensor**, measured by the gradiometer of the GOCE satellite, is the spin-2 field $\Gamma=\tfrac12\eth^2T/R^2=\mathcal{C}_{0\to2}$.

    The classical integral formulas of physical geodesy are lifted convolutions between these spins.  The Vening Meinesz formula, which gives the deflection of the vertical from gravity anomalies, is the spin-changing convolution $\mathcal{C}_{0\to1}$ with the multiplier $\frac{\sqrt{\ell(\ell+1)}}{\gamma(\ell-1)}$; its inverse, and the recovery of the geoid from deflections (astrogeodetic levelling, $\mathcal{C}_{1\to0}$) or from gradients ($\mathcal{C}_{2\to0}$), are single multipliers too.  The recovery of the potential from its spin-2 gradients is the same operator as the Kaiser–Squires mass mapping of weak lensing in Sec. 7, since the shear is likewise $\tfrac12\eth^2$ of the lensing potential.  The titles report the relative errors of these conversions, which hold to machine precision because each is a single multiplier.
    """)
    return


@app.cell
def _(L):
    _N, _ = example_field("geoid", L)
    _el = np.arange(L)
    _dg = apply(_N, multiplier(L, lambda el: GAMMA / R_EARTH * (el - 1) * 1e5), 0, 0)  # C_{0->0}: gravity anomaly (mGal)
    _eps = physics.eth(_N, 0) / R_EARTH * ARCSEC  # C_{0->1}: deflection of the vertical (arcsec)
    _vm = multiplier(L, lambda el: np.where(el > 1, np.sqrt(el * (el + 1)) / np.maximum(el - 1, 1) / (GAMMA * 1e5) * ARCSEC, 0))
    _eps_vm = apply(_dg, _vm, 0, 1)  # Vening Meinesz: C_{0->1} from gravity anomalies
    _grad = physics.eth(physics.eth(_N, 0), 1) * GAMMA / (2 * R_EARTH**2) * 1e9  # C_{0->2}: gradients (Eotvos)
    _N_from_eps = apply(_eps / ARCSEC * R_EARTH, inverse_laplacian(L) * physics.ethbar_multiplier(L, 1), 1, 0)  # C_{1->0}
    _ks = multiplier(L, lambda el: np.where(el > 1, 1 / np.sqrt(np.maximum((el - 1) * el * (el + 1) * (el + 2), 1)), 0))
    _N_from_grad = apply(_grad * 2 * R_EARTH**2 / (GAMMA * 1e9), _ks, 2, 0)  # C_{2->0}, as Kaiser-Squires
    _errs = (l2_rel(_eps_vm, _eps), l2_rel(_N_from_eps, _N), l2_rel(_N_from_grad, _N))

    _fig = Figure(figsize=(13, 7.4), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), _N, cmap="RdBu_r", coast=True, title="geoid height $N$ (m), EGM2008")
    mollweide(_fig.add_subplot(2, 3, 2, projection="mollweide"), _dg, cmap="RdBu_r", coast=True, vmax=robust_vmax(_dg), title="gravity anomaly $\\Delta g=\\mathcal{C}_{0\\to0}N$ (mGal)")
    _ax = _fig.add_subplot(2, 3, 3, projection="mollweide")
    mollweide(_ax, _eps, 1, part="abs", cmap="YlOrBr", coast=True, title="deflection of the vertical $\\varepsilon=\\mathcal{C}_{0\\to1}N$ (arcsec)")
    mollweide_glyphs(_ax, _eps, 1, n_glyph=800, size=0.018)
    orthographic(_fig.add_subplot(2, 3, 4), _N, 0, lat0=np.radians(-5), lon0=np.radians(80), glyph_flm=_eps, glyph_s=1, coast=True,
                 title=f"Indian Ocean geoid low with $\\varepsilon$\nVening Meinesz $\\mathcal{{C}}_{{0\\to1}}\\Delta g$ vs $\\eth N/R$: {_errs[0]:.0e}")
    orthographic(_fig.add_subplot(2, 3, 5), _grad, 2, part="abs", cmap="YlOrBr", lat0=np.radians(-20), lon0=np.radians(-75), coast=True,
                 title=f"gravity gradients $\\Gamma=\\mathcal{{C}}_{{0\\to2}}N$ (E), Andes\n$N$ from $\\Gamma$ by $\\mathcal{{C}}_{{2\\to0}}$: {_errs[2]:.0e}")
    orthographic(_fig.add_subplot(2, 3, 6), _N_from_eps, 0, lat0=np.radians(-5), lon0=np.radians(140), coast=True,
                 title=f"geoid from deflections by $\\mathcal{{C}}_{{1\\to0}}$: error {_errs[1]:.0e}\nIndonesia and New Guinea")
    _fig.suptitle("Gravity: the geoid and its derivatives of spin 0, 1 and 2, related by lifted convolutions", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. The Earth's magnetic field: a potential field of spin 0 and 1, continued to the core

    Outside its sources the main geomagnetic field is the gradient of a potential, $B=-\nabla V$, and the International Geomagnetic Reference Field (IGRF-14, epoch 2025; Alken *et al.* 2021 for the model) gives $V$ as spherical harmonics up to degree 13.  With $v=V/R$ on the Earth's surface, the radial field is the scalar $B_r=\mathcal{C}_{0\to0}[\ell+1]\,v$ and the horizontal field is the spin-1 field $B_\theta+\mathrm{i}B_\varphi=\eth v=\mathcal{C}_{0\to1}[\sqrt{\ell(\ell+1)}]\,v$.  A compass needle points along the horizontal field, and the declination (the angle of the needle east of geographic north) and the inclination (the dip below the horizontal) are its argument and the ratio of its modulus to $B_r$; the contour marks the agonic line of zero declination.

    **Continuation to the core.**  Each degree of a potential field decays with radius as $r^{-(\ell+2)}$, so that the field at the core–mantle boundary, $r_c=3485$ km, follows from that at the surface by the multiplier $(R/r_c)^{\ell+2}$, the downward continuation.  It is spin-preserving, and applies to the radial field as $\mathcal{C}_{0\to0}$ and to the horizontal *vector* field as $\mathcal{C}_{1\to1}$, directly and without any reference to its components.  Downward continuation amplifies the high degrees by up to $(R/r_c)^{15}\approx8500$, and reveals the flux patches of the core field beneath the polar regions and the reversed flux beneath the South Atlantic, the origin of the South Atlantic Anomaly.
    """)
    return


@app.cell
def _(L):
    _v, _ = example_field("geomagnetic_potential", L, taper=False)
    _b_r = apply(_v, multiplier(L, lambda el: el + 1), 0, 0)  # C_{0->0}
    _b_h = physics.eth(_v, 0)  # C_{0->1}: B_theta + i B_phi
    _core = multiplier(L, lambda el: (R_EARTH / R_CORE) ** (el + 2))
    _b_r_core, _b_h_core = apply(_b_r, _core, 0, 0) / 1e3, apply(_b_h, _core, 1, 1) / 1e3  # C_{0->0}, C_{1->1} (muT)
    _f_h, (_th, _ph) = field_grid(_b_h, 1)
    _f_r, _ = field_grid(_b_r, 0)
    _decl = np.degrees(np.arctan2(_f_h.imag, -_f_h.real))  # east of north
    _incl = np.degrees(np.arctan2(-_f_r.real, np.abs(_f_h)))

    _fig = Figure(figsize=(13, 7.4), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), _b_r / 1e3, coast=True, title="radial field $B_r=\\mathcal{C}_{0\\to0}v$ at the surface ($\\mu$T)")
    _ax = _fig.add_subplot(2, 3, 2, projection="mollweide")
    mollweide(_ax, _b_h / 1e3, 1, part="abs", coast=True, title="horizontal field $\\eth v=\\mathcal{C}_{0\\to1}v$ ($\\mu$T): compass needles")
    mollweide_glyphs(_ax, _b_h, 1, n_glyph=600)
    mollweide_grid(_fig.add_subplot(2, 3, 3, projection="mollweide"), _decl, _th, _ph, vmax=30, coast=True, contours=[-20, -10, 0, 10, 20],
                   title="declination (degrees east of north), every $10^\\circ$")
    mollweide_grid(_fig.add_subplot(2, 3, 4, projection="mollweide"), _incl, _th, _ph, vmax=90, coast=True, contours=[0],
                   title="inclination (degrees), magnetic equator")
    mollweide(_fig.add_subplot(2, 3, 5, projection="mollweide"), _b_r_core, coast=True, title="$B_r$ continued to the core–mantle boundary (mT)")
    orthographic(_fig.add_subplot(2, 3, 6), _b_r_core, 0, lat0=np.pi / 2 - 1e-3, lon0=0.0, glyph_flm=_b_h_core, glyph_s=1, coast=True,
                 title="core–mantle boundary under the Arctic:\n$B_r$ and the horizontal field $\\mathcal{C}_{1\\to1}\\eth v$")
    _fig.suptitle("The geomagnetic field (IGRF-14, 2025) as spin-0 and spin-1 fields, and its downward continuation", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Near-surface wind: vorticity, divergence and why spin matters (spin 1)

    The January near-surface wind, averaged over 1991–2020 (NCEP/NCAR reanalysis, Kalnay *et al.* 1996), with northward and eastward components $(v,u)$, is the spin-1 field ${}_1f=-v+\mathrm{i}u$.

    **Vorticity and divergence.**  For a spin-1 field, $\bar\eth\,{}_1f=-R_\oplus(\nabla\cdot v+\mathrm{i}\,\hat r\cdot\nabla\times v)$: the single lifted convolution $\mathcal{C}_{1\to0}$ gives the divergence and the vorticity together, as the real and imaginary parts of a scalar field, with no finite differences and no special treatment of the poles.  The vorticity shows the cyclones of the Aleutian and Icelandic lows (positive in the north) and the anticyclones of the subtropics; the divergence shows the convergence of the trade winds into the intertropical convergence zone.  Since $\eth\bar\eth$ is the Laplacian on spin-1 fields, $\mathcal{C}_{0\to1}$ applied to the inverse Laplacian of each part gives the Helmholtz decomposition into a divergence-free part (the rotational flow of the westerlies and the trades) and a curl-free part (the flow into the convergence zone), which sum to the wind exactly.

    **Why spin matters.**  Smoothing the wind by a Gaussian is the spin-preserving lifted convolution $\mathcal{C}_{1\to1}$, a single multiplier.  A common shortcut smooths the components $v_\theta$ and $v_\varphi$ as if they were two scalar maps.  Since the frame $(e_\theta,e_\varphi)$ turns around the pole, this averages vectors expressed in different frames: near the pole the components change rapidly even where the wind is uniform, and component-wise smoothing distorts the flow.  The poles of the coordinates are a convention, and nothing prevents placing one where the wind is strongest, centred in the lower row.  The spin-1 smoothing gives the same wind in either system of coordinates, to machine precision, since it is rotation-equivariant.  The component-wise smoothing gives a reasonable result in geographic coordinates, where the jet is far from the poles, but with the coordinate pole in the strong flow it mixes the components across the turning frame and destroys the flow at the centre: its result depends on the choice of coordinates.
    """)
    return


@app.cell
def _(L):
    _wind, _ = example_field("wind", L)
    _h = helmholtz(_wind, R_EARTH)
    _view = dict(lat0=np.radians(20), lon0=np.radians(-150))
    _fig = Figure(figsize=(13, 7.4), layout="constrained")
    _ax = _fig.add_subplot(2, 3, 1, projection="mollweide")
    mollweide(_ax, _wind, 1, part="abs", coast=True, title="January near-surface wind ${}_1f$ (m/s)")
    mollweide_glyphs(_ax, _wind, 1)
    mollweide(_fig.add_subplot(2, 3, 2, projection="mollweide"), _h["curl"] * 1e5, coast=True, vmax=robust_vmax(_h["curl"] * 1e5), title="vorticity $-\\mathrm{Im}\\,\\bar\\eth\\,{}_1f/R_\\oplus$ ($10^{-5}$ s$^{-1}$)")
    mollweide(_fig.add_subplot(2, 3, 3, projection="mollweide"), _h["div"] * 1e5, coast=True, vmax=robust_vmax(_h["div"] * 1e5), title="divergence $-\\mathrm{Re}\\,\\bar\\eth\\,{}_1f/R_\\oplus$ ($10^{-5}$ s$^{-1}$)")
    orthographic(_fig.add_subplot(2, 3, 4), _wind, 1, part="abs", **_view, coast=True, title="the wind over the Pacific")
    orthographic(_fig.add_subplot(2, 3, 5), _h["f_rot"], 1, part="abs", **_view, coast=True, title="divergence-free part $\\mathcal{C}_{0\\to1}\\nabla^{-2}\\mathrm{curl}$")
    orthographic(_fig.add_subplot(2, 3, 6), _h["f_div"], 1, part="abs", **_view, coast=True,
                 title=f"curl-free part; the two sum to ${{}}_1f$ to {l2_rel(_h['f_rot'] + _h['f_div'], _wind):.0e}")
    _fig.suptitle("Vorticity, divergence and the Helmholtz decomposition of the wind by lifted convolutions $\\mathcal{C}_{1\\to0}$ and $\\mathcal{C}_{0\\to1}$", fontsize=10)

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
    _vmax = float(np.max(np.abs(field_grid(_spin, 1)[0])))
    _fig2 = Figure(figsize=(13, 3.9), layout="constrained")
    orthographic(_fig2.add_subplot(1, 4, 1), _spin, 1, part="abs", vmax=_vmax, **_jet,
                 title=f"smoothed as a spin-1 field $\\mathcal{{C}}_{{1\\to1}}$; with the\ncoordinate pole at the centre: difference {l2_rel(_spin_moved, _spin):.0e}")
    orthographic(_fig2.add_subplot(1, 4, 2), _comp, 1, part="abs", vmax=_vmax, **_jet, title="components smoothed as scalars,\ngeographic coordinates")
    orthographic(_fig2.add_subplot(1, 4, 3), _comp_moved, 1, part="abs", vmax=_vmax, **_jet,
                 title="components smoothed as scalars,\ncoordinate pole at the centre")
    orthographic(_fig2.add_subplot(1, 4, 4), _comp_moved - _spin, 1, part="abs", cmap="Greys", vmax=_vmax, glyph_floor=0.0, **_jet,
                 title=f"its error, on the same scale:\nrelative $L^2$ error {l2_rel(_comp_moved, _spin):.0e} over the sphere")
    _fig2.suptitle("Smoothing a vector field: the spin-1 convolution against smoothing the components as scalars", fontsize=10)
    mo.vstack([mo.as_html(_fig), mo.as_html(_fig2)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Ocean surface currents: gyres and upwelling (spin 1)

    The surface currents of 2025 (GODAS, Behringer & Xue 2004) are a spin-1 field like the wind, set to zero on land.  The same lifted convolution $\mathcal{C}_{1\to0}$ gives their vorticity and divergence, and the stream function $\psi=R_\oplus^2\nabla^{-2}(\hat r\cdot\nabla\times v)$ of their rotational part, whose contours are the streamlines of the divergence-free flow: they trace the great subtropical gyres, clockwise in the northern hemisphere and anticlockwise in the southern, and the Antarctic Circumpolar Current.  (The stream function is defined over the whole sphere, and has no meaning over land, where the currents vanish.)  The divergence of the surface flow marks upwelling, where the Ekman transport driven by the trade winds carries surface water away from the equator and cold, nutrient-rich water rises in its place: the equatorial Pacific and Atlantic cold tongues and the coastal upwelling off Peru and West Africa.  The Earth's rotation reverses the direction of the Ekman transport across the equator, which is what makes the equatorial surface flow diverge.
    """)
    return


@app.cell
def _(L):
    _cur, _ = example_field("ocean_currents", L)
    _h = helmholtz(_cur, R_EARTH)
    _psi = _h["psi"] / 1e6  # 10^6 m^2/s
    _psi_grid, _ = field_grid(_psi)
    _levels = np.linspace(-1, 1, 13) * np.max(np.abs(_psi_grid.real))
    _fig = Figure(figsize=(13, 7.4), layout="constrained")
    _ax = _fig.add_subplot(2, 2, 1, projection="mollweide")
    mollweide(_ax, _cur, 1, part="abs", coast=True, title="surface current speed (m/s) and direction, 2025")
    mollweide_glyphs(_ax, _cur, 1, n_glyph=1200, size=0.02, floor=0.0)
    mollweide(_fig.add_subplot(2, 2, 2, projection="mollweide"), _psi, coast=True, contours=_levels,
              title="stream function $\\psi$ of the rotational flow ($10^6$ m$^2$/s): the gyres")
    mollweide(_fig.add_subplot(2, 2, 3, projection="mollweide"), _h["div"] * 1e7, coast=True, vmax=robust_vmax(_h["div"] * 1e7),
              title="divergence $-\\mathrm{Re}\\,\\bar\\eth\\,{}_1f/R_\\oplus$ ($10^{-7}$ s$^{-1}$): upwelling")
    orthographic(_fig.add_subplot(2, 2, 4), _h["div"] * 1e7, 0, lat0=np.radians(-5), lon0=np.radians(-110), glyph_flm=_cur, glyph_s=1, coast=True, glyph_floor=0.0, vmax=robust_vmax(_h["div"] * 1e7),
                 title="the eastern equatorial Pacific: divergence and currents")
    _fig.suptitle("Ocean surface currents: stream function and upwelling from the lifted convolution $\\mathcal{C}_{1\\to0}$", fontsize=10)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Polarisation of the cosmic microwave background (spin 2)

    The linear polarisation of the CMB, $Q+\mathrm{i}U$, is a spin-2 field.  Its physically meaningful parts are two scalar fields, the $E$ and $B$ modes: by Proposition VI.8(b), $a_E=-\tfrac12(a_2+a_{-2})$ and $a_B=\tfrac{\mathrm{i}}2(a_2-a_{-2})$ are sums of the lifted convolutions $\mathcal{C}_{2\to0}$ and $\mathcal{C}_{-2\to0}$, and $\mathcal{C}_{0\to2}$ builds the polarisation of each mode alone.  Scalar (density) perturbations in the early Universe make only $E$ modes, whose polarisation is radial or tangential around the extrema of $E$; gravitational lensing of the $E$ modes by the large-scale structure makes a small $B$ mode, whose polarisation is at $45^\circ$ to these directions; primordial gravitational waves would add a further $B$ mode, the target of current experiments.

    The simulated maps are a Gaussian realisation of the lensed power spectra of the Planck 2018 best-fit cosmology (Planck Collaboration 2020), with the temperature correlated with $E$.  The measured maps are the Planck 2018 SMICA maps in Galactic coordinates, with the plane of the Galaxy along the equator of the map; their large-scale polarisation mixes signal with noise and residual Galactic emission, and an analysis would mask the plane.  The temperature is drawn with sticks of the polarisation smoothed by the spin-preserving Gaussian $\mathcal{C}_{2\to2}$, as in the maps of WMAP and Planck.

    **Why spin matters.**  Since the $B$ mode is orders of magnitude fainter than the $E$ mode, any operation that leaks $E$ into $B$ can swamp it.  The last figure smooths a pure-$E$ polarisation field ($B=0$) in two ways.  The spin-2 smoothing $\mathcal{C}_{2\to2}$ is a multiplier of the spin-2 coefficients and leaves $B=0$ exactly.  Smoothing $Q$ and $U$ as two scalar maps leaks $E$ into a spurious $B$ mode, concentrated where the frame turns quickly, around the poles, and larger than the lensing $B$ mode itself (by a factor of about 20 at $L=32$, 6 at $L=64$ and 2 at $L=128$ for this smoothing of six pixels).  The middle figure shows the $E$ and $B$ modes of the simulation and their polarisation, both smoothed so that the patterns around the extrema are visible.
    """)
    return


@app.cell
def _(L):
    _el = np.arange(L)
    _beam = gaussian(L, 4 / L)
    _cmb = {}
    for _src in ("sim", "planck"):
        _t, _ = example_field(f"cmb_{_src}_temperature", L, taper=False)
        _p, _ = example_field(f"cmb_{_src}_polarisation", L, taper=False)
        _e, _b = physics.eb_decomposition(_p)  # C_{2->0} + C_{-2->0}
        _cmb[_src] = dict(t=_t, p=_p, e=_e, b=_b, p_smooth=apply(_p, _beam, 2, 2))
    _fig = Figure(figsize=(13, 7.4), layout="constrained")
    for _row, (_src, _label) in enumerate((("sim", "simulated"), ("planck", "Planck 2018 SMICA"))):
        _c = _cmb[_src]
        _ax = _fig.add_subplot(2, 3, 3 * _row + 1, projection="mollweide")
        mollweide(_ax, _c["t"], title=f"{_label}: temperature ($\\mu$K) and polarisation")
        mollweide_glyphs(_ax, _c["p_smooth"], 2)
        mollweide(_fig.add_subplot(2, 3, 3 * _row + 2, projection="mollweide"), _c["e"], title=f"{_label}: $E=\\mathcal{{C}}_{{2\\to0}}+\\mathcal{{C}}_{{-2\\to0}}$ ($\\mu$K)")
        mollweide(_fig.add_subplot(2, 3, 3 * _row + 3, projection="mollweide"), _c["b"], title=f"{_label}: $B$ ($\\mu$K)")
    _fig.suptitle("CMB temperature and polarisation, and the $E$/$B$ decomposition by lifted convolutions", fontsize=10)

    _sim = _cmb["sim"]
    _pE, _ = physics.eb_to_polarisation(_sim["e"], 0 * _sim["b"])  # C_{0->2}
    _pB, _ = physics.eb_to_polarisation(0 * _sim["e"], _sim["b"])
    # smoothed, so that the patterns around the extrema are visible at the resolution of the glyphs
    _g = gaussian(L, 10 / L)[:, None]
    _eS, _bS, _pES, _pBS = _sim["e"] * _g, _sim["b"] * _g, _pE * _g, _pB * _g
    _fig2 = Figure(figsize=(13, 3.9), layout="constrained")
    orthographic(_fig2.add_subplot(1, 3, 1), _eS, 0, glyph_flm=_pES, glyph_s=2, n_glyph=1200, title="simulated $E$ and its polarisation $\\mathcal{C}_{0\\to2}E$:\nradial and tangential")
    orthographic(_fig2.add_subplot(1, 3, 2), _bS, 0, glyph_flm=_pBS, glyph_s=2, n_glyph=1200, title="simulated lensing $B$ and its polarisation:\nat $45^\\circ$")
    orthographic(_fig2.add_subplot(1, 3, 3), _cmb["planck"]["t"], 0, lat0=np.pi / 2 - 1e-3, lon0=0.0, glyph_flm=_cmb["planck"]["p_smooth"], glyph_s=2,
                 title="Planck: temperature and polarisation\nat the north Galactic pole")

    _smooth_beam = gaussian(L, 6 / L)
    _b_spin = physics.eb_decomposition(apply(_pE, _smooth_beam, 2, 2))[1]
    _b_comp = physics.eb_decomposition(componentwise(_pE, 2, _smooth_beam))[1]
    _b_lens = _sim["b"] * _smooth_beam[:, None]
    _vmax = float(np.max(np.abs(field_grid(_b_comp)[0].real)))
    _rms = lambda _x: float(np.sqrt(np.sum(np.abs(_x) ** 2) / (4 * np.pi)))
    _fig3 = Figure(figsize=(13, 3.3), layout="constrained")
    mollweide(_fig3.add_subplot(1, 3, 1, projection="mollweide"), _b_lens, vmax=_vmax, title=f"lensing $B$, smoothed: rms {_rms(_b_lens):.2g} $\\mu$K")
    mollweide(_fig3.add_subplot(1, 3, 2, projection="mollweide"), _b_spin, vmax=_vmax, title=f"$B$ of pure $E$ smoothed by $\\mathcal{{C}}_{{2\\to2}}$: rms {_rms(_b_spin):.0e} $\\mu$K")
    mollweide(_fig3.add_subplot(1, 3, 3, projection="mollweide"), _b_comp, vmax=_vmax, title=f"$B$ of pure $E$ with $Q$, $U$ smoothed as scalars:\nrms {_rms(_b_comp):.2g} $\\mu$K, {_rms(_b_comp) / _rms(_b_lens):.0f} times the lensing $B$")
    _fig3.suptitle("Leakage of $E$ into $B$: the spin-2 convolution against smoothing $Q$ and $U$ as scalars (same colour scale)", fontsize=10)
    mo.vstack([mo.as_html(_fig), mo.as_html(_fig2), mo.as_html(_fig3)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Weak gravitational lensing: mass mapping (spin 2 → 0)

    Light from distant galaxies is deflected by the matter along the line of sight, which shears their images.  The convergence $\kappa$, the projected mass density, is a scalar field, and the shear $\gamma$ is the spin-2 field $\gamma=\mathcal{C}_{0\to2}[\mathcal{D}]\kappa$ with $\mathcal{D}_\ell=-\sqrt{(\ell+2)(\ell-1)/(\ell(\ell+1))}$ (Proposition VI.8(c)), whose sticks are tangential around mass concentrations.  The simulated convergence is a shifted lognormal field (Hilbert *et al.* 2011) with a realistic power spectrum: the Limber integral of the non-linear matter power spectrum of CAMB (Lewis *et al.* 2000; HMcode, Mead *et al.* 2021) for the Planck 2018 cosmology and a Euclid-like distribution of source galaxies with median redshift 0.9.

    The shear measured from galaxy shapes carries shape noise, here white with a dispersion of $0.26$ per component and a density of galaxies set by the slider.  The spherical Kaiser–Squires estimator $\mathcal{C}_{2\to0}[\mathcal{D}^{-1}]$ (Kaiser & Squires 1993; Wallis *et al.* 2022) recovers $\kappa_E+\mathrm{i}\kappa_B$ from the noisy shear in one step on the full sphere, without the flat-sky approximation of the original method: $\kappa_E$ estimates the convergence and $\kappa_B$, which lensing does not produce, is a null test that contains only noise.  All maps are smoothed by the same axisymmetric Gaussian convolution.
    """)
    return


@app.cell(hide_code=True)
def _():
    ngal_ui = mo.ui.slider(steps=[0.1, 0.3, 1, 3, 10, 30], value=1, show_value=True, label="galaxies per arcmin$^2$")
    ngal_ui
    return (ngal_ui,)


@app.cell
def _(L, ngal_ui):
    _kappa, _ = example_field("convergence", L, taper=False)
    _gamma = physics.kaiser_squires(_kappa)  # C_{0->2}
    _rng = np.random.default_rng(2026)
    _N = 0.26**2 / (ngal_ui.value / (np.pi / 10800) ** 2)  # noise power of each of the E and B modes
    _noise = np.sqrt(_N) * (_rng.standard_normal((L, 2 * L - 1)) + 1j * _rng.standard_normal((L, 2 * L - 1))) * lm_mask(L, 2)
    _gamma_obs = _gamma + _noise
    _k = physics.kaiser_squires_inverse(_gamma_obs)  # C_{2->0}
    _beam = gaussian_axisymmetric(L, 3 / L)
    _smooth = lambda _x: classical.axisymmetric_convolution(_x, _beam, 0)
    _kE, _kB, _kT = _smooth(real_part(_k)), _smooth(imag_part(_k)), _smooth(_kappa)
    _r = np.real(np.vdot(_kT, _kE)) / np.sqrt(np.real(np.vdot(_kT, _kT) * np.vdot(_kE, _kE)))
    _vmax = float(np.max(np.abs(field_grid(_kT)[0])))

    _fig = Figure(figsize=(13, 7.4), layout="constrained")
    mollweide(_fig.add_subplot(2, 3, 1, projection="mollweide"), _kappa, title="simulated convergence $\\kappa$")
    orthographic(_fig.add_subplot(2, 3, 2), _kappa, 0, glyph_flm=_gamma, glyph_s=2, title="$\\kappa$ with its shear $\\gamma=\\mathcal{C}_{0\\to2}[\\mathcal{D}]\\kappa$")
    orthographic(_fig.add_subplot(2, 3, 3), _kappa, 0, glyph_flm=_gamma_obs, glyph_s=2, title="$\\kappa$ with the noisy shear")
    mollweide(_fig.add_subplot(2, 3, 4, projection="mollweide"), _kT, vmax=_vmax, title="$\\kappa$, smoothed")
    mollweide(_fig.add_subplot(2, 3, 5, projection="mollweide"), _kE, vmax=_vmax, title=f"Kaiser–Squires $\\kappa_E$, smoothed: correlation {_r:.2f} with $\\kappa$")
    mollweide(_fig.add_subplot(2, 3, 6, projection="mollweide"), _kB, vmax=_vmax, title="Kaiser–Squires $\\kappa_B$, smoothed: noise")
    _fig.suptitle("Mass mapping from a noisy shear field by lifted convolutions $\\mathcal{C}_{0\\to2}$ and $\\mathcal{C}_{2\\to0}$", fontsize=10)
    _fig
    return


@app.function
def gaussian_axisymmetric(L, sigma, s=0):
    """Axisymmetric spin-s kernel psi_{l0} = sqrt((2l+1)/(4 pi)) exp(-l(l+1) sigma^2 / 2), for which the axisymmetric
    convolution multiplies by the Gaussian beam exp(-l(l+1) sigma^2 / 2)."""
    el = np.arange(L)
    psi = np.zeros((L, 2 * L - 1), np.complex128)
    psi[:, L - 1] = np.sqrt((2 * el + 1) / (4 * np.pi)) * np.exp(-0.5 * el * (el + 1) * sigma**2)
    return psi * lm_mask(L, s)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    Every example applies lifted convolutions $\mathcal{C}_{s\to s'}$ whose input and output spins are those of the physical quantities: derivatives raise or lower the spin ($\eth$, $\bar\eth$, and their compositions such as the Vening Meinesz and Kaiser–Squires operators), smoothing and continuation preserve it, and the directional ridge detector maps a scalar to the tuple of spins that makes up a function on the rotation group.  Each is a single multiplier of the harmonic coefficients, exact for band-limited fields, invertible where the physics allows, and rotation-equivariant, so that the results do not depend on the choice of coordinates.  The comparisons with component-wise processing in Secs 4 and 6 show the errors the spin formalism removes: a distorted flow around the poles and a leakage of $E$ into $B$ larger than the lensing $B$ mode.
    """)
    return


if __name__ == "__main__":
    app.run()
