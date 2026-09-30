"""
Build the example fields of the notebook's "Real fields" section.

Each field is sampled on a McEwen-Wiaux grid at a high band-limit, transformed with
s2conv, truncated to band-limit L_OUT and saved as harmonic coefficients (complex64)
in an .npz file next to this script.  The data are read over OPeNDAP, so only the
slices used are downloaded.  Run once, from the root of the repository, with

    python notebooks/data/make_example_fields.py

This needs netCDF4 (with OPeNDAP support), scipy and astropy, which are not
dependencies of s2conv.  See README.md in this directory for provenance and licences.
"""

import datetime
from pathlib import Path

import jax
import netCDF4
import numpy as np
from astropy.cosmology import Planck18
from scipy.interpolate import RegularGridInterpolator

jax.config.update("jax_enable_x64", True)

from s2conv import transforms  # noqa: E402  (after enabling 64-bit precision)
from s2conv.convolutions import physics  # noqa: E402

OUT = Path(__file__).parent
L_OUT = 64
SEED = 2026

ETOPO = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)
NCEP = (
    "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface/"
    "{}wnd.sig995.mon.ltm.1991-2020.nc"
)
GODAS = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/{}cur.2025.nc"


def to_harmonic(values, lat, lon, L_hi, spin=0):
    """Interpolate gridded values (lat and lon in degrees) onto the MW grid of
    band-limit L_hi, transform and truncate to L_OUT.  Prints the relative error of the
    transform round trip at L_hi, i.e. the part of the field above band-limit L_hi."""
    lon = np.asarray(lon, float) % 360
    order = np.argsort(lon)
    lon, values = lon[order], values[:, order]
    # periodic in longitude
    lon = np.concatenate([lon[-1:] - 360, lon, lon[:1] + 360])
    values = np.concatenate([values[:, -1:], values, values[:, :1]], axis=1)
    lat = np.asarray(lat, float)
    if lat[0] > lat[-1]:
        lat, values = lat[::-1], values[::-1]
    interp = RegularGridInterpolator(
        (lat, lon), values, bounds_error=False, fill_value=0.0
    )
    thetas, phis = transforms.sphere_angles(L_hi)
    tt, pp = np.meshgrid(90 - np.degrees(thetas), np.degrees(phis), indexing="ij")
    f = interp(np.stack([tt, pp], axis=-1))
    flm_hi = np.asarray(transforms.forward(f, L_hi, spin))
    flm = flm_hi[:L_OUT, L_hi - L_OUT : L_hi + L_OUT - 1]
    # band-limited round trip: the part of f above L_OUT is what truncation discards
    f_lo = np.asarray(transforms.inverse(flm, L_OUT, spin))
    f_hi_lo = np.asarray(transforms.inverse(flm_hi, L_hi, spin))
    err = np.linalg.norm(f_hi_lo - f) / np.linalg.norm(f)
    print(
        f"    transform round trip at L={L_hi}: {err:.1e}; |f| at L={L_OUT}: "
        f"{np.abs(f_lo).max():.3g}"
    )
    return flm


def save(name, flm, spin, units, source, citation):
    np.savez_compressed(
        OUT / f"{name}.npz",
        flm=flm.astype(np.complex64),
        L=L_OUT,
        spin=spin,
        units=units,
        source=source,
        citation=citation,
        licence="Public domain (U.S. Government work)"
        if "noaa" in source
        else "Generated, CC0",
        created=datetime.date.today().isoformat(),
    )
    print(f"    saved {name}.npz ({(OUT / f'{name}.npz').stat().st_size / 1e3:.0f} kB)")


def check_spin1_convention(L=16):
    r"""A tangent vector field v = v_theta e_theta + v_phi e_phi is the spin-1 field
    f = v_theta + i v_phi.  With eth-bar of physics.ethbar,
    ethbar f = -(div v + i curl v), curl v = e_r . (nabla x v).  Checked on a gradient
    field (v = grad sin(theta) cos(phi)) and on a solid-body rotation
    (v_phi = sin(theta))."""
    thetas, phis = transforms.sphere_angles(L)
    t, p = np.meshgrid(thetas, phis, indexing="ij")
    cases = [
        # v_theta, v_phi, divergence, curl
        (np.cos(t) * np.cos(p), -np.sin(p), -2 * np.sin(t) * np.cos(p), 0 * t),
        (0 * t, np.sin(t), 0 * t, 2 * np.cos(t)),
    ]
    for v_t, v_p, div, curl in cases:
        flm = transforms.forward(v_t + 1j * v_p, L, spin=1)
        g = np.asarray(transforms.inverse(physics.ethbar(flm, 1), L, spin=0))
        err = np.max(np.abs(g + (div + 1j * curl)))
        assert err < 1e-10, f"spin-1 convention check failed: {err:.2e}"
    print("spin-1 convention: ethbar(v_theta + i v_phi) = -(div v + i curl v)  [ok]")


def earth_topography():
    print("Earth topography (ETOPO 2022, 60 arcsec, every 15th sample)")
    d = netCDF4.Dataset(ETOPO)
    lat, lon = d["lat"][::15], d["lon"][::15]
    z = np.asarray(d["z"][::15, ::15], float) / 1e3
    # 2 x 2 block means: 0.5-degree cells
    z = z.reshape(z.shape[0] // 2, 2, z.shape[1] // 2, 2).mean(axis=(1, 3))
    lat = lat.reshape(-1, 2).mean(1)
    lon = lon.reshape(-1, 2).mean(1)
    flm = to_harmonic(z, lat, lon, L_hi=256)
    save(
        "earth_topography",
        flm,
        0,
        "km",
        ETOPO,
        "NOAA National Centers for Environmental Information (2022), ETOPO 2022 15 "
        "Arc-Second Global Relief Model, doi:10.25921/fd45-gt74",
    )


def wind():
    print("Wind (NCEP/NCAR reanalysis, sigma 0.995, January mean 1991-2020)")
    u = netCDF4.Dataset(NCEP.format("u"))
    v = netCDF4.Dataset(NCEP.format("v"))
    lat, lon = u["lat"][:], u["lon"][:]
    uwnd = np.asarray(u["uwnd"][0], float)
    vwnd = np.asarray(v["vwnd"][0], float)
    flm = to_harmonic(-vwnd + 1j * uwnd, lat, lon, L_hi=128, spin=1)
    save(
        "wind",
        flm,
        1,
        "m/s",
        NCEP.format("u/v"),
        "Kalnay et al. (1996), The NCEP/NCAR 40-year reanalysis project, "
        "Bull. Amer. Meteor. Soc. 77, 437",
    )


def ocean_currents():
    print("Ocean surface currents (GODAS, 5 m, 2025 annual mean)")
    u = netCDF4.Dataset(GODAS.format("u"))
    v = netCDF4.Dataset(GODAS.format("v"))
    lat, lon = u["lat"][:], u["lon"][:]
    ucur = np.ma.filled(u["ucur"][:, 0].mean(axis=0), 0.0).astype(float)
    vcur = np.ma.filled(v["vcur"][:, 0].mean(axis=0), 0.0).astype(float)
    flm = to_harmonic(-vcur + 1j * ucur, lat, lon, L_hi=256, spin=1)
    save(
        "ocean_currents",
        flm,
        1,
        "m/s",
        GODAS.format("u/v"),
        "Behringer & Xue (2004), Evaluation of the global ocean data assimilation "
        "system at NCEP: the Pacific Ocean, AMS Symposium on Integrated Observing and "
        "Assimilation Systems",
    )


def eisenstein_hu_nowiggle(k, h, om, ob, ns=0.965):
    """Linear matter power spectrum (arbitrary normalisation), k in 1/Mpc, with the
    no-wiggle transfer function of Eisenstein & Hu (1998)."""
    omh2, obh2, fb = om * h**2, ob * h**2, ob / om
    theta = 2.7255 / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2**0.75)
    alpha = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb**2
    gamma = om * h * (alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))
    q = k * theta**2 / gamma / h
    L0 = np.log(2 * np.e + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return k**ns * (L0 / (L0 + C0 * q**2)) ** 2


def convergence_cl(ells, z_s=1.0, sigma8=0.81):
    """Lensing convergence power spectrum for one source plane, by the Limber
    approximation with the linear power spectrum grown by the linear growth factor."""
    cosmo = Planck18
    h, om, ob = cosmo.h, cosmo.Om0, cosmo.Ob0
    # sigma8 normalisation
    k = np.logspace(-4, 2, 4000)
    x = k * 8 / h
    w = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    pk = eisenstein_hu_nowiggle(k, h, om, ob)
    amp = sigma8**2 / np.trapezoid(k**2 * pk * w**2 / (2 * np.pi**2), k)
    # linear growth D(a) normalised to 1 today
    a_grid = np.linspace(1e-3, 1, 4000)
    E = cosmo.efunc(1 / a_grid - 1)
    integrand = 1 / (a_grid * E) ** 3
    cum = np.concatenate(
        [[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(a_grid))]
    )
    D = E * cum
    D /= D[-1]
    z = np.linspace(1e-3, z_s, 400)
    chi = cosmo.comoving_distance(z).value
    chi_s = cosmo.comoving_distance(z_s).value
    c_over_H0 = 299792.458 / cosmo.H0.value
    growth = np.interp(1 / (1 + z), a_grid, D)
    weight = 1.5 * om / c_over_H0**2 * (1 + z) * chi * (chi_s - chi) / chi_s
    dchi_dz = c_over_H0 / cosmo.efunc(z)
    cl = []
    for ell in ells:
        kk = (ell + 0.5) / chi
        p = amp * eisenstein_hu_nowiggle(kk, h, om, ob) * growth**2
        cl.append(np.trapezoid(weight**2 / chi**2 * p * dchi_dz, z))
    return np.array(cl)


def convergence(shift=0.02):
    """A shifted lognormal convergence field: the Gaussian field g has power spectrum
    C_l / shift^2, and kappa = shift (exp(g - var(g)/2) - 1)."""
    print("Convergence (simulated, lognormal, source plane z = 1)")
    L_hi = 256
    ells = np.arange(L_hi)
    cl = np.zeros(L_hi)
    cl[2:] = convergence_cl(ells[2:])
    rng = np.random.default_rng(SEED)
    glm = np.zeros((L_hi, 2 * L_hi - 1), complex)
    for ell in range(2, L_hi):
        sd = np.sqrt(cl[ell]) / shift
        glm[ell, L_hi - 1] = sd * rng.standard_normal()
        a = sd * (rng.standard_normal(ell) + 1j * rng.standard_normal(ell)) / np.sqrt(2)
        m = np.arange(1, ell + 1)
        glm[ell, L_hi - 1 + m] = a
        glm[ell, L_hi - 1 - m] = (-1) ** m * np.conj(a)
    g = np.real(np.asarray(transforms.inverse(glm, L_hi)))
    var = np.sum((2 * ells + 1) * cl) / (4 * np.pi) / shift**2
    kappa = shift * (np.exp(g - var / 2) - 1)
    print(
        f"    std(g) = {np.sqrt(var):.2f}; "
        f"kappa range [{kappa.min():.3f}, {kappa.max():.3f}]"
    )
    flm = np.array(transforms.forward(kappa, L_hi))[
        :L_OUT, L_hi - L_OUT : L_hi + L_OUT - 1
    ]
    flm[:2] = 0  # no monopole or dipole in a convergence field
    save(
        "convergence",
        flm,
        0,
        "dimensionless",
        "simulated",
        "Linear power spectrum of Eisenstein & Hu (1998), ApJ 496, 605, with "
        "Planck 2018 cosmology; shifted lognormal field",
    )


if __name__ == "__main__":
    check_spin1_convention()
    earth_topography()
    wind()
    ocean_currents()
    convergence()
