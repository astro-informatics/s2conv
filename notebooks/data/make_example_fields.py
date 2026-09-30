"""
Build the example fields of the notebook's "Real fields" section.

Each field is sampled on a McEwen-Wiaux grid at a high band-limit, transformed with
s2conv, truncated to band-limit L_OUT and saved as harmonic coefficients (complex64)
in an .npz file next to this script.  The geophysical data are read over OPeNDAP, so
only the slices used are downloaded; the Planck files are downloaded once to CACHE.
Run from the root of the repository, with

    python notebooks/data/make_example_fields.py              # every field
    python notebooks/data/make_example_fields.py cmb planck   # some of them

This needs netCDF4 (with OPeNDAP support), scipy, healpy and camb, which are not
dependencies of s2conv.  See README.md in this directory for provenance and licences.
"""

import datetime
import sys
import urllib.request
from pathlib import Path

import camb
import healpy as hp
import jax
import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator

jax.config.update("jax_enable_x64", True)

from s2fft.sampling.reindex import flm_hp_to_2d_fast  # noqa: E402
from scipy.special import lpmv  # noqa: E402

from s2conv import transforms  # noqa: E402  (after enabling 64-bit precision)
from s2conv.convolutions import physics  # noqa: E402
from s2conv.utils.evaluation import evaluate  # noqa: E402
from s2conv.utils.indexing import conjugate  # noqa: E402

OUT = Path(__file__).parent
L_OUT = 128  # band-limit of the stored fields, where the data resolve it
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
PLANCK = "https://irsa.ipac.caltech.edu/data/Planck/release_3/"
PLANCK_SMICA = (
    PLANCK
    + "all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits"
)
PLANCK_THEORY = (
    PLANCK + "ancillary-data/cosmoparams/"
    "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
)
EGM2008 = (
    "https://icgem.gfz-potsdam.de/getmodel/gfc/"
    "c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc"
)
IGRF = "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt"
COASTLINE = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_110m_coastline.geojson"
)
# large downloads (the SMICA map is 2 GB) are kept here between runs
CACHE = Path.home() / ".cache" / "s2conv-example-data"


def to_harmonic(values, lat, lon, L_hi, spin=0, L_out=L_OUT):
    """Interpolate gridded values (lat and lon in degrees) onto the MW grid of
    band-limit L_hi, transform and truncate to L_out.  Prints the relative error of the
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
    flm = flm_hi[:L_out, L_hi - L_out : L_hi + L_out - 1]
    f_lo = np.asarray(transforms.inverse(flm, L_out, spin))
    f_hi_lo = np.asarray(transforms.inverse(flm_hi, L_hi, spin))
    err = np.linalg.norm(f_hi_lo - f) / np.linalg.norm(f)
    print(
        f"    transform round trip at L={L_hi}: {err:.1e}; |f| at L={L_out}: "
        f"{np.abs(f_lo).max():.3g}"
    )
    return flm


def save(name, flm, spin, units, source, citation, licence=None):
    """Save the coefficients, at their own band-limit.  For a real scalar field only
    the orders m >= 0 are stored (real=True), since f_{l,-m} = (-1)^m f_lm^*."""
    L = flm.shape[0]
    real = spin == 0 and np.allclose(
        flm, conjugate(flm, 0), atol=1e-6 * np.abs(flm).max()
    )
    np.savez_compressed(
        OUT / f"{name}.npz",
        flm=(flm[:, L - 1 :] if real else flm).astype(np.complex64),
        real=real,
        L=L,
        spin=spin,
        units=units,
        source=source,
        citation=citation,
        licence=licence
        or (
            "Public domain (U.S. Government work)"
            if "noaa" in source
            else "Generated, CC0"
        ),
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
    # the 2.5-degree grid resolves band-limits up to about 72
    flm = to_harmonic(-vwnd + 1j * uwnd, lat, lon, L_hi=128, spin=1, L_out=64)
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


def standard_alm(L, rng, n=1):
    """n independent sets of harmonic coefficients of real fields with unit variance
    per coefficient, [n, L, 2L - 1]; multiplied by sqrt(C_l) they give Gaussian fields
    with power spectrum C_l."""
    alm = np.zeros((n, L, 2 * L - 1), complex)
    for ell in range(L):
        alm[:, ell, L - 1] = rng.standard_normal(n)
        m = np.arange(1, ell + 1)
        a = rng.standard_normal((n, ell)) + 1j * rng.standard_normal((n, ell))
        alm[:, ell, L - 1 + m] = a / np.sqrt(2)
        alm[:, ell, L - 1 - m] = (-1) ** m * np.conj(a) / np.sqrt(2)
    return alm


def planck2018_camb():
    """CAMB parameters for the Planck 2018 base-LCDM best fit (TT,TE,EE+lowE+lensing,
    Planck Collaboration 2020, A&A 641, A6, Table 2), with HMcode-2020 non-linear
    corrections to the matter power spectrum."""
    return camb.set_params(
        H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, tau=0.0544,
        As=2.100e-9, ns=0.9649, halofit_version="mead2020", WantTransfer=True,
    )  # fmt: skip


def convergence_cl(L, z_median=0.9):
    r"""Weak-lensing convergence power spectrum for the source redshift distribution
    n(z) ~ z^2 exp(-(z/z_0)^{3/2}), z_0 = z_median / 1.412 (Smail et al. 1994; the
    Euclid forecast distribution, Euclid Collaboration: Blanchard et al. 2020), by
    the Limber approximation with the non-linear matter power spectrum of CAMB for the
    Planck 2018 cosmology."""
    pars = planck2018_camb()
    results = camb.get_background(pars)
    pk = camb.get_matter_power_interpolator(
        pars, zmax=4.0, kmax=50.0, nonlinear=True, hubble_units=False, k_hunit=False
    )
    z = np.linspace(0.005, 4.0, 800)
    chi = results.comoving_radial_distance(z)
    dchi_dz = 299792.458 / results.hubble_parameter(z)
    n = z**2 * np.exp(-((z / (z_median / 1.412)) ** 1.5))
    n /= np.trapezoid(n, z)
    # lensing efficiency: int_z^inf dz' n(z') (chi' - chi) / chi'
    eff = np.array(
        [np.trapezoid(n[i:] * (1 - chi[i] / chi[i:]), z[i:]) for i in range(len(z))]
    )
    h = pars.H0 / 100
    omega_m = (pars.ombh2 + pars.omch2 + pars.omnuh2) / h**2
    q = 1.5 * omega_m * (pars.H0 / 299792.458) ** 2 * chi * (1 + z) * eff
    cl = np.zeros(L)
    for ell in range(2, L):
        k = (ell + 0.5) / chi
        p = pk.P(z, k, grid=False)
        cl[ell] = np.trapezoid(q**2 / chi**2 * p * dchi_dz, z)
    return cl


def convergence(shift=0.025):
    """A shifted lognormal convergence field: the Gaussian field g has power spectrum
    C_l / shift^2 and kappa = shift (exp(g - var(g)/2) - 1), with the shift of
    Hilbert, Hartlap & Schneider (2011, A&A 536, A85) at z = 0.9."""
    print(
        "Convergence (simulated: CAMB non-linear spectrum, Euclid-like n(z), lognormal)"
    )
    L_hi = 256
    cl = convergence_cl(L_hi)
    print(f"    C_l at l = 10, 30, 60: {cl[10]:.2e}, {cl[30]:.2e}, {cl[60]:.2e}")
    rng = np.random.default_rng(SEED)
    glm = standard_alm(L_hi, rng)[0] * np.sqrt(cl)[:, None] / shift
    g = np.real(np.asarray(transforms.inverse(glm, L_hi)))
    var = np.sum((2 * np.arange(L_hi) + 1) * cl) / (4 * np.pi) / shift**2
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
        "Limber power spectrum of the CAMB non-linear matter power spectrum (HMcode "
        "2020) for the Planck 2018 cosmology and a Euclid-like source distribution "
        "with median redshift 0.9; shifted lognormal field (shift 0.025)",
    )


def cmb_simulation():
    """Gaussian CMB temperature and polarisation with the Planck 2018 best-fit
    lensed power spectra (TT, TE, EE and the lensing BB), in muK."""
    print("CMB (simulated, Planck 2018 best-fit lensed spectra)")
    path = cached(PLANCK_THEORY)
    ell, tt, te, ee, bb = np.loadtxt(path, usecols=(0, 1, 2, 3, 4), unpack=True)
    to_cl = 2 * np.pi / (ell * (ell + 1))  # the file tabulates D_l = l(l+1) C_l / 2 pi
    cl = {}
    for key, d in (("tt", tt), ("te", te), ("ee", ee), ("bb", bb)):
        cl[key] = np.zeros(L_OUT)
        cl[key][2:] = (d * to_cl)[: L_OUT - 2]  # the file starts at l = 2
    rng = np.random.default_rng(SEED + 1)
    z = standard_alm(L_OUT, rng, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_amp = np.sqrt(cl["tt"])
        e_from_t = np.where(cl["tt"] > 0, cl["te"] / t_amp, 0)
        e_amp = np.sqrt(np.clip(cl["ee"] - e_from_t**2, 0, None))
    t = t_amp[:, None] * z[0]
    e = e_from_t[:, None] * z[0] + e_amp[:, None] * z[1]
    b = np.sqrt(cl["bb"])[:, None] * z[2]
    p, _ = physics.eb_to_polarisation(e, b)
    cite = (
        "Planck Collaboration (2020), Planck 2018 results VI, A&A 641, A6: best-fit "
        "lensed power spectra (base-plikHM-TTTEEE-lowl-lowE-lensing); Gaussian "
        "simulation"
    )
    save("cmb_sim_temperature", t, 0, "muK_CMB", "simulated", cite)
    save("cmb_sim_polarisation", p, 2, "muK_CMB", "simulated", cite)


def cmb_planck(nside_low=256):
    """The Planck 2018 SMICA CMB temperature and polarisation maps (Galactic
    coordinates, muK), degraded to nside_low and analysed to band-limit L_OUT."""
    print("CMB (Planck 2018 SMICA)")
    path = cached(PLANCK_SMICA)
    maps = hp.read_map(path, field=(0, 1, 2), dtype=np.float32)
    maps = [hp.ud_grade(1e6 * m.astype(float), nside_low) for m in maps]
    alm = hp.map2alm(maps, lmax=L_OUT - 1, pol=True, iter=3)
    # undo the smoothing by the pixels of the degraded map
    pw_t, pw_p = hp.pixwin(nside_low, pol=True, lmax=L_OUT - 1)
    pw_p = np.where(pw_p > 0, pw_p, 1.0)
    alm = [
        hp.almxfl(alm[0], 1 / pw_t),
        hp.almxfl(alm[1], 1 / pw_p),
        hp.almxfl(alm[2], 1 / pw_p),
    ]
    t, e, b = (np.asarray(flm_hp_to_2d_fast(a, L_OUT)) for a in alm)
    p, _ = physics.eb_to_polarisation(e, b)
    # s2conv's spin-2 field -(E + i B) is healpy's Q + i U: check on the degraded map
    theta, phi = hp.pix2ang(nside_low, np.arange(0, 12 * nside_low**2, 997))
    q_u = hp.alm2map(alm, nside_low, lmax=L_OUT - 1, pol=True, pixwin=False)
    ref = (q_u[1] + 1j * q_u[2])[np.arange(0, 12 * nside_low**2, 997)]
    err = np.max(np.abs(np.asarray(evaluate(p, 2, theta, phi)) - ref)) / np.max(
        np.abs(ref)
    )
    assert err < 1e-8, f"spin-2 convention check failed: {err:.2e}"
    print(
        f"    spin-2 convention: -(E + iB) synthesises healpy's Q + iU  [ok, {err:.0e}]"
    )
    cite = (
        "Planck Collaboration (2020), Planck 2018 results IV, A&A 641, A4: SMICA CMB "
        "map, Planck Legacy Archive (" + PLANCK_SMICA + ")"
    )
    licence = (
        "Based on observations obtained with Planck (http://www.esa.int/Planck), an "
        "ESA science mission; free use with acknowledgement"
    )
    save("cmb_planck_temperature", t, 0, "muK_CMB", PLANCK_SMICA, cite, licence)
    save("cmb_planck_polarisation", p, 2, "muK_CMB", PLANCK_SMICA, cite, licence)


def real_to_complex(c, s, schmidt=False):
    r"""Coefficients f_lm of the complex orthonormal harmonics (with the Condon-Shortley
    phase) of the real field sum_{l, m >= 0} (C_lm cos m phi + S_lm sin m phi) P_lm(cos
    theta), with c[l, m] and s[l, m], [L, L], for the fully normalised Legendre
    functions of geodesy (mean square of each term over the sphere 1, schmidt=False)
    or the Schmidt semi-normalised ones of geomagnetism (schmidt=True); neither has the
    Condon-Shortley phase.  Then f_l0 = sqrt(4 pi) C_l0 and, for m > 0,
    f_lm = sqrt(2 pi) (-1)^m (C_lm - i S_lm), f_l,-m = sqrt(2 pi) (C_lm + i S_lm), each
    divided by sqrt(2l + 1) in the Schmidt case."""
    L = c.shape[0]
    el = np.arange(L)
    scale = np.sqrt(4 * np.pi) / (np.sqrt(2 * el + 1) if schmidt else 1.0)
    flm = np.zeros((L, 2 * L - 1), complex)
    flm[:, L - 1] = scale * c[:, 0]
    for m in range(1, L):
        z = scale / np.sqrt(2) * (c[:, m] - 1j * s[:, m])
        flm[:, L - 1 + m] = (-1) ** m * z
        flm[:, L - 1 - m] = np.conj(z)
    return flm


def check_real_to_complex(L=8, schmidt=False):
    """Check real_to_complex against a direct evaluation with scipy's Legendre
    functions, which carry the Condon-Shortley phase."""
    from math import factorial

    rng = np.random.default_rng(0)
    c, s = np.tril(rng.standard_normal((L, L))), np.tril(rng.standard_normal((L, L)))
    s[:, 0] = 0
    theta, phi = rng.uniform(0.1, 3.0, 20), rng.uniform(0, 2 * np.pi, 20)
    direct = np.zeros(20)
    for ell in range(L):
        for m in range(ell + 1):
            norm = np.sqrt((2 - (m == 0)) * factorial(ell - m) / factorial(ell + m))
            if not schmidt:
                norm *= np.sqrt(2 * ell + 1)
            p = (-1) ** m * lpmv(m, ell, np.cos(theta)) * norm
            direct += (c[ell, m] * np.cos(m * phi) + s[ell, m] * np.sin(m * phi)) * p
    f = np.asarray(evaluate(real_to_complex(c, s, schmidt), 0, theta, phi)).real
    assert np.max(np.abs(f - direct)) < 1e-10, "real_to_complex check failed"


def gravity(L_out=L_OUT):
    """The geoid height of EGM2008 to band-limit L_out, in the spherical
    approximation N = a sum (C_lm - C_lm^WGS84) P_lm (Bruns' formula with gamma =
    GM / a^2), where the normal field of the WGS84 ellipsoid has only the even zonal
    terms below."""
    print("Gravity (EGM2008 geoid)")
    check_real_to_complex(schmidt=False)
    header, c, s = {}, np.zeros((L_out, L_out)), np.zeros((L_out, L_out))
    with urllib.request.urlopen(
        EGM2008
    ) as response:  # ordered by degree: read the start
        for raw in response:
            words = raw.decode().split()
            if not words:
                continue
            if words[0] == "gfc":
                n, m = int(words[1]), int(words[2])
                if n >= L_out:
                    break
                c[n, m] = float(words[3].replace("d", "e"))
                s[n, m] = float(words[4].replace("d", "e"))
            elif len(words) == 2:
                header[words[0]] = words[1]
    a = float(header["radius"])
    wgs84 = {2: -0.484166774985e-3, 4: 0.790303733511e-6, 6: -0.168724961151e-8,
             8: 0.346052468394e-11, 10: -0.265002225747e-14}  # fmt: skip
    c[0, 0] = 0.0
    for n, cn in wgs84.items():
        c[n, 0] -= cn
    flm = a * real_to_complex(c, s)
    f = np.asarray(transforms.inverse(flm, L_out)).real
    print(f"    geoid height range [{f.min():.1f}, {f.max():.1f}] m")
    save(
        "geoid",
        flm,
        0,
        "m",
        EGM2008,
        "Pavlis, Holmes, Kenyon & Factor (2012), The development and evaluation of the "
        "Earth Gravitational Model 2008 (EGM2008), J. Geophys. Res. 117, B04406; via "
        "ICGEM (Ince et al. 2019)",
        "EGM2008 (U.S. National Geospatial-Intelligence Agency): public release",
    )


def magnetic(epoch="2025.0"):
    """The geomagnetic main-field potential of IGRF-14 at the Earth's surface, V / a in
    nT, so that B_r = sum (l + 1) v_lm Y_lm and B_theta + i B_phi = eth v there."""
    print(f"Geomagnetic field (IGRF-14, epoch {epoch})")
    check_real_to_complex(schmidt=True)
    lines = urllib.request.urlopen(IGRF).read().decode().splitlines()
    rows = [line.split() for line in lines if line and not line.startswith("#")]
    head = next(r for r in rows if r[0] == "g/h")
    col = head.index(epoch)
    L = 1 + max(int(r[1]) for r in rows if r[0] in "gh")
    g, h = np.zeros((L, L)), np.zeros((L, L))
    for r in rows:
        if r[0] in ("g", "h"):
            (g if r[0] == "g" else h)[int(r[1]), int(r[2])] = float(r[col])
    flm = real_to_complex(g, h, schmidt=True)
    # check against the known field: London (51.5 N, 0.1 W) in 2025
    theta, phi = np.radians([90 - 51.5]), np.radians([-0.1])
    el = np.arange(L)
    b_r = evaluate(flm * (el + 1)[:, None], 0, theta, phi).real[0]
    b_h = evaluate(physics.eth(flm, 0), 1, theta, phi)[0]
    x, y, z = -b_h.real, b_h.imag, -b_r
    print(
        f"    London: declination {np.degrees(np.arctan2(y, x)):.1f} deg, inclination "
        f"{np.degrees(np.arctan2(z, np.hypot(x, y))):.1f} deg, "
        f"intensity {np.sqrt(x**2 + y**2 + z**2):.0f} nT"
    )
    save(
        "geomagnetic_potential",
        flm,
        0,
        "nT (V / a)",
        IGRF,
        "Alken et al. (2021), International Geomagnetic Reference Field: the "
        "thirteenth generation, Earth Planets Space 73, 49; IGRF-14 coefficients, IAGA "
        "Division V-MOD",
        "IAGA: free use with acknowledgement",
    )


def coastlines():
    """Natural Earth coastlines at 1:110m, as longitude and latitude in degrees with
    NaN between the lines."""
    print("Coastlines (Natural Earth 1:110m)")
    import json

    features = json.load(urllib.request.urlopen(COASTLINE))["features"]
    lon, lat = [], []
    for feature in features:
        geometry = feature["geometry"]
        parts = (
            [geometry["coordinates"]]
            if geometry["type"] == "LineString"
            else geometry["coordinates"]
        )
        for part in parts:
            xy = np.asarray(part)
            lon += [*xy[:, 0], np.nan]
            lat += [*xy[:, 1], np.nan]
    np.savez_compressed(
        OUT / "coastlines.npz",
        lon=np.asarray(lon, np.float32),
        lat=np.asarray(lat, np.float32),
        source=COASTLINE,
        licence="Natural Earth: public domain",
    )
    print(f"    saved coastlines.npz ({len(lon)} points)")


def cached(url):
    """Download url to CACHE once, and return the local path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"    downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


FIELDS = {
    "topography": earth_topography,
    "wind": wind,
    "ocean": ocean_currents,
    "convergence": convergence,
    "cmb": cmb_simulation,
    "planck": cmb_planck,
    "gravity": gravity,
    "magnetic": magnetic,
    "coastlines": coastlines,
}

if __name__ == "__main__":
    # build every field, or those named on the command line, e.g. "cmb planck"
    check_spin1_convention()
    for name in sys.argv[1:] or FIELDS:
        FIELDS[name]()
