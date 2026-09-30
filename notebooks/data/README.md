# Example fields

Harmonic coefficients of the fields used in the notebook `real_fields.py`, on the
McEwen-Wiaux sampling, stored as complex64 in `.npz` files with the keys `flm`, `real`,
`L`, `spin`, `units`, `source`, `citation`, `licence` and `created`.  Each field is stored
at the band-limit its data resolve (L = 128, except the wind, at 64, and the IGRF
potential, at 14); for a real scalar field (`real` true) only the orders m ≥ 0 are
stored, since f_{l,−m} = (−1)^m f_lm^*.  `coastlines.npz` holds the Natural Earth
coastlines for the maps.
`make_example_fields.py` builds them.  It reads the geophysical data over OPeNDAP,
downloads the Planck files once to `~/.cache/s2conv-example-data` (the SMICA map is 2 GB),
and needs `netCDF4`, `scipy`, `healpy` and `camb`, which are not dependencies of s2conv:

```bash
python notebooks/data/make_example_fields.py              # every field
python notebooks/data/make_example_fields.py planck gravity  # some of them
```

| File | Spin | Field | Source | Licence |
|:--|:--|:--|:--|:--|
| `earth_topography.npz` | 0 | Surface elevation and bathymetry (km), 0.5° cell means | NOAA NCEI, ETOPO 2022 60″ surface elevation, doi:10.25921/fd45-gt74 | Public domain |
| `wind.npz` | 1 | Near-surface (σ = 0.995) wind (m/s), January mean 1991–2020 | NOAA PSL, NCEP/NCAR reanalysis (Kalnay et al. 1996, Bull. Amer. Meteor. Soc. 77, 437) | Public domain |
| `ocean_currents.npz` | 1 | Surface (5 m) ocean currents (m/s), 2025 annual mean; zero on land and outside 74.5° S – 64.5° N | NOAA PSL, GODAS (Behringer & Xue 2004) | Public domain |
| `convergence.npz` | 0 | Simulated weak-lensing convergence: shifted lognormal field (shift 0.025, Hilbert, Hartlap & Schneider 2011, A&A 536, A85) whose Gaussian field has the Limber power spectrum of the non-linear matter power spectrum of CAMB (HMcode 2020) for the Planck 2018 cosmology and the Euclid-like source distribution n(z) ∝ z² exp[−(z/z₀)^1.5] with median redshift 0.9; seed 2026 | Generated | CC0 |
| `geoid.npz` | 0 | Geoid height (m) of EGM2008 above the WGS84 ellipsoid, to degree 127, in the spherical approximation N = a Σ (C̄_lm − C̄_lm^WGS84) P̄_lm | ICGEM, EGM2008 (Pavlis et al. 2012, J. Geophys. Res. 117, B04406) | Public release (NGA) |
| `geomagnetic_potential.npz` | 0 | Geomagnetic main-field potential V/a (nT) at the Earth's surface, IGRF-14 epoch 2025.0, degree 13 | IAGA V-MOD, IGRF-14 coefficients (`igrf14coeffs.txt`) | Free use with acknowledgement |
| `coastlines.npz` | – | Coastlines at 1:110m, longitude and latitude (degrees) with NaN between lines | Natural Earth | Public domain |
| `cmb_planck_temperature.npz` | 0 | Planck 2018 SMICA CMB temperature (μK), Galactic coordinates, degraded from N_side 2048 to 256 before the harmonic analysis | `COM_CMB_IQU-smica_2048_R3.00_full.fits`, Planck Legacy Archive via IRSA (Planck Collaboration 2020, A&A 641, A4) | Planck (ESA): free use with acknowledgement |
| `cmb_planck_polarisation.npz` | 2 | Planck 2018 SMICA CMB polarisation Q + iU (μK), as above | As above | As above |

A tangent vector field with northward and eastward components (v, u) is stored as the
spin-1 field f = v_θ + i v_φ = −v + i u, so that ð̄f = −(div + i curl), where curl is the
radial component of the curl; the script checks this on analytic fields before building
the data.  The polarisation Q + iU is the spin-2 field −(E + iB) of s2conv, which is the
healpy (and Planck) convention: the script checks that it synthesises healpy's Q + iU.

The Planck maps are based on observations obtained with Planck
(http://www.esa.int/Planck), an ESA science mission with instruments and contributions
directly funded by ESA Member States, NASA, and Canada.
