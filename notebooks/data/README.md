# Example fields

Harmonic coefficients of the fields used in the "Real fields" section of the notebook,
on the McEwen-Wiaux sampling, at band-limit L = 64, stored as complex64 in `.npz` files
with the keys `flm`, `L`, `spin`, `units`, `source`, `citation`, `licence` and `created`.
`make_example_fields.py` builds them; it reads the data over OPeNDAP and needs `netCDF4`,
`scipy` and `astropy`, which are not dependencies of s2conv:

```bash
python notebooks/data/make_example_fields.py
```

| File | Spin | Field | Source | Licence |
|:--|:--|:--|:--|:--|
| `earth_topography.npz` | 0 | Surface elevation and bathymetry (km), 0.5° cell means | NOAA NCEI, ETOPO 2022 60″ surface elevation, doi:10.25921/fd45-gt74 | Public domain |
| `wind.npz` | 1 | Near-surface (σ = 0.995) wind (m/s), January mean 1991–2020 | NOAA PSL, NCEP/NCAR reanalysis (Kalnay et al. 1996, Bull. Amer. Meteor. Soc. 77, 437) | Public domain |
| `ocean_currents.npz` | 1 | Surface (5 m) ocean currents (m/s), 2025 annual mean; zero on land and outside 74.5° S – 64.5° N | NOAA PSL, GODAS (Behringer & Xue 2004) | Public domain |
| `convergence.npz` | 0 | Simulated weak-lensing convergence for sources at z = 1: shifted lognormal field (shift 0.02) with the Limber power spectrum of the linear Eisenstein & Hu (1998) no-wiggle spectrum, Planck 2018 cosmology, σ8 = 0.81, seed 2026 | Generated | CC0 |

A tangent vector field with northward and eastward components (v, u) is stored as the
spin-1 field f = v_θ + i v_φ = −v + i u, so that ð̄f = −(div + i curl), where curl is the
radial component of the curl; the script checks this on analytic fields before building
the data.
