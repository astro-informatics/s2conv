"""
Export the figures of the notebook real_fields.py for the article.

The notebook is run in print mode, in which each figure is drawn at the text width
of the one-column layout of the article (6.5 inches) with 8 pt text and panel titles
reduced to their letters, which the captions describe.  Each figure is saved as a
vector PDF in notebooks/figures, with the map images rasterised at 300 dpi and the
glyphs, coastlines, colour bars and text as vectors, together with
real_fields_captions.md, which gives the caption of each figure as computed by the
notebook, the data it uses and the commit of the code.  Run from the root of the
repository, with

    python notebooks/export_real_fields_figures.py
"""

import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "figures"
sys.path.insert(0, str(HERE))

import real_fields  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})

# the data files each figure uses, for the provenance in the captions file
DATA = {
    "01_overview": [
        "earth_topography",
        "geoid",
        "geomagnetic_potential",
        "wind",
        "ocean_currents",
        "cmb_planck_temperature",
        "cmb_planck_polarisation",
        "convergence",
    ],
    "02_topography": ["earth_topography"],
    "03_gravity": ["geoid"],
    "04_geomagnetism": ["geomagnetic_potential"],
    "05_wind": ["wind"],
    "06_wind_smoothing": ["wind"],
    "07_ocean_currents": ["ocean_currents"],
    "08_cmb_polarisation": ["cmb_planck_temperature", "cmb_planck_polarisation"],
    "09_lensing": ["convergence"],
}


def main():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "notebooks/real_fields.py", "s2conv"],
        capture_output=True,
        text=True,
        cwd=HERE.parent,
    ).stdout.strip()
    real_fields.PRINT["on"] = True
    _, defs = real_fields.app.run()
    L = defs["L"]
    OUT.mkdir(exist_ok=True)
    lines = [
        "# Figures of the notebook real_fields.py",
        "",
        f"Produced by `notebooks/export_real_fields_figures.py` from s2conv commit "
        f"`{commit}`{' (with uncommitted changes)' if dirty else ''}, at band-limit "
        f"L = {L}, the default of the notebook.  Each figure is "
        f"{real_fields.PRINT['width']} inches wide, with 8 pt text; the panel titles "
        "are reduced to their letters, and the captions below, which the notebook "
        "computes, describe the panels.  The map images are rasterised at 300 dpi; "
        "glyphs, coastlines, colour bars and text are vectors.  The Earth fields are "
        "tapered by "
        "exp(-l(l+1)/L^2) (Data, in the introduction of the notebook); the directional "
        "kernels of the topography have azimuthal orders |m| < 5.",
        "",
    ]
    for name, fig in real_fields.PRINT["figures"].items():
        path = OUT / f"real_fields_{name}.pdf"
        fig.savefig(path, dpi=300)
        width, height = fig.get_size_inches()
        lines += [
            f"## real_fields_{name}.pdf",
            "",
            f"Size {width:.2f} x {height:.2f} inches, "
            f"{path.stat().st_size / 1e3:.0f} kB.",
            "",
            real_fields.PRINT["captions"][name].strip(),
            "",
            "Data:",
            "",
        ]
        for field in DATA[name]:
            data = np.load(HERE / "data" / f"{field}.npz")
            lines.append(
                f"- `{field}.npz` (spin {int(data['spin'])}, stored at L = "
                f"{int(data['L'])}, {data['units']}): {data['citation']}; "
                f"licence: {data['licence']}"
            )
        lines.append("")
        print(f"saved {path.name} ({path.stat().st_size / 1e3:.0f} kB)")
    (OUT / "real_fields_captions.md").write_text("\n".join(lines))
    print("saved real_fields_captions.md")


if __name__ == "__main__":
    main()
