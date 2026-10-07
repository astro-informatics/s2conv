"""
Export the figures of the notebook real_fields.py for the article.

The notebook is run in print mode, in which each figure is drawn at the text width
of the one-column layout of the article (6.5 inches), with at most three panels per
row, each as wide as its column, 8 pt text, and the letter of each panel, which the
captions describe, beneath its colour bar.  Each figure is saved as a vector PDF in
notebooks/figures, with the map images rasterised at 300 dpi and the glyphs,
coastlines, colour bars and text as vectors, together with real_fields_captions.md,
which gives the caption of each figure as computed by the notebook, the data it uses
and the commit of the code.  The notebook is then run in panel mode, which saves each
panel of each figure as its own PDF, real_fields_NN_name_x.pdf for panel (x), without
its letter, for subcaptions in LaTeX, at the width that it occupies in the article
(see PANELS).  Run from the root of the repository, with

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


# the rows of panels of each figure in the article, with the width in inches of the
# panels of each row: the text width of the article is 5.5 inches, so that three
# panels to a row are 1.76 inches wide and two are 2.64
THREE, TWO = 1.76, 2.64
PANELS = {
    "01_overview": [("abc", THREE), ("de", THREE), ("fg", THREE)],
    "02_topography": [("abc", THREE), ("def", THREE)],
    "03_gravity": [("abc", THREE), ("def", THREE)],
    "04_geomagnetism": [("abc", THREE), ("def", THREE)],
    "05_wind": [("abc", THREE), ("def", THREE)],
    "06_wind_smoothing": [("ab", TWO), ("cd", TWO)],
    "07_ocean_currents": [("ab", TWO), ("cd", TWO)],
    "08_cmb_polarisation": [("ab", THREE), ("cde", THREE)],
    "09_lensing": [("ab", TWO), ("cd", TWO)],
}


def main():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "notebooks/real_fields.py",
            "notebooks/export_real_fields_figures.py",
            "s2conv",
        ],
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
        f"{real_fields.PRINT['width']} inches wide, with at most three panels per row, "
        "each as wide as its column, and 8 pt text; each panel is labelled by its "
        "letter beneath its colour bar, and the captions below, which the notebook "
        "computes, describe the panels.  The map images are rasterised at 300 dpi; "
        "glyphs, coastlines, colour bars and text are vectors.  The Earth fields are "
        "tapered by "
        "exp(-l(l+1)/L^2) (Data, in the introduction of the notebook); the directional "
        "kernels of the topography have azimuthal orders |m| < 5.",
        "",
    ]
    for name, fig in real_fields.PRINT["figures"].items():
        path = OUT / f"real_fields_{name}.pdf"
        real_fields.fit_height(name)  # the largest panels that the width allows
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
    # panel mode: every panel its own PDF, at the width it occupies in the article
    real_fields.PRINT.update(
        panels=True,
        panel_widths={
            name: [width for letters, width in rows for _ in letters]
            for name, rows in PANELS.items()
        },
        figures={},
        layouts={},
        panel_figures={},
    )
    real_fields.app.run()
    lines += [
        "# Panels",
        "",
        "Each panel is also saved as its own PDF, real_fields_NN_name_x.pdf for panel "
        "(x), without its letter, for subcaptions in LaTeX, at the width it occupies "
        "in the article (text width 5.5 inches): 1.76 inches for three panels to a "
        "row and 2.64 for two, with its colour bar and unit beneath it.  Maps keep "
        "their 2:1 shape across the width of the panel; the orthographic views have "
        "diameters of 0.62 of the width of the panel in Figs. 4(f), 6, 7(d) and 9(d), "
        "and 0.82 elsewhere.",
        "",
    ]
    for name, rows in PANELS.items():
        real_fields.fit_panel(name)
        letters = "".join(row for row, _ in rows)
        lines += [
            f"## real_fields_{name}: rows "
            + ", ".join(f"({','.join(row)}) at {width} in" for row, width in rows),
            "",
        ]
        for letter, (fig, _, _) in zip(
            letters, real_fields.PRINT["panel_figures"][name]
        ):
            path = OUT / f"real_fields_{name}_{letter}.pdf"
            fig.savefig(path, dpi=300)
            width, height = fig.get_size_inches()
            lines.append(
                f"- `{path.name}`: {width:.2f} x {height:.2f} inches, "
                f"{path.stat().st_size / 1e3:.0f} kB"
            )
        lines.append("")
        print(f"saved the {len(letters)} panels of real_fields_{name}")
    (OUT / "real_fields_captions.md").write_text("\n".join(lines))
    print("saved real_fields_captions.md")


if __name__ == "__main__":
    main()
