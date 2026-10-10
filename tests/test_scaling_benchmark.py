"""
The scaling benchmark, benchmarks/scaling.py: its power-law fit, its outputs and its command line.

The fast tests check the fit and the formatting; the slow test runs the four lifted convolutions at two small
band-limits and checks the JSON record and the plot.
"""

import json

import pytest


def _scaling():
    return pytest.importorskip("benchmarks.scaling")


def test_fit_power_law():
    scaling = _scaling()
    bandlimits = [8, 16, 32, 64, 128]
    seconds = [2e-9 * L**3 for L in bandlimits]
    exponent, prefactor = scaling.fit_power_law(bandlimits, seconds, 8)
    assert exponent == pytest.approx(3.0)
    assert prefactor == pytest.approx(2e-9)
    # band-limits below L_min do not enter the fit
    seconds[0] = 1.0
    assert scaling.fit_power_law(bandlimits, seconds, 16)[0] == pytest.approx(3.0)
    assert scaling.fit_power_law(bandlimits, seconds, 128) is None


def test_duration_and_table():
    scaling = _scaling()
    assert scaling.duration(1.234e-4) == "0.123 ms"
    assert scaling.duration(0.2134) == "213 ms"
    assert scaling.duration(12.21) == "12.2 s"
    medians = {"spin_2_1": {8: 1e-3, 16: 8e-3}, "group_G": {8: 2e-3, 16: 1.6e-2}}
    exponents = {"spin_2_1": (3.0, 1.0), "group_G": None}
    table = scaling.markdown_table(medians, [8, 16], exponents, 8)
    assert table.splitlines()[0] == "| L | spin_2_1 | group_G |"
    assert table.splitlines()[-1] == "| Exponent, L ≥ 8 | 3.00 | - |"


def test_band_limits_at_least_N():
    scaling = _scaling()
    assert scaling.parse_args(["-L", "16", "8", "16"]).bandlimits == [8, 16]
    with pytest.raises(SystemExit):
        scaling.parse_args(["-L", str(scaling.N - 1), "8"])


@pytest.mark.slow
def test_scaling_command_line(tmp_path):
    pytest.importorskip("matplotlib")
    scaling = _scaling()
    json_path, plot_path = tmp_path / "scaling.json", tmp_path / "scaling.pdf"
    scaling.main(
        [
            "-L",
            "6",
            "8",
            "--repeats",
            "1",
            "--fit-from",
            "6",
            "--device",
            "default",
            "-o",
            str(json_path),
            "--plot",
            str(plot_path),
            "--plot-from",
            "6",
        ]
    )
    record = json.loads(json_path.read_text())
    assert list(record["median_seconds"]) == list(scaling.OPERATORS)
    for name in scaling.OPERATORS:
        assert set(record["median_seconds"][name]) == {"6", "8"}
        assert len(record["seconds"][name]["8"]) == 1
        assert record["exponents"][name]["fit_from"] == 6
    assert set(record["code"]) == {"s2conv", "benchmark", "s2fft"}
    assert record["hardware"] == {"kind": "default"}
    assert plot_path.read_bytes().startswith(b"%PDF")
