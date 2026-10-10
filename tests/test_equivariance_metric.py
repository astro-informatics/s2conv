"""
The equivariance-error protocol of s2conv.utils.equivariance, and the paper-scale benchmark built on it.

The fast tests check the metric itself: its random signals and rotations, exact zero and agreement of the two
normalisations for equivariant maps, agreement between the harmonic, point and section paths, and the Parseval norms,
and the git commits that the benchmark records.  The slow tests run every operator of benchmarks/equivariance.py at
20 signals x 20 rotations.
"""

import os
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import s2conv
from s2conv import kernels, transforms
from s2conv.utils import equivariance as eq
from s2conv.utils.evaluation import evaluate
from s2conv.utils.indexing import lm_mask, m_axis, wigner_mask

L = 8
N = 3


def _keys(rng, n):
    return jax.random.split(jax.random.key(int(rng.integers(2**31))), n)


def test_random_signals_support(rng):
    key = _keys(rng, 1)[0]
    f = eq.random_signals(key, 3, L, spin=2)
    assert f.shape == (3, L, 2 * L - 1)
    assert np.all(np.asarray(f)[:, ~lm_mask(L, 2)] == 0)
    assert np.all(np.asarray(f)[:, lm_mask(L, 2)] != 0)
    F = eq.random_signals(key, 3, L, N=N)
    assert F.shape == (3, 2 * N - 1, L, 2 * L - 1)
    assert np.all(np.asarray(F)[:, ~wigner_mask(L, N)] == 0)


def test_random_signals_are_standard_normal(rng):
    f = np.asarray(eq.random_signals(_keys(rng, 1)[0], 400, L))[:, lm_mask(L)]
    assert abs(np.mean(f.real)) < 0.05 and abs(np.mean(f.imag)) < 0.05
    assert abs(np.var(f.real) - 1) < 0.05 and abs(np.var(f.imag) - 1) < 0.05


def test_random_rotations_are_haar(rng):
    alpha, beta, gamma = (
        np.asarray(a) for a in eq.random_rotations(_keys(rng, 1)[0], 20000)
    )
    assert alpha.shape == beta.shape == gamma.shape == (20000,)
    assert np.all((0 <= alpha) & (alpha < 2 * np.pi)) and np.all(
        (0 <= gamma) & (gamma < 2 * np.pi)
    )
    # Haar measure: alpha and gamma uniform, cos(beta) uniform on [-1, 1]
    assert (
        abs(np.mean(np.cos(beta))) < 0.02
        and abs(np.mean(np.cos(beta) ** 2) - 1 / 3) < 0.02
    )
    assert abs(np.mean(alpha) - np.pi) < 0.05 and abs(np.mean(gamma) - np.pi) < 0.05


def test_identity_has_zero_error(rng):
    k1, k2 = _keys(rng, 2)
    errors = eq.equivariance_errors(
        lambda x: x, eq.random_signals(k1, 3, L, spin=1), eq.random_rotations(k2, 4)
    )
    assert errors.rotated_output.shape == (3, 4)
    assert float(jnp.max(errors.rotated_output)) < 1e-14
    assert float(jnp.max(errors.output_of_rotated_input)) < 1e-14


@pytest.mark.parametrize(
    ("s_in", "s_out"), [(0, 0), (2, 1), ("G", "G"), (1, "G"), ("G", -2)]
)
def test_equivariant_operator(rng, s_in, s_out):
    k1, k2, k3 = _keys(rng, 3)
    N_in = N if s_in == "G" else abs(s_in) + 1
    kernel = eq.random_signals(k1, 1, L, N=N_in)[0]
    signals = eq.random_signals(
        k2, 4, L, **({"N": N} if s_in == "G" else {"spin": s_in})
    )
    N_out = N if s_out == "G" else None
    errors = eq.equivariance_errors(
        lambda x: s2conv.lifted_convolution(x, kernel, s_in, s_out, N_out),
        signals,
        eq.random_rotations(k3, 5),
    )
    assert float(jnp.max(errors.rotated_output)) < 1e-13
    # the two normalisations agree to relative order of the error itself
    np.testing.assert_allclose(
        errors.rotated_output, errors.output_of_rotated_input, rtol=1e-12, atol=1e-20
    )


def test_non_equivariant_operator(rng):
    k1, k2, k3 = _keys(rng, 3)
    g = eq.random_signals(k1, 1, L)[0]
    errors = eq.equivariance_errors(
        lambda x: s2conv.diagonal.sifting_convolution(x, g),
        eq.random_signals(k2, 4, L),
        eq.random_rotations(k3, 5),
    )
    assert float(jnp.mean(errors.rotated_output)) > 0.5
    assert float(jnp.mean(errors.output_of_rotated_input)) > 0.5


def test_batch_size_does_not_change_results(rng):
    k1, k2, k3 = _keys(rng, 3)
    g = eq.random_signals(k1, 1, L)[0]
    signals, rotations = eq.random_signals(k2, 2, L), eq.random_rotations(k3, 7)

    def op(x):
        return s2conv.diagonal.sifting_convolution(x, g)

    whole = eq.equivariance_errors(op, signals, rotations, batch_size=100)
    split = eq.equivariance_errors(op, signals, rotations, batch_size=3)
    np.testing.assert_allclose(whole.rotated_output, split.rotated_output, rtol=1e-14)


def test_points_path_matches_harmonic_path(rng):
    """For band-limited outputs the quadrature of the points path is exact, so the two paths agree."""
    k1, k2, k3 = _keys(rng, 3)
    g = eq.random_signals(k1, 1, L)[0]
    signals, rotations = eq.random_signals(k2, 3, L), eq.random_rotations(k3, 4)
    harmonic = eq.equivariance_errors(
        lambda x: s2conv.diagonal.sifting_convolution(x, g), signals, rotations
    )
    points = eq.equivariance_errors_on_points(
        lambda x: (
            lambda t, p: evaluate(s2conv.diagonal.sifting_convolution(x, g), 0, t, p)
        ),
        signals,
        rotations,
    )
    np.testing.assert_allclose(
        points.rotated_output, harmonic.rotated_output, rtol=1e-11
    )
    np.testing.assert_allclose(
        points.output_of_rotated_input, harmonic.output_of_rotated_input, rtol=1e-11
    )


def test_points_path_spin_output(rng):
    k1, k2, k3 = _keys(rng, 3)
    kernel = kernels.multiplier(np.arange(L) + 1.0, 2, -1)
    errors = eq.equivariance_errors_on_points(
        lambda x: (
            lambda t, p: evaluate(s2conv.lifted_convolution(x, kernel, 2, -1), -1, t, p)
        ),
        eq.random_signals(k2, 3, L, spin=2),
        eq.random_rotations(k3, 4),
        spin_out=-1,
    )
    assert float(jnp.max(errors.rotated_output)) < 1e-13


def _section_evaluator(kernel, N_sec, chi=None):
    def evaluator(x):
        comps = s2conv.section.section_components(x, kernel, 0, N_sec)

        def at(theta, phi):
            total = 0.0
            for sp, comp in zip(range(-(N_sec - 1), N_sec), comps, strict=False):
                values = evaluate(comp, sp, theta, phi)
                total = total + (
                    values
                    if chi is None
                    else jnp.exp(-1j * sp * chi(theta, phi)) * values
                )
            return total

        return at

    return evaluator


@pytest.mark.parametrize("sadeghi", [False, True])
def test_section_path_matches_points_path(rng, sadeghi):
    k1, k2, k3 = _keys(rng, 3)
    psi = eq.random_signals(k1, 1, L)[0] * (np.abs(m_axis(L)) < N)
    chi = (lambda t, p: jnp.pi - p + 0.0 * t) if sadeghi else None
    kernel = kernels.lifted(s2conv.utils.conjugate(psi, 0) if sadeghi else psi, 0)
    signals, rotations = eq.random_signals(k2, 2, L), eq.random_rotations(k3, 3)
    section = eq.equivariance_errors_of_section(
        kernel, 0, N, signals, rotations, chi=chi, L_quad=2 * L
    )
    points = eq.equivariance_errors_on_points(
        _section_evaluator(kernel, N, chi), signals, rotations, L_quad=2 * L
    )
    np.testing.assert_allclose(
        section.rotated_output, points.rotated_output, rtol=1e-11
    )
    assert float(jnp.mean(section.rotated_output)) > 0.5


def test_section_is_equivariant_under_polar_rotations(rng):
    k1, k2 = _keys(rng, 2)
    psi = eq.random_signals(k1, 1, L)[0] * (np.abs(m_axis(L)) < N)
    polar = (jnp.array([0.4, 1.1, 2.5]), jnp.zeros(3), jnp.zeros(3))
    errors = eq.equivariance_errors_of_section(
        kernels.lifted(psi, 0), 0, N, eq.random_signals(k2, 2, L), polar
    )
    assert float(jnp.max(errors.rotated_output)) < 1e-13


def test_parseval_weights_are_the_l2_norm_on_the_rotation_group(rng):
    """sum_l (2l+1)/(8 pi^2) sum |F^l_mn|^2 equals the integral of |F|^2, by exact Gauss-Legendre quadrature."""
    F = eq.random_signals(_keys(rng, 1)[0], 1, L, N=N)[0]
    parseval = float(jnp.sum(eq._parseval_weights(F.shape) * jnp.abs(F) ** 2))
    samples = np.asarray(transforms.wigner_inverse(F, L, N, sampling="gl"))
    _, _, row_weights = eq._gl_grid(L)
    quadrature = float(
        np.sum(row_weights[None, :, None] * np.abs(samples) ** 2)
        * 2
        * np.pi
        / (2 * N - 1)
    )
    assert abs(parseval - quadrature) < 1e-12 * parseval


def test_parseval_weights_reject_other_shapes():
    with pytest.raises(ValueError, match="harmonic coefficients"):
        eq._parseval_weights((L * L,))


def test_quadrature_is_exact_for_band_limited_fields(rng):
    theta, phi, weights = eq.quadrature(L)
    for spin in (0, 2, -3):
        f = eq.random_signals(_keys(rng, 1)[0], 1, L, spin=spin)[0]
        values = np.asarray(evaluate(f, spin, theta, phi))
        assert abs(
            np.sum(weights * np.abs(values) ** 2) - np.sum(np.abs(f) ** 2)
        ) < 1e-12 * np.sum(np.abs(f) ** 2)


def test_summarise(rng):
    k1, k2 = _keys(rng, 2)
    errors = eq.equivariance_errors(
        lambda x: x, eq.random_signals(k1, 2, L), eq.random_rotations(k2, 3)
    )
    summary = eq.summarise(errors)
    assert set(summary) == {"rotated_output", "output_of_rotated_input"}
    assert set(summary["rotated_output"]) == {"mean", "std", "max"}


# ------------------------------------------------------------------------ paper-scale protocol


def _benchmark():
    equivariance = pytest.importorskip("benchmarks.equivariance")
    return equivariance


def test_git_revision(tmp_path, monkeypatch, capsys):
    if shutil.which("git") is None:
        pytest.skip("requires git")
    bench = _benchmark()
    # a repository of its own, independent of the user's git configuration
    for variable in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        monkeypatch.setenv(variable, os.devnull)
    identity = ["-c", "user.name=s2conv", "-c", "user.email=s2conv@example.com"]

    def git(*args):
        command = ["git", "-C", str(tmp_path), *identity, *args]
        return subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, check=True
        ).stdout.strip()

    module = tmp_path / "module.py"
    module.write_text("x = 1\n")
    git("init", "-q")
    assert bench.git_revision(module) is None  # untracked
    git("add", "module.py")
    git("commit", "-q", "-m", "Add a module")
    revision = bench.git_revision(module)
    assert revision["commit"] == git("rev-parse", "HEAD")
    assert os.path.samefile(revision["repository"], tmp_path)
    assert not revision["dirty"] and revision["modified"] == []
    (tmp_path / "notes.txt").write_text("untracked files are not changes to the code\n")
    assert not bench.git_revision(module)["dirty"]
    module.write_text("x = 2\n")
    revision = bench.git_revision(module)
    assert revision["dirty"] and revision["modified"] == ["module.py"]
    code = {"s2conv": revision, "benchmark": revision, "s2fft": None}
    assert bench.code_summary(code) == (
        f"s2conv commit {revision['commit']} with uncommitted changes"
    )
    capsys.readouterr()
    bench.warn_about_code(code)
    assert "uncommitted changes (module.py)" in capsys.readouterr().err
    # a script run against s2conv from another checkout
    code["s2conv"] = dict(revision, repository="elsewhere", dirty=False, modified=[])
    bench.warn_about_code(code)
    assert "not from the checkout of this script" in capsys.readouterr().err
    assert "benchmark script commit" in bench.code_summary(code)


@pytest.mark.slow
@pytest.mark.parametrize("index", range(20))
def test_benchmark_operators(index):
    """Every operator of the benchmark at 20 signals x 20 rotations, L = 16, as in the paper protocol."""
    bench = _benchmark()
    cases = bench.catalogue()
    if index >= len(cases):
        pytest.skip("no such operator")
    case = cases[index]
    key = jax.random.fold_in(jax.random.key(2026), index)
    build_key, signal_key, rotation_key = jax.random.split(key, 3)
    spec = case.build(16, build_key)
    signals = eq.random_signals(signal_key, 20, 16, **spec.inputs)
    summary = eq.summarise(spec.errors(signals, eq.random_rotations(rotation_key, 20)))
    if case.equivariant:
        assert summary["rotated_output"]["max"] < 1e-12, case.name
        assert summary["output_of_rotated_input"]["max"] < 1e-12, case.name
    else:
        assert summary["rotated_output"]["mean"] > 0.1, case.name
    if spec.reduction is not None:
        assert bench.reduction_error(spec.reduction(signals[:3])) < 1e-12, case.name


@pytest.mark.slow
def test_benchmark_command_line(tmp_path):
    bench = _benchmark()
    json_path, tex_path = tmp_path / "results.json", tmp_path / "table.tex"
    bench.main(
        [
            "-L",
            "6",
            "--signals",
            "2",
            "--rotations",
            "2",
            "--time-L",
            "8",
            "--repeats",
            "1",
            "--operators",
            "lifted",
            "sifting",
            "-o",
            str(json_path),
            "--latex",
            str(tex_path),
        ]
    )
    import json

    record = json.loads(json_path.read_text())
    assert {r["name"] for r in record["results"]} >= {"spin_0_0", "group_G", "sifting"}
    assert set(record["timings"]["seconds"]) == {r["name"] for r in record["results"]}
    table = tex_path.read_text()
    assert r"\begin{tabular}" in table
    # the commit of the code that ran, in both files, when it runs from a git checkout
    assert set(record["code"]) == {"s2conv", "benchmark", "s2fft"}
    assert record["code"]["s2conv"] == bench.git_revision(s2conv.__file__)
    assert f"% Code: {bench.code_summary(record['code'])}." in table
    if record["code"]["s2conv"] is not None:
        assert record["code"]["s2conv"]["commit"] in table
