r"""
Equivariance benchmark: mean relative equivariance errors of s2conv's operators.

The protocol follows Cobb et al. (2021) and Ocampo et al. (2023): random band-limited signals
with standard normal harmonic coefficients, Haar-random rotations, and for each operator the
relative error :math:`\lVert T(\mathcal{R}f) - \mathcal{R}(Tf)\rVert` normalised by
:math:`\lVert\mathcal{R}(Tf)\rVert` (reported first) and by :math:`\lVert T(\mathcal{R}f)\rVert`
(Cobb et al.), averaged over all pairs; see :mod:`s2conv.utils.equivariance`.  For each operator
the script also reports a reduction error, the largest relative error between its closed form
and its form as a lifted convolution, and the run time of the operator end to end, with its
transforms.  Run it from the repository root, for example::

    python benchmarks/equivariance.py -L 16 32 64 -o equivariance.json --latex equivariance.tex
    python benchmarks/equivariance.py -L 16 --precision single

It prints a markdown table per band-limit, and optionally writes the results with system
information as JSON and a LaTeX table with the columns operator, reduction error, equivariance
error and time.  Every output records the git commit of the code that ran, s2conv's and, when
it is installed from a checkout, s2fft's, and flags uncommitted changes, so commit before a run
whose results are to be kept.
"""

import argparse
import datetime
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

N_DIR = 5  # azimuthal band-limit of the directional kernels and of the outputs on SO(3)


@dataclass
class Spec:
    """An operator at one band-limit: its inputs, errors, reduction check and timed pipeline."""

    inputs: dict  # keyword arguments of random_signals: {"spin": s} or {"N": N}
    errors: Callable  # (signals, rotations) -> EquivarianceErrors
    reduction: Callable | None  # (signals) -> list of (closed form, lifted form) pairs
    timed: Callable  # samples -> samples, end to end with the transforms


@dataclass
class Case:
    """An operator of the benchmark."""

    name: str
    label: str  # LaTeX
    domain: str
    group: str
    equivariant: bool
    build: Callable  # (L, key) -> Spec


def catalogue():
    """The operators of the benchmark, built lazily so that the precision is set first."""
    import jax.numpy as jnp
    import numpy as np

    import s2conv
    from s2conv import kernels, lifting, transforms
    from s2conv.convolutions import classical, diagonal, lifted, physics, section
    from s2conv.utils import equivariance as eq
    from s2conv.utils.indexing import conjugate, m_axis

    def random_kernel(key, L, N_k, N_n=None):
        """A random kernel with orders |k| < N_k and, if given, |n| < N_n."""
        kernel = eq.random_signals(key, 1, L, N=N_k)[0]
        if N_n is not None:
            kernel = kernel * (np.abs(m_axis(L)) < N_n)
        return kernel

    def random_field(key, L, spin=0, orders=None):
        """A random spin field, keeping the orders |m| < orders if given."""
        field = eq.random_signals(key, 1, L, spin=spin)[0]
        return field if orders is None else field * (np.abs(m_axis(L)) < orders)

    def on_samples(op, L, s_in, s_out):
        """Op between coefficients, wrapped by forward and inverse transforms of samples."""

        def forward(x):
            return (
                transforms.wigner_forward(x, L, N_DIR)
                if s_in == "G"
                else transforms.forward(x, L, s_in)
            )

        def inverse(y):
            return (
                transforms.wigner_inverse(y, L, N_DIR)
                if s_out == "G"
                else transforms.inverse(y, L, s_out)
            )

        return lambda x: inverse(op(forward(x)))

    def harmonic(op, inputs, L, s_in, s_out, reduction=None):
        return Spec(
            inputs=inputs,
            errors=lambda f, r: eq.equivariance_errors(op, f, r),
            reduction=reduction,
            timed=on_samples(op, L, s_in, s_out),
        )

    def definition_v1(x, kernel, s_in, s_out, N_out=None):
        """Definition V.1 composed from the lift, the group convolution and the projection."""
        F = x if s_in == "G" else lifting.lift(x, s_in)
        if s_out == "G":
            return lifted.group_convolution(F, kernel, N_out)
        return lifting.project(
            lifted.group_convolution(F, kernel, abs(s_out) + 1), s_out
        )

    def lifted_case(s_in, s_out):
        def build(L, key):
            N_k = N_DIR if s_in == "G" else abs(s_in) + 1
            kernel = random_kernel(key, L, N_k, N_DIR if s_out == "G" else None)
            N_out = N_DIR if s_out == "G" else None

            def op(x):
                return s2conv.lifted_convolution(x, kernel, s_in, s_out, N_out)

            def reduction(f):
                if s_in == "G" and s_out == "G":
                    # Eq. (46) written out, (Psi * F)^l_{mn} = sum_k F^l_{mk} Psi^{l*}_{nk}, since the
                    # composition of Definition V.1 is the group convolution itself here
                    block = jnp.conj(kernel[:, :, L - N_DIR : L + N_DIR - 1])
                    return [(op(x), jnp.einsum("klm,kln->nlm", x, block)) for x in f]
                return [
                    (op(x), definition_v1(x, kernel, s_in, s_out, N_out)) for x in f
                ]

            inputs = {"N": N_DIR} if s_in == "G" else {"spin": s_in}
            return harmonic(op, inputs, L, s_in, s_out, reduction)

        return build

    def axisymmetric(L, key):
        psi = random_field(key, L, orders=1)
        kernel = kernels.lifted(psi, 0)

        def op(x):
            return classical.axisymmetric_convolution(x, psi)

        def reduction(f):
            return [
                (op(x), s2conv.lifted_convolution(x, kernel, 0, 0) / np.sqrt(2 * np.pi))
                for x in f
            ]

        return harmonic(op, {"spin": 0}, L, 0, 0, reduction)

    def left(L, key):
        h = random_field(key, L)
        kernel = kernels.left(h)

        def op(x):
            return classical.left_convolution(x, h)

        return harmonic(
            op,
            {"spin": 0},
            L,
            0,
            0,
            lambda f: [(op(x), s2conv.lifted_convolution(x, kernel, 0, 0)) for x in f],
        )

    def directional(L, key):
        psi = random_field(key, L, spin=2, orders=N_DIR)
        kernel = kernels.lifted(psi, 2)

        def op(x):
            return classical.directional_convolution(x, psi, 2, N_DIR)

        def reduction(f):
            return [
                (op(x), s2conv.lifted_convolution(x, kernel, 2, "G", N_DIR)) for x in f
            ]

        return harmonic(op, {"spin": 2}, L, 2, "G", reduction)

    def synthesis(L, key):
        psi = random_field(key, L, spin=2, orders=N_DIR)
        kernel = kernels.flipped(kernels.lifted(psi, 2))

        def op(x):
            return classical.directional_synthesis(x, psi, 2)

        def reduction(f):
            return [(op(x), s2conv.lifted_convolution(x, kernel, "G", 2)) for x in f]

        return harmonic(op, {"N": N_DIR}, L, "G", 2, reduction)

    def multiplier_case(fn, multiplier, s_in, s_out):
        def build(L, key):
            kernel = kernels.multiplier(multiplier(L), s_in, s_out)

            def op(x):
                return fn(x)

            def reduction(f):
                return [
                    (op(x), s2conv.lifted_convolution(x, kernel, s_in, s_out))
                    for x in f
                ]

            return harmonic(op, {"spin": s_in}, L, s_in, s_out, reduction)

        return build

    def e_mode(L, key):
        half = -0.5 * np.ones(L)
        k_plus, k_minus = (
            kernels.multiplier(half, 2, 0),
            kernels.multiplier(half, -2, 0),
        )

        def op(x):
            return physics.eb_decomposition(x)[0]

        def reduction(f):
            return [
                (
                    op(x),
                    s2conv.lifted_convolution(x, k_plus, 2, 0)
                    + s2conv.lifted_convolution(conjugate(x, 2), k_minus, -2, 0),
                )
                for x in f
            ]

        return harmonic(op, {"spin": 2}, L, 2, 0, reduction)

    def standard_correlation(L, key):
        psi = random_field(key, L, orders=N_DIR)
        kernel = kernels.lifted(psi, 0)

        def reduction(f):
            return [
                (
                    section.standard_correlation(x, psi, N_DIR),
                    transforms.wigner_inverse(
                        s2conv.lifted_convolution(x, kernel, 0, "G", N_DIR), L, N_DIR
                    )[0],
                )
                for x in f
            ]

        return Spec(
            inputs={"spin": 0},
            errors=lambda f, r: eq.equivariance_errors_of_section(
                kernel, 0, N_DIR, f, r
            ),
            reduction=reduction,
            timed=lambda x: section.standard_correlation(
                transforms.forward(x, L), psi, N_DIR
            ),
        )

    def commutative_anisotropic(L, key):
        h = random_field(key, L, orders=N_DIR)
        kernel = kernels.lifted(conjugate(h, 0), 0)

        def chi(theta, phi):
            return jnp.pi - phi + 0.0 * theta

        return Spec(
            inputs={"spin": 0},
            errors=lambda f, r: eq.equivariance_errors_of_section(
                kernel, 0, N_DIR, f, r, chi=chi
            ),
            reduction=None,
            timed=lambda x: section.commutative_anisotropic_convolution(
                transforms.forward(x, L), h, N_DIR
            ),
        )

    def diagonal_case(fn):
        def build(L, key):
            g = random_field(key, L)

            def op(x):
                return fn(x, g)

            return harmonic(op, {"spin": 0}, L, 0, 0)

        return build

    return [
        Case(
            "spin_0_0",
            r"$\mathcal{C}_{0\to0}[\Psi]$",
            "S2, s=0 -> 0",
            "lifted",
            True,
            lifted_case(0, 0),
        ),
        Case(
            "spin_2_1",
            r"$\mathcal{C}_{2\to1}[\Psi]$",
            "S2, s=2 -> 1",
            "lifted",
            True,
            lifted_case(2, 1),
        ),
        Case(
            "spin_-1_2",
            r"$\mathcal{C}_{-1\to2}[\Psi]$",
            "S2, s=-1 -> 2",
            "lifted",
            True,
            lifted_case(-1, 2),
        ),
        Case(
            "spin_0_G",
            r"$\mathcal{C}_{0\to G}[\Psi]$",
            f"S2, s=0 -> SO(3), N={N_DIR}",
            "lifted",
            True,
            lifted_case(0, "G"),
        ),
        Case(
            "spin_2_G",
            r"$\mathcal{C}_{2\to G}[\Psi]$",
            f"S2, s=2 -> SO(3), N={N_DIR}",
            "lifted",
            True,
            lifted_case(2, "G"),
        ),
        Case(
            "group_0",
            r"$\mathcal{C}_{G\to0}[\Psi]$",
            f"SO(3), N={N_DIR} -> S2, s=0",
            "lifted",
            True,
            lifted_case("G", 0),
        ),
        Case(
            "group_2",
            r"$\mathcal{C}_{G\to2}[\Psi]$",
            f"SO(3), N={N_DIR} -> S2, s=2",
            "lifted",
            True,
            lifted_case("G", 2),
        ),
        Case(
            "group_G",
            r"$\mathcal{C}_{G\to G}[\Psi]$",
            f"SO(3) -> SO(3), N={N_DIR}",
            "lifted",
            True,
            lifted_case("G", "G"),
        ),
        Case(
            "axisymmetric",
            "axisymmetric convolution",
            "S2, s=0 -> 0",
            "classical",
            True,
            axisymmetric,
        ),
        Case(
            "left",
            "left convolution (Driscoll--Healy)",
            "S2, s=0 -> 0",
            "classical",
            True,
            left,
        ),
        Case(
            "directional",
            "directional convolution",
            f"S2, s=2 -> SO(3), N={N_DIR}",
            "classical",
            True,
            directional,
        ),
        Case(
            "synthesis",
            "directional synthesis",
            f"SO(3), N={N_DIR} -> S2, s=2",
            "classical",
            True,
            synthesis,
        ),
        Case(
            "eth",
            r"$\eth$",
            "S2, s=1 -> 2",
            "physics",
            True,
            multiplier_case(
                lambda x: physics.eth(x, 1),
                lambda L: physics.eth_multiplier(L, 1),
                1,
                2,
            ),
        ),
        Case(
            "ethbar",
            r"$\bar\eth$",
            "S2, s=1 -> 0",
            "physics",
            True,
            multiplier_case(
                lambda x: physics.ethbar(x, 1),
                lambda L: physics.ethbar_multiplier(L, 1),
                1,
                0,
            ),
        ),
        Case(
            "e_mode",
            r"$E$ mode from $Q + \mathrm{i}U$",
            "S2, s=2 -> 0",
            "physics",
            True,
            e_mode,
        ),
        Case(
            "kaiser_squires",
            "Kaiser--Squires",
            "S2, s=0 -> 2",
            "physics",
            True,
            multiplier_case(
                physics.kaiser_squires, physics.kaiser_squires_multiplier, 0, 2
            ),
        ),
        Case(
            "standard_correlation",
            "standard correlation",
            f"S2, s=0 -> S2 (section), N={N_DIR}",
            "not equivariant",
            False,
            standard_correlation,
        ),
        Case(
            "commutative_anisotropic",
            "commutative anisotropic convolution",
            f"S2, s=0 -> S2 (section), N={N_DIR}",
            "not equivariant",
            False,
            commutative_anisotropic,
        ),
        Case(
            "sifting",
            "sifting convolution",
            "S2, s=0 -> 0",
            "not equivariant",
            False,
            diagonal_case(diagonal.sifting_convolution),
        ),
        Case(
            "harmonic_multiplication",
            "harmonic multiplication",
            "S2, s=0 -> 0",
            "not equivariant",
            False,
            diagonal_case(diagonal.harmonic_multiplication),
        ),
    ]


def reduction_error(pairs) -> float:
    """The largest relative error ||a - b|| / ||b|| over pairs of arrays."""
    import numpy as np

    return max(
        float(
            np.linalg.norm(np.asarray(a) - np.asarray(b))
            / np.linalg.norm(np.asarray(b))
        )
        for a, b in pairs
    )


def pipeline_times(spec: Spec, L: int, key, repeats: int) -> list:
    """Run times of the operator end to end, JIT compiled, compilation excluded."""
    import jax

    from s2conv import transforms
    from s2conv.utils import equivariance as eq

    x = eq.random_signals(key, 1, L, **spec.inputs)[0]
    if "N" in spec.inputs:
        samples = transforms.wigner_inverse(x, L, spec.inputs["N"])
    else:
        samples = transforms.inverse(x, L, spec.inputs["spin"])
    run = jax.jit(spec.timed)
    jax.block_until_ready(run(samples))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(run(samples))
        times.append(time.perf_counter() - start)
    return times


def median(times: list) -> float:
    """The median of the run times, the upper one of the middle two for an even number."""
    return sorted(times)[len(times) // 2]


def time_pipeline(spec: Spec, L: int, key, repeats: int) -> float:
    """Median run time of the operator end to end, JIT compiled, compilation excluded."""
    return median(pipeline_times(spec, L, key, repeats))


def system_info() -> dict:
    """Versions and devices, as recorded by the s2fft benchmarks."""
    from importlib import metadata

    import jax
    import numpy as np

    import s2conv

    try:
        s2fft_version = metadata.version("s2fft")
    except metadata.PackageNotFoundError:
        s2fft_version = "unknown"
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": jax.__version__,
        "s2fft": s2fft_version,
        "s2conv": s2conv.__version__,
        "devices": [str(device) for device in jax.devices()],
        "x64": bool(jax.config.read("jax_enable_x64")),
    }


def git_revision(file) -> dict | None:
    """The commit of the git checkout that tracks file and its uncommitted changes, or None if there is none."""
    import subprocess
    from pathlib import Path

    path = Path(file).resolve()

    def git(*args):
        command = ["git", "-C", str(path.parent), "--no-optional-locks", *args]
        return subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        # tracked, not merely inside a checkout, as a package in a virtual environment there is
        git("ls-files", "--error-unmatch", path.name)
        modified = git("diff", "--name-only", "HEAD").splitlines()
        return {
            "repository": git("rev-parse", "--show-toplevel"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(modified),
            "modified": modified,
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def code_revisions(script: str = __file__) -> dict:
    """The git revisions of the code that runs: the s2conv package, the benchmark script and s2fft."""
    import s2fft

    import s2conv

    return {
        "s2conv": git_revision(s2conv.__file__),
        "benchmark": git_revision(script),
        "s2fft": git_revision(s2fft.__file__),
    }


def checkouts(code: dict) -> list:
    """The distinct git checkouts of code_revisions as (name, revision) pairs."""
    names = {"s2conv": "s2conv", "benchmark": "benchmark script", "s2fft": "s2fft"}
    return [
        (names[key], revision)
        for key, revision in code.items()
        if revision is not None
        and not (key == "benchmark" and revision == code["s2conv"])
    ]


def code_summary(code: dict) -> str:
    """The commits of the code that runs, in one line."""
    parts = [
        f"{name} commit {revision['commit']}"
        + (" with uncommitted changes" if revision["dirty"] else "")
        for name, revision in checkouts(code)
    ]
    return ", ".join(parts) or "no git checkouts"


def warn_about_code(code: dict) -> None:
    """Warn if the results will not identify the code that runs by its commits alone."""
    s2conv_revision, script = code["s2conv"], code["benchmark"]
    if s2conv_revision is None:
        print(
            "warning: s2conv is not imported from a git checkout; the results record its version only",
            file=sys.stderr,
        )
    elif script is not None and script["repository"] != s2conv_revision["repository"]:
        print(
            f"warning: s2conv is imported from {s2conv_revision['repository']}, "
            f"not from the checkout of this script, {script['repository']}",
            file=sys.stderr,
        )
    if script is None:
        print(
            "warning: the benchmark script is not tracked by git; the results do not record "
            "its code",
            file=sys.stderr,
        )
    for name, revision in checkouts(code):
        if revision["dirty"]:
            print(
                f"warning: the {name} checkout {revision['repository']} has uncommitted changes "
                f"({', '.join(revision['modified'])}); the results record commit {revision['commit']} "
                "and flag the changes",
                file=sys.stderr,
            )


def scientific(value: float, latex: bool = False) -> str:
    """A value to two significant figures, in LaTeX notation if asked."""
    if value is None:
        return "--" if latex else "-"
    if value == 0:
        return "$0$" if latex else "0"
    text = f"{value:.1e}"
    if not latex:
        return text
    mantissa, exponent = text.split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def markdown_table(results: list, timings: dict, time_L: int) -> str:
    """One markdown table of the results at one band-limit."""
    lines = [
        "| Operator | Domain | Mean error, norm. by R(Tf) | Mean error, norm. by T(Rf) | Max | Reduction | "
        + (f"Time (L = {time_L}) |" if time_L else "Time |"),
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        main, cobb = (
            r["errors"]["rotated_output"],
            r["errors"]["output_of_rotated_input"],
        )
        seconds = timings.get(r["name"])
        lines.append(
            f"| {r['name']} | {r['domain']} | {main['mean']:.2e} ± {main['std']:.1e} | {cobb['mean']:.2e} | "
            f"{main['max']:.1e} | {scientific(r['reduction'])} | {'-' if seconds is None else f'{1e3 * seconds:.2f} ms'} |"
        )
    return "\n".join(lines)


def latex_table(
    results: list, timings: dict, L: int, time_L: int, args, provenance: list = ()
) -> str:
    """
    The results at one band-limit as a LaTeX table: operator, reduction and equivariance errors, time.

    The lines of provenance, such as the commits of the code, head the table as comments.
    """
    lines = [f"% {line}" for line in provenance] + [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Operator & Reduction error & Equivariance error & "
        + (rf"Time ($L={time_L}$) \\" if time_L else r"Time \\"),
        r"\midrule",
    ]
    for r in results:
        seconds = timings.get(r["name"])
        time_text = "--" if seconds is None else rf"{1e3 * seconds:.2f}\,ms"
        lines.append(
            rf"{r['label']} & {scientific(r['reduction'], True)} & "
            rf"{scientific(r['errors']['rotated_output']['mean'], True)} & {time_text} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            rf"% Equivariance errors at L = {L}: mean over {args.signals} random signals and {args.rotations} "
            rf"Haar-random rotations of ||T(Rf) - R(Tf)|| / ||R(Tf)||, {args.precision} precision"
            rf" ({args.matmul_precision} matrix-product precision)."
        ),
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    """Command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-L",
        "--bandlimits",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="band-limits (default 16 32 64)",
    )
    parser.add_argument(
        "--signals",
        type=int,
        default=100,
        help="random signals per operator (default 100)",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=100,
        help="random rotations per operator (default 100)",
    )
    parser.add_argument(
        "--precision",
        choices=("double", "single"),
        default="double",
        help="floating-point precision",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "default"),
        default="highest",
        help=(
            "precision of single-precision matrix products: 'highest' computes them in float32; "
            "'default' lets JAX use TF32 on NVIDIA GPUs, which limits the errors to about 1e-4"
        ),
    )
    parser.add_argument(
        "--operators",
        nargs="*",
        default=None,
        help="names or groups of operators to run (default all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--reduction-signals",
        type=int,
        default=10,
        help="signals for the reduction errors (default 10)",
    )
    parser.add_argument(
        "--time-L",
        type=int,
        default=128,
        help="band-limit of the timings; 0 skips them (default 128)",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="timed runs per operator (default 5)"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="write the results as JSON to this file"
    )
    parser.add_argument(
        "--latex",
        default=None,
        help="write a LaTeX table (at the largest band-limit) to this file",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the operators and exit"
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the benchmark and report the results."""
    args = parse_args(argv)
    import jax

    jax.config.update("jax_enable_x64", args.precision == "double")
    if args.matmul_precision == "highest":
        # otherwise JAX computes float32 matrix products in TF32 on NVIDIA GPUs
        jax.config.update("jax_default_matmul_precision", "highest")
    from s2conv.utils import equivariance as eq

    cases = catalogue()
    if args.list:
        for case in cases:
            print(f"{case.name:26s} {case.group:16s} {case.domain}")
        return
    if args.operators:
        cases = [
            c for c in cases if c.name in args.operators or c.group in args.operators
        ]
    date_time = datetime.datetime.now().isoformat(timespec="seconds")
    system, code = system_info(), code_revisions()
    warn_about_code(code)
    provenance = [
        f"s2conv equivariance benchmark, {date_time}, JAX {system['jax']}, "
        f"s2fft {system['s2fft']}, {', '.join(system['devices'])}.",
        f"Code: {code_summary(code)}.",
    ]
    print("\n".join(provenance))
    key = jax.random.key(args.seed)

    timings = {}
    if args.time_L:
        for i, case in enumerate(cases):
            build_key, time_key = jax.random.split(jax.random.fold_in(key, 1000 + i))
            timings[case.name] = time_pipeline(
                case.build(args.time_L, build_key), args.time_L, time_key, args.repeats
            )

    results = []
    for L in args.bandlimits:
        rows = []
        for i, case in enumerate(cases):
            build_key, signal_key, rotation_key = jax.random.split(
                jax.random.fold_in(key, i), 3
            )
            spec = case.build(L, build_key)
            signals = eq.random_signals(signal_key, args.signals, L, **spec.inputs)
            rotations = eq.random_rotations(rotation_key, args.rotations)
            start = time.perf_counter()
            errors = spec.errors(signals, rotations)
            jax.block_until_ready(errors)
            seconds = time.perf_counter() - start
            reduction = None
            if spec.reduction is not None:
                reduction = reduction_error(
                    spec.reduction(signals[: args.reduction_signals])
                )
            rows.append(
                {
                    "name": case.name,
                    "label": case.label,
                    "domain": case.domain,
                    "group": case.group,
                    "equivariant": case.equivariant,
                    "L": L,
                    "errors": eq.summarise(errors),
                    "reduction": reduction,
                    "seconds": seconds,
                }
            )
            print(
                f"L = {L:3d}  {case.name:26s} done in {seconds:6.1f} s", file=sys.stderr
            )
        print(
            f"\n### L = {L}, {args.signals} signals x {args.rotations} rotations, {args.precision} precision\n"
        )
        print(markdown_table(rows, timings, args.time_L))
        results += rows

    if args.output:
        record = {
            "date_time": date_time,
            "command": " ".join(sys.argv),
            "arguments": vars(args),
            "system": system,
            "code": code,
            "results": results,
            "timings": {"L": args.time_L, "repeats": args.repeats, "seconds": timings},
        }
        with open(args.output, "w") as file:
            json.dump(record, file, indent=2)
    if args.latex:
        largest = max(args.bandlimits)
        with open(args.latex, "w") as file:
            file.write(
                latex_table(
                    [r for r in results if r["L"] == largest],
                    timings,
                    largest,
                    args.time_L,
                    args,
                    provenance,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
