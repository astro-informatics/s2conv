r"""
Scaling benchmark: computation time of the four lifted convolutions against band-limit.

The script times one operator of each of the four kinds, :math:`\mathcal{C}_{2\to1}`,
:math:`\mathcal{C}_{2\to G}`, :math:`\mathcal{C}_{G\to2}` and :math:`\mathcal{C}_{G\to G}`, from the
catalogue of benchmarks/equivariance.py, over a range of band-limits.  As in the time column of
that benchmark, each operator runs end to end, with its forward and inverse transforms, JIT
compiled and with compilation excluded, and its time is the median of several runs.  The
azimuthal band-limit of the fields and kernels on the rotation group is fixed, N = 5, so that
every operator costs O(L^3) (Table III of the article).  The script fits a power law to the times
from a given band-limit up, and plots the times, and the times divided by L^3, against
band-limit.

By default it runs on one CPU core, the fastest available, with the multithreading of XLA off,
so that the time counts the operations.  On a GPU, or on many cores, the time grows more slowly
than L^3 until the band-limit is large enough to occupy the hardware.  Run it from the
repository root, for example::

    python benchmarks/scaling.py -o scaling.json --plot scaling.pdf > scaling.md

It prints the provenance of the run and a markdown table of the times and fitted exponents, and
optionally writes them, with every run time and the system information, as JSON, and the plot
as PDF, which needs matplotlib.  ``--replot scaling.json --plot scaling.pdf`` redraws the plot of a
recorded run without timing again.
"""

import argparse
import datetime
import json
import os
import sys

# the equivariance benchmark beside this file, not another package named benchmarks, such as s2fft's
if __package__:  # imported as benchmarks.scaling, as by the tests
    from . import equivariance as common
else:  # run as a script, python benchmarks/scaling.py
    import equivariance as common

N = common.N_DIR  # azimuthal band-limit of the operators on the rotation group

# one operator of each kind, from the catalogue of benchmarks/equivariance.py, with its label
OPERATORS = {
    "spin_2_1": r"$\mathcal{C}_{2\to1}$",
    "spin_2_G": rf"$\mathcal{{C}}_{{2\to G}}$, $N={N}$",
    "group_2": rf"$\mathcal{{C}}_{{G\to2}}$, $N={N}$",
    "group_G": rf"$\mathcal{{C}}_{{G\to G}}$, $N={N}$",
}

BANDLIMITS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def max_frequency(core: int) -> int | None:
    """The maximum clock frequency of a CPU core in MHz, where Linux reports it."""
    try:
        with open(
            f"/sys/devices/system/cpu/cpu{core}/cpufreq/cpuinfo_max_freq"
        ) as file:
            return int(file.read()) // 1000
    except (OSError, ValueError):
        return None


def fastest_core() -> int | None:
    """The available CPU core with the highest maximum frequency, or None if unknown."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    return max(
        sorted(os.sched_getaffinity(0)), key=lambda core: max_frequency(core) or 0
    )


def use_one_core(core: int | None) -> dict:
    """
    Confine the run to one CPU core, with the multithreading of XLA off.

    It must be called before JAX starts, as the threads of JAX inherit the affinity of the
    thread that starts them.  It returns a description of the hardware for the results.
    """
    if "jax" in sys.modules:
        print(
            "warning: JAX has already started, so the benchmark runs on its devices, unconfined",
            file=sys.stderr,
        )
        return {"kind": "default", "core": None}
    core = fastest_core() if core is None else core
    os.environ["JAX_PLATFORMS"] = "cpu"
    flags = [os.environ.get("XLA_FLAGS", ""), "--xla_cpu_multi_thread_eigen=false"]
    os.environ["XLA_FLAGS"] = " ".join(flag for flag in flags if flag)
    if core is None:
        print(
            "warning: this system cannot confine a process to one core, so the benchmark "
            "runs on all the cores of the CPU",
            file=sys.stderr,
        )
        return {"kind": "cpu", "core": None, "xla_flags": os.environ["XLA_FLAGS"]}
    os.sched_setaffinity(0, {core})
    return {
        "kind": "cpu-core",
        "core": core,
        "max_mhz": max_frequency(core),
        "xla_flags": os.environ["XLA_FLAGS"],
    }


def describe(hardware: dict, system: dict) -> str:
    """The hardware of a run, in words."""
    if hardware["kind"] == "cpu-core":
        clock = f", {hardware['max_mhz']} MHz" if hardware["max_mhz"] else ""
        return f"one CPU core (core {hardware['core']}{clock})"
    return ", ".join(system["devices"])


def fit_power_law(bandlimits: list, seconds: list, L_min: int) -> tuple | None:
    """
    Exponent and prefactor of a power law fitted to the times from band-limit L_min up.

    The fit is by least squares of the logarithm of the time against the logarithm of the
    band-limit.  It returns None if fewer than two band-limits are at least L_min.
    """
    import numpy as np

    pairs = [(L, t) for L, t in zip(bandlimits, seconds, strict=True) if L >= L_min]
    if len(pairs) < 2:
        return None
    log_L, log_t = np.log(np.array(pairs, dtype=float)).T
    exponent, log_prefactor = np.polyfit(log_L, log_t, 1)
    return float(exponent), float(np.exp(log_prefactor))


def duration(seconds: float) -> str:
    """A run time to three significant figures, in milliseconds below a second."""
    return f"{1e3 * seconds:.3g} ms" if seconds < 1 else f"{seconds:.3g} s"


def markdown_table(
    medians: dict, bandlimits: list, exponents: dict, fit_from: int
) -> str:
    """The median times against band-limit, with the fitted exponents, as a markdown table."""
    names = list(medians)
    lines = ["| L | " + " | ".join(names) + " |", "|--:|" + "--:|" * len(names)]
    for L in bandlimits:
        times = " | ".join(duration(medians[name][L]) for name in names)
        lines.append(f"| {L} | {times} |")
    fitted = " | ".join(
        "-" if exponents[name] is None else f"{exponents[name][0]:.2f}"
        for name in names
    )
    lines.append(f"| Exponent, L ≥ {fit_from} | {fitted} |")
    return "\n".join(lines)


def plot(medians: dict, labels: dict, bandlimits: list, path: str) -> None:
    """Plot the times against band-limit on logarithmic axes, with an L^3 reference."""
    import matplotlib
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.ticker import NullLocator

    L = np.array(bandlimits, dtype=float)
    markers = "osD^vP*X"  # distinct in greyscale too
    with matplotlib.rc_context({"font.size": 8, "pdf.fonttype": 42}):
        fig = Figure(figsize=(4, 3), layout="constrained")
        ax = fig.subplots()
        for (name, label), marker in zip(labels.items(), markers, strict=False):
            seconds = np.array([medians[name][b] for b in bandlimits])
            ax.loglog(L, seconds, marker=marker, markersize=3, linewidth=1, label=label)
        # the reference slope, above the slowest operator
        slowest = max(medians[name][bandlimits[-1]] for name in labels)
        reference = 2 * slowest * (L / L[-1]) ** 3
        ax.loglog(L, reference, ":", color="0.4", linewidth=1, label=r"$\propto L^3$")
        ticks = [
            b for b in bandlimits if b & (b - 1) == 0
        ] or bandlimits  # powers of two
        ax.set_xticks(ticks, [str(b) for b in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("band-limit $L$")
        ax.set_ylabel("time (s)")
        ax.legend(frameon=False)
        fig.savefig(path)


def replot(record: str, path: str, plot_from: int) -> None:
    """Plot the times of a recorded run, from the band-limit plot_from, without timing."""
    with open(record) as file:
        data = json.load(file)
    names = data["arguments"]["operators"]
    bandlimits = [L for L in data["arguments"]["bandlimits"] if L >= plot_from]
    medians = {
        name: {L: data["median_seconds"][name][str(L)] for L in bandlimits}
        for name in names
    }
    labels = {
        name: OPERATORS.get(name, data["operators"][name]["label"]) for name in names
    }
    plot(medians, labels, bandlimits, path)


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
        default=BANDLIMITS,
        help=f"band-limits, at least N = {N} (default {' '.join(map(str, BANDLIMITS))})",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=list(OPERATORS),
        help=f"operators of benchmarks/equivariance.py to time (default {' '.join(OPERATORS)})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed runs per operator and band-limit (default 5)",
    )
    parser.add_argument(
        "--fit-from",
        type=int,
        default=32,
        help="smallest band-limit of the power-law fit (default 32)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "default"),
        default="cpu",
        help="'cpu' runs on one CPU core, so that the time counts the operations; "
        "'default' runs on JAX's default devices, such as a GPU (default cpu)",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=None,
        help="the CPU core for --device cpu (default: the fastest available)",
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
        help="precision of single-precision matrix products, as in benchmarks/equivariance.py",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "-o", "--output", default=None, help="write the results as JSON to this file"
    )
    parser.add_argument(
        "--plot", default=None, help="write the plot to this file, such as scaling.pdf"
    )
    parser.add_argument(
        "--plot-from",
        type=int,
        default=16,
        help="smallest band-limit of the plot (default 16)",
    )
    parser.add_argument(
        "--replot",
        default=None,
        metavar="JSON",
        help="plot the times recorded in this JSON file to --plot, without timing",
    )
    args = parser.parse_args(argv)
    args.bandlimits = sorted(set(args.bandlimits))
    if args.bandlimits[0] < N:
        parser.error(f"the band-limits must be at least N = {N}")
    return args


def main(argv=None):
    """Run the benchmark and report the results."""
    args = parse_args(argv)
    if args.replot:
        if not args.plot:
            raise SystemExit("--replot needs --plot")
        return replot(args.replot, args.plot, args.plot_from)
    hardware = use_one_core(args.core) if args.device == "cpu" else {"kind": "default"}
    import jax

    jax.config.update("jax_enable_x64", args.precision == "double")
    if args.matmul_precision == "highest":
        jax.config.update("jax_default_matmul_precision", "highest")
    cases = {case.name: case for case in common.catalogue()}
    unknown = [name for name in args.operators if name not in cases]
    if unknown:
        raise SystemExit(f"unknown operators: {' '.join(unknown)}")
    labels = {name: OPERATORS.get(name, cases[name].label) for name in args.operators}

    date_time = datetime.datetime.now().isoformat(timespec="seconds")
    system, code = common.system_info(), common.code_revisions(__file__)
    common.warn_about_code(code)
    provenance = [
        f"s2conv scaling benchmark, {date_time}, JAX {system['jax']}, "
        f"s2fft {system['s2fft']}, {describe(hardware, system)}.",
        f"Code: {common.code_summary(code)}.",
    ]
    print("\n".join(provenance))

    key = jax.random.key(args.seed)
    seconds = {name: {} for name in args.operators}
    for L in args.bandlimits:
        for i, name in enumerate(args.operators):
            build_key, time_key = jax.random.split(
                jax.random.fold_in(jax.random.fold_in(key, i), L)
            )
            spec = cases[name].build(L, build_key)
            seconds[name][L] = common.pipeline_times(spec, L, time_key, args.repeats)
            median = duration(common.median(seconds[name][L]))
            print(f"L = {L:3d}  {name:10s} {median:>10s}", file=sys.stderr)
    medians = {
        name: {L: common.median(runs) for L, runs in by_L.items()}
        for name, by_L in seconds.items()
    }
    exponents = {
        name: fit_power_law(
            args.bandlimits, [medians[name][L] for L in args.bandlimits], args.fit_from
        )
        for name in args.operators
    }
    print(
        f"\n### Time against band-limit, {args.precision} precision, median of "
        f"{args.repeats} runs\n"
    )
    print(
        f"spin_2_1 is C_{{2→1}}; spin_2_G, group_2 and group_G are C_{{2→G}}, C_{{G→2}} and "
        f"C_{{G→G}}, with N = {N}.\n"
    )
    print(markdown_table(medians, args.bandlimits, exponents, args.fit_from))

    if args.output:
        record = {
            "date_time": date_time,
            "command": " ".join(sys.argv),
            "arguments": vars(args),
            "system": system,
            "hardware": hardware,
            "code": code,
            "operators": {
                name: {"label": cases[name].label, "domain": cases[name].domain}
                for name in args.operators
            },
            "seconds": {
                name: {str(L): runs for L, runs in by_L.items()}
                for name, by_L in seconds.items()
            },
            "median_seconds": {
                name: {str(L): t for L, t in by_L.items()}
                for name, by_L in medians.items()
            },
            "exponents": {
                name: None
                if fit is None
                else {
                    "exponent": fit[0],
                    "prefactor_seconds": fit[1],
                    "fit_from": args.fit_from,
                }
                for name, fit in exponents.items()
            },
        }
        with open(args.output, "w") as file:
            json.dump(record, file, indent=2)
    if args.plot:
        plot(
            medians,
            labels,
            [L for L in args.bandlimits if L >= args.plot_from],
            args.plot,
        )


if __name__ == "__main__":
    main()
