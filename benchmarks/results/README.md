# Benchmark results

The outputs of the benchmarks, as the scripts wrote them, unedited.

## Equivariance

The outputs of `benchmarks/equivariance.py`:

| File | Contents |
|:--|:--|
| `equivariance.md` | The printed output: two provenance lines and a markdown table per band-limit |
| `equivariance.json` | All results, with the arguments, the system and the git commits of the code |
| `equivariance.tex` | The LaTeX table at L = 128: operator, reduction error, equivariance error and time |

They were produced on 6 October 2026 by running, from the repository root,

```bash
python benchmarks/equivariance.py -L 32 64 128 --reduction-signals 100 -o benchmarks/results/equivariance.json \
    --latex benchmarks/results/equivariance.tex > benchmarks/results/equivariance.md
```

that is, with 100 random signals and 100 Haar-random rotations per operator, and 100 signals for the reduction error, at band-limits 32, 64 and 128, in double
precision, and with timings at L = 128 (median of 5 runs), on an NVIDIA GB10 GPU with JAX 0.11.2 and Python 3.12.
Each file names the code that produced it: s2conv commit 01c7370a817591d18fed3a34214a27a25dc7811e and s2fft commit
86a4dff303539202ebbfa28661d85331e0e2bff4 (version 1.4.1.dev21), both without uncommitted changes.

The equivariance error of an operator T is the mean, over all pairs of a signal f and a rotation 𝓡, of
‖T(𝓡f) − 𝓡(Tf)‖ / ‖𝓡(Tf)‖, which is the column of the LaTeX table.  The markdown tables and the JSON also give the
mean normalised by ‖T(𝓡f)‖, as in Cobb et al. (2021) and Ocampo et al. (2023), and the standard deviations and
maxima.  The [user guide](https://astro-informatics.github.io/s2conv/user_guide/equivariance.html) describes the
protocol.

## Computation time

The outputs of `benchmarks/scaling.py`:

| File | Contents |
|:--|:--|
| `scaling.md` | The printed output: two provenance lines and the table of times and fitted exponents |
| `scaling.json` | Every run time, the medians and the fits, with the arguments, the system, the hardware and the git commits of the code |
| `scaling.pdf` | The plot: the times against band-limit from L = 16 on logarithmic axes, with an L³ reference |

They were produced on 1 October 2026 by running, from the repository root,

```bash
python benchmarks/scaling.py -o benchmarks/results/scaling.json --plot benchmarks/results/scaling.pdf \
    > benchmarks/results/scaling.md
```

that is, by timing C<sub>2→1</sub>, C<sub>2→G</sub>, C<sub>G→2</sub> and C<sub>G→G</sub>, with random kernels and N = 5,
end to end with their transforms, at band-limits from 8 to 256, as the median of 5 runs in double precision, on one
Cortex-X925 core of an NVIDIA GB10 (core 5, 3.9 GHz) with the multithreading of XLA off.  Each file names the code that
produced it: s2conv commit bd6af5d8b83911bf3055de1dcfa81dd7e8f4a1fc and s2fft commit
86a4dff303539202ebbfa28661d85331e0e2bff4, both without uncommitted changes.  The power laws fitted from L = 32 have
exponents 3.12, 3.18, 3.23 and 3.22.  The [user guide](https://astro-informatics.github.io/s2conv/user_guide/scaling.html)
discusses the results.

`scaling.pdf` was redrawn on 6 October 2026, as a single panel of the times from L = 16 without the fitted exponents in
the legend, from the recorded `scaling.json` without timing again, by running

```bash
python benchmarks/scaling.py --replot benchmarks/results/scaling.json --plot benchmarks/results/scaling.pdf
```

with s2conv commit 01c7370a817591d18fed3a34214a27a25dc7811e.  The times, the fits and the other files are those of the
run above.
