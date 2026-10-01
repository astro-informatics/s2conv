# Equivariance benchmark results

The outputs of `benchmarks/equivariance.py`, as the script wrote them, unedited:

| File | Contents |
|:--|:--|
| `equivariance.md` | The printed output: two provenance lines and a markdown table per band-limit |
| `equivariance.json` | All results, with the arguments, the system and the git commits of the code |
| `equivariance.tex` | The LaTeX table at L = 128: operator, reduction error, equivariance error and time |

They were produced on 1 October 2026 by running, from the repository root,

```bash
python benchmarks/equivariance.py -L 32 64 128 -o benchmarks/results/equivariance.json \
    --latex benchmarks/results/equivariance.tex > benchmarks/results/equivariance.md
```

that is, with 100 random signals and 100 Haar-random rotations per operator at band-limits 32, 64 and 128, in double
precision, and with timings at L = 128 (median of 5 runs), on an NVIDIA GB10 GPU with JAX 0.11.2 and Python 3.12.
Each file names the code that produced it: s2conv commit c5648767b29a4936cd42ca08954decb3944c39f0 and s2fft commit
86a4dff303539202ebbfa28661d85331e0e2bff4 (version 1.4.1.dev21), both without uncommitted changes.

The equivariance error of an operator T is the mean, over all pairs of a signal f and a rotation 𝓡, of
‖T(𝓡f) − 𝓡(Tf)‖ / ‖𝓡(Tf)‖, which is the column of the LaTeX table.  The markdown tables and the JSON also give the
mean normalised by ‖T(𝓡f)‖, as in Cobb et al. (2021) and Ocampo et al. (2023), and the standard deviations and
maxima.  The [user guide](https://astro-informatics.github.io/s2conv/user_guide/equivariance.html) describes the
protocol.
