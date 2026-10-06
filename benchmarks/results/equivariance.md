s2conv equivariance benchmark, 2026-10-06T00:53:35, JAX 0.11.2, s2fft 1.4.1.dev21, cuda:0.
Code: s2conv commit 01c7370a817591d18fed3a34214a27a25dc7811e, s2fft commit 86a4dff303539202ebbfa28661d85331e0e2bff4.

### L = 32, 100 signals x 100 rotations, double precision

| Operator | Domain | Mean error, norm. by R(Tf) | Mean error, norm. by T(Rf) | Max | Reduction | Time (L = 128) |
|:--|:--|--:|--:|--:|--:|--:|
| spin_0_0 | S2, s=0 -> 0 | 4.85e-16 ± 1.4e-17 | 4.85e-16 | 5.4e-16 | 1.4e-16 | 18.10 ms |
| spin_2_1 | S2, s=2 -> 1 | 4.89e-16 ± 1.4e-17 | 4.89e-16 | 5.6e-16 | 1.3e-16 | 17.79 ms |
| spin_-1_2 | S2, s=-1 -> 2 | 4.68e-16 ± 1.5e-17 | 4.68e-16 | 5.3e-16 | 1.3e-16 | 18.49 ms |
| spin_0_G | S2, s=0 -> SO(3), N=5 | 4.79e-16 ± 7.2e-18 | 4.79e-16 | 5.0e-16 | 1.3e-16 | 41.70 ms |
| spin_2_G | S2, s=2 -> SO(3), N=5 | 4.88e-16 ± 7.5e-18 | 4.88e-16 | 5.2e-16 | 1.4e-16 | 42.44 ms |
| group_0 | SO(3), N=5 -> S2, s=0 | 5.28e-16 ± 1.2e-17 | 5.28e-16 | 5.7e-16 | 2.1e-16 | 56.38 ms |
| group_2 | SO(3), N=5 -> S2, s=2 | 5.27e-16 ± 1.2e-17 | 5.27e-16 | 5.7e-16 | 2.2e-16 | 55.85 ms |
| group_G | SO(3) -> SO(3), N=5 | 8.70e-16 ± 7.9e-18 | 8.70e-16 | 9.0e-16 | 0 | 79.57 ms |
| axisymmetric | S2, s=0 -> 0 | 3.95e-16 ± 1.4e-17 | 3.95e-16 | 4.5e-16 | 1.9e-16 | 18.40 ms |
| left | S2, s=0 -> 0 | 4.33e-16 ± 1.4e-17 | 4.33e-16 | 5.0e-16 | 0 | 18.86 ms |
| directional | S2, s=2 -> SO(3), N=5 | 4.42e-16 ± 7.0e-18 | 4.42e-16 | 4.7e-16 | 1.5e-16 | 42.08 ms |
| synthesis | SO(3), N=5 -> S2, s=2 | 5.06e-16 ± 1.0e-17 | 5.06e-16 | 5.5e-16 | 1.6e-16 | 56.41 ms |
| eth | S2, s=1 -> 2 | 5.13e-16 ± 1.3e-17 | 5.13e-16 | 5.6e-16 | 0 | 19.21 ms |
| ethbar | S2, s=1 -> 0 | 5.13e-16 ± 1.3e-17 | 5.13e-16 | 5.6e-16 | 0 | 18.15 ms |
| e_mode | S2, s=2 -> 0 | 5.57e-16 ± 1.3e-17 | 5.57e-16 | 6.1e-16 | 0 | 19.25 ms |
| kaiser_squires | S2, s=0 -> 2 | 4.73e-16 ± 1.1e-17 | 4.73e-16 | 5.1e-16 | 0 | 19.19 ms |
| standard_correlation | S2, s=0 -> S2 (section), N=5 | 1.37e+00 ± 1.5e-01 | 1.37e+00 | 1.6e+00 | 3.5e-16 | 71.42 ms |
| commutative_anisotropic | S2, s=0 -> S2 (section), N=5 | 1.31e+00 ± 7.8e-02 | 1.31e+00 | 1.5e+00 | - | 69.53 ms |
| sifting | S2, s=0 -> 0 | 1.39e+00 ± 2.6e-02 | 1.40e+00 | 1.5e+00 | - | 17.89 ms |
| harmonic_multiplication | S2, s=0 -> 0 | 1.40e+00 ± 3.1e-02 | 1.39e+00 | 1.5e+00 | - | 18.00 ms |

### L = 64, 100 signals x 100 rotations, double precision

| Operator | Domain | Mean error, norm. by R(Tf) | Mean error, norm. by T(Rf) | Max | Reduction | Time (L = 128) |
|:--|:--|--:|--:|--:|--:|--:|
| spin_0_0 | S2, s=0 -> 0 | 6.64e-16 ± 1.0e-17 | 6.64e-16 | 7.1e-16 | 1.4e-16 | 18.10 ms |
| spin_2_1 | S2, s=2 -> 1 | 6.51e-16 ± 9.2e-18 | 6.51e-16 | 6.9e-16 | 1.4e-16 | 17.79 ms |
| spin_-1_2 | S2, s=-1 -> 2 | 6.34e-16 ± 9.2e-18 | 6.34e-16 | 6.7e-16 | 1.4e-16 | 18.49 ms |
| spin_0_G | S2, s=0 -> SO(3), N=5 | 9.00e-16 ± 6.6e-18 | 9.00e-16 | 9.3e-16 | 1.4e-16 | 41.70 ms |
| spin_2_G | S2, s=2 -> SO(3), N=5 | 9.10e-16 ± 6.8e-18 | 9.10e-16 | 9.4e-16 | 1.3e-16 | 42.44 ms |
| group_0 | SO(3), N=5 -> S2, s=0 | 9.72e-16 ± 1.2e-17 | 9.72e-16 | 1.0e-15 | 2.0e-16 | 56.38 ms |
| group_2 | SO(3), N=5 -> S2, s=2 | 9.66e-16 ± 1.1e-17 | 9.66e-16 | 1.0e-15 | 2.2e-16 | 55.85 ms |
| group_G | SO(3) -> SO(3), N=5 | 1.18e-15 ± 5.5e-18 | 1.18e-15 | 1.2e-15 | 0 | 79.57 ms |
| axisymmetric | S2, s=0 -> 0 | 5.82e-16 ± 1.2e-17 | 5.82e-16 | 6.2e-16 | 2.1e-16 | 18.40 ms |
| left | S2, s=0 -> 0 | 5.83e-16 ± 9.4e-18 | 5.83e-16 | 6.2e-16 | 0 | 18.86 ms |
| directional | S2, s=2 -> SO(3), N=5 | 8.18e-16 ± 7.2e-18 | 8.18e-16 | 8.5e-16 | 1.5e-16 | 42.08 ms |
| synthesis | SO(3), N=5 -> S2, s=2 | 9.16e-16 ± 1.0e-17 | 9.16e-16 | 9.6e-16 | 1.6e-16 | 56.41 ms |
| eth | S2, s=1 -> 2 | 7.01e-16 ± 9.3e-18 | 7.01e-16 | 7.4e-16 | 0 | 19.21 ms |
| ethbar | S2, s=1 -> 0 | 7.00e-16 ± 9.3e-18 | 7.00e-16 | 7.4e-16 | 0 | 18.15 ms |
| e_mode | S2, s=2 -> 0 | 7.28e-16 ± 9.1e-18 | 7.28e-16 | 7.6e-16 | 0 | 19.25 ms |
| kaiser_squires | S2, s=0 -> 2 | 6.41e-16 ± 7.3e-18 | 6.41e-16 | 6.7e-16 | 0 | 19.19 ms |
| standard_correlation | S2, s=0 -> S2 (section), N=5 | 1.35e+00 ± 1.4e-01 | 1.35e+00 | 1.5e+00 | 3.9e-16 | 71.42 ms |
| commutative_anisotropic | S2, s=0 -> S2 (section), N=5 | 1.34e+00 ± 7.3e-02 | 1.34e+00 | 1.5e+00 | - | 69.53 ms |
| sifting | S2, s=0 -> 0 | 1.41e+00 ± 1.3e-02 | 1.41e+00 | 1.5e+00 | - | 17.89 ms |
| harmonic_multiplication | S2, s=0 -> 0 | 1.40e+00 ± 1.5e-02 | 1.41e+00 | 1.5e+00 | - | 18.00 ms |

### L = 128, 100 signals x 100 rotations, double precision

| Operator | Domain | Mean error, norm. by R(Tf) | Mean error, norm. by T(Rf) | Max | Reduction | Time (L = 128) |
|:--|:--|--:|--:|--:|--:|--:|
| spin_0_0 | S2, s=0 -> 0 | 9.23e-16 ± 7.4e-18 | 9.23e-16 | 9.5e-16 | 1.4e-16 | 18.10 ms |
| spin_2_1 | S2, s=2 -> 1 | 9.03e-16 ± 6.9e-18 | 9.03e-16 | 9.3e-16 | 1.4e-16 | 17.79 ms |
| spin_-1_2 | S2, s=-1 -> 2 | 8.70e-16 ± 7.2e-18 | 8.70e-16 | 9.0e-16 | 1.4e-16 | 18.49 ms |
| spin_0_G | S2, s=0 -> SO(3), N=5 | 1.24e-15 ± 4.6e-18 | 1.24e-15 | 1.3e-15 | 1.3e-16 | 41.70 ms |
| spin_2_G | S2, s=2 -> SO(3), N=5 | 1.25e-15 ± 4.6e-18 | 1.25e-15 | 1.3e-15 | 1.3e-16 | 42.44 ms |
| group_0 | SO(3), N=5 -> S2, s=0 | 1.32e-15 ± 8.1e-18 | 1.32e-15 | 1.4e-15 | 2.0e-16 | 56.38 ms |
| group_2 | SO(3), N=5 -> S2, s=2 | 1.32e-15 ± 7.6e-18 | 1.32e-15 | 1.4e-15 | 2.1e-16 | 55.85 ms |
| group_G | SO(3) -> SO(3), N=5 | 1.62e-15 ± 3.9e-18 | 1.62e-15 | 1.6e-15 | 0 | 79.57 ms |
| axisymmetric | S2, s=0 -> 0 | 8.31e-16 ± 7.0e-18 | 8.31e-16 | 8.5e-16 | 1.7e-16 | 18.40 ms |
| left | S2, s=0 -> 0 | 7.58e-16 ± 6.6e-18 | 7.58e-16 | 7.8e-16 | 0 | 18.86 ms |
| directional | S2, s=2 -> SO(3), N=5 | 1.10e-15 ± 5.3e-18 | 1.10e-15 | 1.1e-15 | 1.5e-16 | 42.08 ms |
| synthesis | SO(3), N=5 -> S2, s=2 | 1.25e-15 ± 6.7e-18 | 1.25e-15 | 1.3e-15 | 1.6e-16 | 56.41 ms |
| eth | S2, s=1 -> 2 | 9.62e-16 ± 6.4e-18 | 9.62e-16 | 9.8e-16 | 0 | 19.21 ms |
| ethbar | S2, s=1 -> 0 | 9.62e-16 ± 6.4e-18 | 9.62e-16 | 9.9e-16 | 0 | 18.15 ms |
| e_mode | S2, s=2 -> 0 | 9.67e-16 ± 6.4e-18 | 9.67e-16 | 9.9e-16 | 0 | 19.25 ms |
| kaiser_squires | S2, s=0 -> 2 | 8.77e-16 ± 5.1e-18 | 8.77e-16 | 9.0e-16 | 0 | 19.19 ms |
| standard_correlation | S2, s=0 -> S2 (section), N=5 | 1.33e+00 ± 1.4e-01 | 1.33e+00 | 1.5e+00 | 3.8e-16 | 71.42 ms |
| commutative_anisotropic | S2, s=0 -> S2 (section), N=5 | 1.32e+00 ± 7.4e-02 | 1.32e+00 | 1.4e+00 | - | 69.53 ms |
| sifting | S2, s=0 -> 0 | 1.41e+00 ± 6.5e-03 | 1.41e+00 | 1.4e+00 | - | 17.89 ms |
| harmonic_multiplication | S2, s=0 -> 0 | 1.41e+00 ± 7.6e-03 | 1.41e+00 | 1.4e+00 | - | 18.00 ms |
