s2conv scaling benchmark, 2026-10-01T08:34:33, JAX 0.11.2, s2fft 1.4.1.dev21, one CPU core (core 5, 3900 MHz).
Code: s2conv commit bd6af5d8b83911bf3055de1dcfa81dd7e8f4a1fc, s2fft commit 86a4dff303539202ebbfa28661d85331e0e2bff4.

### Time against band-limit, double precision, median of 5 runs

spin_2_1 is C_{2→1}; spin_2_G, group_2 and group_G are C_{2→G}, C_{G→2} and C_{G→G}, with N = 5.

| L | spin_2_1 | spin_2_G | group_2 | group_G |
|--:|--:|--:|--:|--:|
| 8 | 0.0906 ms | 0.183 ms | 0.311 ms | 0.392 ms |
| 12 | 0.266 ms | 0.488 ms | 0.861 ms | 1.17 ms |
| 16 | 0.371 ms | 1.1 ms | 1.97 ms | 2.71 ms |
| 24 | 0.976 ms | 3.56 ms | 5.68 ms | 8.31 ms |
| 32 | 2.06 ms | 8.14 ms | 12.4 ms | 18.6 ms |
| 48 | 6.92 ms | 27 ms | 47.5 ms | 67.6 ms |
| 64 | 16.4 ms | 68.6 ms | 121 ms | 173 ms |
| 96 | 55.8 ms | 248 ms | 460 ms | 650 ms |
| 128 | 139 ms | 619 ms | 1.15 s | 1.63 s |
| 192 | 528 ms | 2.35 s | 4.15 s | 5.97 s |
| 256 | 1.36 s | 5.87 s | 10.2 s | 14.8 s |
| Exponent, L ≥ 32 | 3.12 | 3.18 | 3.23 | 3.22 |
