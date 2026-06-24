# GraphGP backends on real ECHOES graphs

Benchmark of the GraphGP implementations on the graphs the ECHOES field pipeline actually builds —
BOSS DR12 CMASS-South *randoms* embedded `(n̂, α·z)` (4-D, the candidate set field generation runs
on) and 2M++ comoving xyz (3-D) — across an N sweep at the production `k = 30`. Reproduce with:

```bash
./run_echoes_bench.sh "boss:120000 boss:560000 boss:2400000 local:70000 local:1400000"
python report_sweep.py results "boss:120000 boss:560000 boss:2400000 local:70000 local:1400000"
```

Hardware: RTX A6000 (46 GB), ~2 TB host RAM. JAX 0.10.x cuda12; the Julia GPU path runs in its own
process (no PJRT clash). CPU paths share 16 cores via `taskset` with capped BLAS threads.

## Table 1 — correctness (Float64, vs `jax-cpu`)

L2-relative difference of logdet / xi / grad(cov_vals).

| survey | N | path | logdet | xi | grad |
| --- | ---: | --- | ---: | ---: | ---: |
| boss | 120,000 | `julia-cpu` | 1.2e-16 | 9.5e-15 | 1.4e-14 |
| boss | 120,000 | `cuda-gpu` | 3.3e-06 | 5.2e-06 | n/a |
| boss | 560,000 | `julia-cpu` | 1.5e-16 | 2.1e-14 | 9.3e-14 |
| boss | 560,000 | `cuda-gpu` | 1.0e-05 | 1.2e-05 | n/a |
| boss | 2,400,000 | `julia-cpu` | 0.0e+00 | 3.7e-14 | 2.4e-13 |
| boss | 2,400,000 | `cuda-gpu` | 1.4e-04 | 2.1e-05 | n/a |
| local | 70,000 | `julia-cpu` | *(ref NaN)* | *(ref NaN)* | *(ref NaN)* |
| local | 1,400,000 | `julia-cpu` | 1.1e-16 | 2.4e-14 | 2.1e-13 |
| local | 1,400,000 | `cuda-gpu` | 1.6e-05 | 1.3e-05 | n/a |

**Julia-CPU reproduces the JAX reference to f64 machine precision everywhere**, and is the only GPU-
capable backend that also returns the gradient (the CUDA extension has no autodiff). The CUDA
extension agrees to its f32 precision (1e-4…1e-6).

**Robustness note (local 70k).** The real 2M++ catalog has **23 coincident lattice positions** (group
members at near-identical sky/redshift). Those duplicate rows make the dense first-block covariance
singular: **`jax-cpu` and `cuda-gpu` both return NaN, while `julia-cpu` stays finite** (logdet
−141,999, no NaN xi). So GraphGP.jl degrades gracefully on degenerate real-data points where the
JAX path and the CUDA extension fail. The cross-check is blank only because the *reference* is NaN.

## Table 2 — throughput (Float32, M pts/s; refine_logdet / refine_inv / grad)

| survey | N | `jax-cpu` | `julia-cpu` | `cuda-gpu` | `jax-gpu` | `julia-gpu` |
| --- | ---: | --- | --- | --- | --- | --- |
| boss | 120,000 | 0.039 / 0.034 / 0.024 | 0.53 / 0.52 / 0.26 | 1.1 / 1.1 / n/a | 1.2 / 1.0 / 0.0022 | **12.9 / 12.0 / 2.7** |
| boss | 560,000 | 0.034 / 0.032 / 0.023 | 0.51 / 0.56 / 0.28 | 3.5 / 3.3 / n/a | 1.2 / 1.0 / 0.0019 | **13.0 / 11.9 / 2.7** |
| boss | 2,400,000 | 0.034 / 0.029 / 0.021 | 0.6 / 0.6 / 0.3 | 9.1 / 8.4 / n/a | **OOM** | **13.2 / 12.0 / 2.7** |
| local | 70,000 | 0.039 / 0.034 / 0.024 | 0.62 / 0.6 / 0.3 | 0.66 / 0.58 / n/a | 1.2 / 1.0 / 0.0039 | **13.1 / 12.1 / 2.8** |
| local | 1,400,000 | 0.033 / 0.03 / 0.021 | 0.57 / 0.57 / 0.29 | 6.4 / 5.4 / n/a | **OOM** | **13.9 / 12.5 / 2.7** |

- **Julia-GPU is steady at ~13 / ~12 / ~2.7 M pts/s** across every N — `O(N·k)` with no tensor
  materialization. It leads the CUDA extension at small/medium N; the CUDA ext catches up on the
  forward ops by 2.4M (9.1) as its launch overhead amortizes, but it **never provides the gradient**.
- **Pure-JAX-GPU's gradient is catastrophic** (~0.002 M pts/s — the 560k grad took **300 s**, autodiff
  through the materialized `(M,31,31)` tensor) and **OOMs entirely at 2.4M / 1.4M**.
- **Julia-CPU is ~15× JAX-CPU** (matched cores/BLAS), and the only CPU path with a usable gradient.

## Table 3 — pure-JAX-GPU out-of-memory at k=30 (the no-OOM headline)

Pure-JAX GPU materialises the dense `(M, k+1, k+1)` tensor; GraphGP.jl and the CUDA ext never do.

| survey | N | M | pure-JAX-GPU | Julia GPU |
| --- | ---: | ---: | --- | --- |
| boss | 120,000 | 118,976 | ran | ran |
| boss | 560,000 | 558,976 | ran | ran |
| boss | 2,400,000 | 2,398,976 | **OOM** | ran |
| local | 70,000 | 68,976 | ran | ran |
| local | 1,400,000 | 1,400,000 | **OOM** | ran |

At the production BOSS scale (`n_cand ≈ 2.4M`, `k = 30`) pure-JAX-GPU cannot form the field at all on
a 46 GB card, while GraphGP.jl generates it at ~13 M pts/s **and** can differentiate it. This is the
decisive win for routing ECHOES' heavy Vecchia field generation through the Julia backend.
