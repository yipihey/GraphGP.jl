# Sample output of run_all.sh (N=2M, K=10, D=3, 64 cores; EPYC 7763 + A6000)
# Illustrative `run_all.sh` capture. CPU throughput at this N has run-to-run variance (NUMA/load,
# ~12–19 M/s for julia-cpu); the robust claim is the ~50–100× margin over jax-cpu. The canonical,
# current numbers — including the at-scale / GPU-fill forward+derivative comparison — are in
# docs/benchmarks.md. Regenerate this with ./run_all.sh.

## Correctness cross-check (Float64)

Reference: `jax-cpu`. L2-norm relative / max-absolute difference vs reference:

| path | logdet (rel) | xi (rel, abs) | grad cov_vals (rel, abs) |
| --- | --- | --- | --- |
| `cuda-gpu` | 1.1e-02 | 8.4e-05, 2.2e-02 | n/a (no log-det-grad rule) |
| `jax-cpu`  *(ref)* | 0.0e+00 | 0.0e+00, 0.0e+00 | 0.0e+00, 0.0e+00 |
| `jax-gpu` | 0.0e+00 | 1.3e-13, 3.3e-11 | 3.0e-15, 2.4e-06 |
| `julia-cpu` | 0.0e+00 | 1.6e-13, 4.3e-11 | 9.3e-14, 8.2e-05 |
| `julia-gpu` | 2.4e-06 | 5.8e-05, 1.8e-02 | 2.3e-05, 2.0e+04 |

## Throughput (Float32, M pts/s; higher is better)

| path | refine_logdet | refine_inv | grad (cov_vals) |
| --- | --- | --- | --- |
| `jax-cpu` | 0.2 | 0.1 | 0.1 |
| `jax-gpu` | 8.6 | 7.8 | 0.0 |
| `cuda-gpu` | 132.2 | 118.9 | n/a |
| `julia-cpu` | 19.3 | 18.4 | 9.6 |
| `julia-gpu` | 123.1 | 109.4 | 30.2 |

(`grad (cov_vals)` is the log-det gradient `refine_logdet_grad_vals`; the CUDA extension has no rule
for it — it differentiates the forward `generate`, not `refine_logdet`. The `julia-gpu` grad max-abs
of 2e4 is a near-zero `xi` denominator artefact; the L2-relative 2.3e-5 is the robust metric.)
