# Sample output of run_all.sh (N=2M, K=10, D=3, 64 threads; EPYC 7763 + A6000)
# Machine was under load (other jobs); CPU numbers are a floor. Regenerate with ./run_all.sh.

## Correctness cross-check (Float64)

Reference: `jax-cpu`. L2-norm relative / max-absolute difference vs reference:

| path | logdet (rel) | xi (rel, abs) | grad cov_vals (rel, abs) |
| --- | --- | --- | --- |
| `cuda-gpu` | 1.1e-02 | 8.4e-05, 2.2e-02 | n/a (no autodiff) |
| `jax-cpu`  *(ref)* | 0.0e+00 | 0.0e+00, 0.0e+00 | 0.0e+00, 0.0e+00 |
| `jax-gpu` | 0.0e+00 | 1.3e-13, 3.3e-11 | 4.1e-15, 3.5e-06 |
| `julia-cpu` | 0.0e+00 | 1.7e-13, 4.4e-11 | 9.3e-14, 8.3e-05 |
| `julia-gpu` | 2.4e-06 | 5.8e-05, 1.8e-02 | 2.3e-05, 2.0e+04 |

## Throughput (Float32, M pts/s; higher is better)

| path | refine_logdet | refine_inv | grad (cov_vals) |
| --- | --- | --- | --- |
| `jax-cpu` | 0.2 | 0.1 | 0.1 |
| `jax-gpu` | 8.6 | 7.7 | 0.0 |
| `cuda-gpu` | 128.1 | 115.5 | n/a |
| `julia-cpu` | 12.8 | 12.6 | 6.4 |
| `julia-gpu` | 201.3 | 156.5 | 45.4 |

