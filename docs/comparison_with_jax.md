# GraphGP.jl vs the JAX `graphgp` implementation

This note compares GraphGP.jl — a backend-agnostic (KernelAbstractions + CUDA) rewrite of the
Vecchia / nearest-neighbour GP refinement core — against the original JAX `graphgp` package and
its companion CUDA extension. It leads with correctness ("same answer") and then reports
throughput. Everything here is reproducible with the harness in
[`bench/compare/`](../bench/compare/README.md); the numbers below were measured on a 2-socket
AMD EPYC 7763 (128 cores, 16 NUMA nodes) + 1× NVIDIA RTX A6000.

## Summary

- **Same answer.** On an identical graph, GraphGP.jl reproduces the JAX reference to **~1e-13**
  on CPU (Float64) and **~1e-5** on GPU (Float32 / fast-math) — the latter being the same
  agreement level as `graphgp`'s own CUDA extension.
- **GPU: parity + autodiff.** GraphGP.jl matches the hand-written CUDA extension at scale, and
  is faster at small/medium N (the extension is launch-overhead-bound there). Unlike the CUDA
  extension, GraphGP.jl provides analytic gradients (∂/∂cov_vals, ∂/∂hyperparameters, ∂/∂xi,
  ∂/∂points) on both CPU and GPU.
- **CPU: a large speedup over pure JAX.** GraphGP.jl is roughly **5× faster per core** and
  **~12–15× better-scaling across cores**, for ~**50–90×** end-to-end on a fully-utilised host.
  This is structural, not a tuning artefact (analysis below).

## Method and fairness controls

- **One shared graph.** All paths consume the identical dumped graph: same integer-lattice
  points, same neighbour lists, same covariance table. Differences are implementation-only.
- **Two precision tiers.** Correctness is checked in Float64 (independent of f32 summation
  order); throughput is measured in Float32, the production precision for this problem (the
  inputs, the Vecchia approximation, and the downstream science do not carry f64 accuracy).
- **Matched CPU resources.** JAX-CPU and Julia-CPU are pinned to the *same* cores with
  `taskset`, and JAX's OpenBLAS/OMP thread count is capped to the same budget. Comparisons were
  taken back-to-back so both face identical machine load.
- **Conservative for Julia.** Where the two used different graphs in ad-hoc runs, Julia used
  *random*-neighbour synthetic graphs (worse gather locality) and JAX used real spatial graphs —
  i.e. the harder case for Julia. The shared-graph harness removes this confound entirely.

## Correctness (Float64, L2-norm relative error vs JAX-CPU)

| path | logdet | xi (refine_inv) | grad ∂logdet/∂cov_vals |
| --- | --- | --- | --- |
| `jax-gpu` (pure JAX) | 0 | 8.6e-14 | 2.1e-15 |
| `julia-cpu` | 0 | 1.2e-13 | 9.6e-14 |
| `cuda-gpu` (CUDA ext, f32) | 2.7e-6 | 6.1e-5 | n/a (no autodiff) |
| `julia-gpu` (f32/fast-math) | 5.3e-6 | 8.1e-5 | 5.4e-5 |

The CPU/Float64 paths are identical to round-off (~1e-13). The GPU paths agree to ~1e-5 — the
expected level for Float32 with fused fast-math arithmetic, and the *same* level as the CUDA
extension. (Per-element relative error is not used here: `xi` has near-zero entries where it is
undefined; the L2-norm relative error is the robust metric.)

## Throughput (Float32, M points/s; higher is better)

<!-- HEADLINE_TABLE -->

Two regimes worth separating:

- **GPU.** GraphGP.jl is at parity with the CUDA extension at scale and ahead at small/medium N
  (the extension is launch-overhead-bound below ~1 M points). See
  [`GPU_BENCHMARK_RESULTS.md`](../test/GPU_BENCHMARK_RESULTS.md) for the 0.2 M–20 M sweep; pure
  JAX OOMs past ~20 M because it materialises the `(M, k+1, k+1)` tensor.
- **CPU.** GraphGP.jl is ~50–90× faster than pure JAX. The decomposition:

  | factor | measurement | cause |
  | --- | --- | --- |
  | ~5× per core | 1-thread, same core | fused per-point kernel: no `(M,k+1,k+1)` materialisation, lower-triangle-only Cholesky |
  | ~12–15× scaling | 1→128 cores | native threads scale ~80×; XLA-CPU scales only ~5× |

## Why JAX-CPU is slow here (and can't easily be tuned)

The pure-JAX CPU path is:

```python
K = jax.vmap(compute_cov_matrix)(...)   # materialise the full (M, k+1, k+1) tensor (~1 GB at 2 M)
L = jnp.linalg.cholesky(K)              # batched Cholesky over all M matrices
```

Two structural costs follow, neither fixable from stock JAX:

1. **Materialisation.** XLA must write and re-read the full covariance tensor (memory-bound),
   where the fused kernel keeps each `(k+1)²` block in registers. This is also why pure JAX
   OOMs at large N.
2. **Batched small-matrix Cholesky.** `jnp.linalg.cholesky` over millions of 11×11 matrices
   lowers on CPU to a poorly-parallelised loop of LAPACK calls. Measured in isolation it caps at
   **~0.6 M/s** on 64 cores — i.e. JAX's Cholesky *step alone* is already ~25–35× slower than
   GraphGP.jl's *entire* fused logdet.

We verified this is not a configuration miss:

- XLA fast-math + Eigen-threading flags (`--xla_cpu_enable_fast_math`,
  `--xla_cpu_multi_thread_eigen`): no measurable change.
- Adding cores barely helps (1→64 cores moves JAX ~5×; the native path scales ~80×).
- The `jnp.linalg.cholesky` floor (~0.6 M/s) is independent of the surrounding code.

Closing the gap in JAX would require a hand-written custom kernel (Pallas / CUDA) — which is
exactly what the `graphgp` CUDA extension *is*, and that extension is GPU-only and has no
autodiff. GraphGP.jl provides the fused-kernel approach as a single portable implementation that
runs on CPU **and** GPU **and** differentiates.

## Reproduce

```bash
cd julia/GraphGP/bench/compare
./run_all.sh 2000000 10 3 64      # N K D NTHREADS
cat results/report.md
```

See [`bench/compare/README.md`](../bench/compare/README.md) for knobs and the per-path scripts.
