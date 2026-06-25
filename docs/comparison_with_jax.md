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
- **GPU: parity + a fuller gradient surface.** GraphGP.jl matches the hand-written CUDA extension
  at scale, and is faster at small/medium N (the extension is launch-overhead-bound there). The
  CUDA extension differentiates the forward `generate` w.r.t. `xi` and `cov_vals` (on-GPU, fast),
  but has **no gradient rule for the `refine_logdet` / `refine_inv` ops** (`NotImplementedError`).
  GraphGP.jl provides both the forward-generate derivatives **and** the analytic log-det /
  inverse-loss / hyperparameter / point-position gradients, on CPU and GPU.
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

## Correctness (Float64, L2-norm relative error vs JAX-CPU; N = 2 M)

| path | logdet | xi (refine_inv) | grad ∂logdet/∂cov_vals |
| --- | --- | --- | --- |
| `jax-gpu` (pure JAX) | 0 | 1.3e-13 | 4.1e-15 |
| `julia-cpu` | 0 | 1.7e-13 | 9.3e-14 |
| `cuda-gpu` (CUDA ext, f32) | 1.1e-2 | 8.4e-5 | n/a (no log-det-grad rule) |
| `julia-gpu` (f32/fast-math) | 2.4e-6 | 5.8e-5 | 2.3e-5 |

The CPU/Float64 paths are identical to round-off (~1e-13). The GPU paths agree on `xi` to
~1e-5 — the expected level for Float32 with fused fast-math arithmetic. (Per-element relative
error is not used here: `xi` has near-zero entries where it is undefined; the L2-norm relative
error is the robust metric.) The `cuda-gpu` `logdet` (1.1e-2) reflects the extension summing 2 M
per-point terms naively in Float32; GraphGP.jl's reduction is more accurate (2.4e-6).

## Throughput (Float32, M points/s; higher is better)

N = 2 M, K = 10, D = 3. CPU paths confined to the **same 64 cores** (pure-JAX with OpenBLAS/OMP
capped to 64); GPU paths on 1× A6000.

| path | refine_logdet | refine_inv | grad (∂cov_vals) |
| --- | --- | --- | --- |
| `jax-cpu` (pure JAX, 64 cores) | 0.2 | 0.1 | 0.1 |
| `julia-cpu` (GraphGP.jl, 64 cores) | **12.8** | **12.6** | **6.4** |
| `jax-gpu` (pure JAX) | 8.6 | 7.7 | <0.05 |
| `cuda-gpu` (graphgp CUDA ext) | 128.1 | 115.5 | n/a (no log-det-grad rule) |
| `julia-gpu` (GraphGP.jl) | **201.3** | **156.5** | **45.4** |

At matched cores, GraphGP.jl-CPU is **~64×** (logdet) to **~126×** (inv) faster than pure-JAX-CPU.
On GPU it is ~1.4–1.6× ahead of the CUDA extension *at this N* (still the launch-overhead
regime — see below) and ~23× over pure-JAX-GPU, while also producing the log-det gradient (`grad`
column) fast: the CUDA extension has no rule for it (`NotImplementedError`), and pure-JAX-GPU can
but only at <0.05 M/s. (The CUDA extension *does* differentiate the forward `generate` w.r.t. `xi`
and `cov_vals` on-GPU — that surface is benchmarked in `docs/benchmarks.md` §3.)

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
exactly what the `graphgp` CUDA extension *is*, and that extension is GPU-only and — though it differentiates the forward generate —
has no log-det/inverse-loss gradient rule. GraphGP.jl provides the fused-kernel approach as a single portable implementation that
runs on CPU **and** GPU **and** differentiates.

## What's new in the Julia port (map)

The port is this repository (`GraphGP.jl`). It is a standalone Julia package — no
Python or C++ dependency — built on KernelAbstractions so one kernel set runs on CPU and CUDA.

- **One fused kernel, two backends.** [`src/kernels.jl`](../src/kernels.jl) (one workitem per
  refined point; assembles the `(k+1)²` covariance, factorises it in registers, never
  materialises the batch) + [`src/linalg.jl`](../src/linalg.jl) (the per-point Cholesky / solve
  primitives). The CPU backend dispatches to a native threaded path,
  [`src/cpu_native.jl`](../src/cpu_native.jl) (the KA CPU backend is 5–13× slower; see below).
- **Analytic gradients, including the ones the CUDA extension lacks** (log-det / inverse-loss /
  hyperparameter / point-position; the extension only differentiates the forward `generate`).
  [`src/kernels_adjoint.jl`](../src/kernels_adjoint.jl),
  [`src/grad.jl`](../src/grad.jl) — reverse-mode Cholesky pullback for ∂/∂cov_vals, plus
  ∂/∂hyperparameters, ∂/∂xi, and ∂/∂points (positions treated as continuous), CPU **and** GPU.
  [`src/chainrules.jl`](../src/chainrules.jl) exposes them as ChainRules `rrule`s so Zygote
  composes arbitrary scalar losses.
- **Full graph construction in Julia, on GPU.** [`src/graph_build.jl`](../src/graph_build.jl),
  [`src/tree_gpu.jl`](../src/tree_gpu.jl) — `build_graph_ka` runs tree build → k-NN query →
  depths → reorder → quantise entirely on the device (`build_graph_ka(CuArray(points), …) →
  generate/refine/gradients`, no host round-trip).
- **Benchmarks & validation.** [`test/GPU_BENCHMARK_RESULTS.md`](../test/GPU_BENCHMARK_RESULTS.md)
  (0.2 M–20 M sweep, GPU vs the CUDA extension vs pure JAX; CPU NUMA scaling),
  [`bench/compare/`](../bench/compare/README.md) (this note's harness),
  [`FEATURE_COVERAGE.md`](../FEATURE_COVERAGE.md) (parity audit vs the Python API).
- **Try it.** [`examples/parity_and_autodiff.jl`](../examples/parity_and_autodiff.jl) — build a
  graph, evaluate forward, and take all four gradients in ~40 lines; CPU out of the box, GPU by
  uncommenting one `to_backend` block.

## Reproduce

```bash
# the comparison in this note (correctness cross-check + throughput)
cd bench/compare
./run_all.sh 2000000 10 3 64      # N K D NTHREADS
cat results/report.md             # also committed as sample_report_2M.md

# the package test suite (CPU + auto GPU testset if CUDA is present)
julia --project=. -e 'using Pkg; Pkg.test()'

# the example
julia --project=. examples/parity_and_autodiff.jl
```

See [`bench/compare/README.md`](../bench/compare/README.md) for knobs and the per-path scripts.
