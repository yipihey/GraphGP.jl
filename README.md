# GraphGP.jl

A standalone Julia rewrite of
[graphgp](../../README.md) (scalable Gaussian processes via a Vecchia /
nearest-neighbor-graph approximation), using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for
backend-agnostic CPU + GPU kernels, hand-written analytic adjoints, and
[ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) for composable
differentiability.

The package implements the **entire pipeline in Julia** — k-d tree build, neighbor query,
depth ordering, and the hot per-point inner loop (assemble each `(k+1)×(k+1)` covariance on the
fly, factorize it, emit the per-point scalar) — with no Python/JAX dependency. The inner loop
dominates runtime when the number of refined points `M = N − n0` reaches the billions.

## Design choices

- **f32 by default.** All real arithmetic is `Float32`; a `Float64` path exists only as a
  debugging oracle.
- **Integer lattice coordinates.** Points live on a 21-bit-per-axis (`UInt32`) Morton-friendly
  lattice. Coordinate differences are exact integer subtractions; squared distances accumulate
  in `Int64` and are cast to `Float32` only at the `sqrt` → kernel-lookup step.
- **One workitem per point.** With `k ≈ 10` the per-point matrix is ~`11×11`, kept in private
  memory; the full `(M, k+1, k+1)` tensor is never materialized (the memory win over JAX).
- **Backend-agnostic, CUDA-free core.** The package has no hard CUDA dependency; GPU support is
  selected automatically from `CuArray` inputs (KernelAbstractions), with the few GPU-specific
  primitives (`sortperm`/`sort!` in the graph build) provided by a `CUDA` package extension.

## Implemented

| Routine | Status |
| --- | --- |
| `build_tree` / `query_preceding_neighbors` / `compute_depths` / `order_by_depth` / `build_graph` | ✅ graph construction in Julia (CPU; GPU build in progress) |
| `check_graph` | ✅ graph-invariant validator |
| `generate` / `generate_inv` / `generate_logdet` (+ `*_dense`) | ✅ forward / inverse / logdet (dense first layer + Vecchia refine) |
| `refine` / `refine!`, `refine_inv` / `refine_inv!`, `refine_logdet` | ✅ per-point kernels (CPU + GPU) |
| `compute_cov_matrix` | ✅ dense reference covariance builder |
| `refine_logdet_grad_vals`, `refine_inv_loss_grad_vals` (+ `generate_*`) | ✅ hand-written reverse-mode adjoints (CPU threaded / GPU atomic) |

The hand-written logdet gradient is cross-checked against an Enzyme-through-KA reference
(`refine_logdet_grad_vals_enzyme`) and the JAX f64 oracle.

## Differentiability

Gradients are provided by hand-written analytic adjoints (not generic autodiff), exposed as
`ChainRulesCore.rrule`s so any reverse-mode AD framework (e.g. Zygote) can compose them for an
**arbitrary scalar loss**:

- **w.r.t. the discretized covariance `cov_vals`** — for `logdet` and the inverse quadratic
  form `0.5‖xi‖²`; chains to kernel hyperparameters via `hyperparam_grad`.
- **w.r.t. the white-noise parameters `xi`** — `generate`/`refine` are linear in `xi`, so the
  VJP is exact and cheap.
- **w.r.t. point positions** — supported through a continuous (dequantized) coordinate path;
  the integer lattice is treated straight-through (gradients are w.r.t. the dequantized
  positions, consistent with the forward value).

The fast forward path is unchanged (integer lattice); the point-gradient path uses float
positions only when point derivatives are requested.

## Benchmarks (CPU)

4-core CPU, `N = 200_000`, `k = 10`, `d = 3`, f32 (median of repeated runs):

| Op | GraphGP.jl (4 threads) | JAX (CPU, jit) | speedup |
| --- | --- | --- | --- |
| `refine_logdet` | 0.44 s | 2.24 s | **5.1×** |
| `refine_inv` | 0.66 s | 2.54 s | **3.9×** |
| `refine_logdet_grad_vals` | 0.48 s | 3.02 s | **6.3×** |

The kernels never materialize the `(M, k+1, k+1)` covariance tensor. The reverse pass uses a
hand-written analytic Cholesky pullback; on CPU it accumulates into per-task private
histograms (no atomics), on GPU it scatters with atomics (the one-workitem-per-point design
targets CUDA.jl — GPU validation runs on hardware with a GPU, not available in CI here).

Reproduce: `julia -t auto --project=julia/GraphGP julia/GraphGP/test/bench.jl 200000 10 3`
and `python julia/GraphGP/test/bench_jax.py 200000 10 3`.

GPU benchmarks (GraphGP.jl vs JAX vs the custom `graphgp-cuda` reference, on an RTX A6000)
live in [`test/GPU_BENCHMARK_RESULTS.md`](test/GPU_BENCHMARK_RESULTS.md); they run from the
dedicated environment in `bench/` (`julia --project=julia/GraphGP/bench …`) so the package
itself stays CUDA-free.

## Usage

```julia
using GraphGP
prob = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals)
ld  = refine_logdet(prob)        # Σ log(std)
xi  = refine_inv(prob, values)   # recover unit-normal parameters
```

`coords` is `(D, N)` `UInt32`, `neighbors` is `(K, M)` 1-based integer indices, `bins`/`vals`
are the discretized covariance. The CPU backend is the default; a CUDA array input selects the
GPU backend automatically (KernelAbstractions).

## Validation & benchmarks

Reference inputs/outputs (and gradients) are dumped from the JAX implementation:

```bash
python julia/GraphGP/test/dump_reference.py        # writes test/reference/*.npz
julia --project=julia/GraphGP -e 'using Pkg; Pkg.test()'
```

Tests assert the f32 kernels track JAX f32 closely and stay within a bounded distance of the
f64 oracle, and that the f64 Julia path reproduces the JAX f64 oracle tightly.
