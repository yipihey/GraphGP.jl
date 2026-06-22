# GraphGP.jl

A Julia rewrite of the performance-critical numeric core of
[graphgp](../../README.md) (scalable Gaussian processes via a Vecchia /
nearest-neighbor-graph approximation), using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for
backend-agnostic CPU + GPU kernels and (planned)
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for differentiability.

The k-d tree build, neighbor query and depth ordering stay in the Python/JAX side. This
package consumes a **prebuilt graph** and owns only the hot per-point inner loop: assemble
each `(k+1)×(k+1)` covariance on the fly, factorize it, and emit the per-point scalar — the
work that dominates runtime when the number of points `M = N − n0` reaches the billions.

## Design choices

- **f32 by default.** All real arithmetic is `Float32`; a `Float64` path exists only as a
  debugging oracle.
- **Integer lattice coordinates.** Points live on a 21-bit-per-axis (`UInt32`) Morton-friendly
  lattice. Coordinate differences are exact integer subtractions; squared distances accumulate
  in `Int64` and are cast to `Float32` only at the `sqrt` → kernel-lookup step.
- **One workitem per point.** With `k ≈ 10` the per-point matrix is ~`11×11`, kept in private
  memory; the full `(M, k+1, k+1)` tensor is never materialized (the memory win over JAX).
- **Gradients w.r.t. `cov_vals` only.** Coordinates are fixed integer data, so the only
  differentiable input is the discretized covariance (the kernel hyperparameters).

## Implemented (validated against JAX on CPU)

| Routine | Status |
| --- | --- |
| `refine_logdet` | ✅ forward |
| `refine_inv` / `refine_inv!` | ✅ forward |
| `refine` / `refine!` | ✅ forward generation (sequential depth-batch scan) |
| `refine_logdet_grad_vals` | ✅ hand-written reverse-mode adjoint (CPU threaded / GPU atomic) |
| `refine_inv_loss_grad_vals` | ✅ Enzyme gradient of `0.5‖xi‖²` |

The hand-written logdet gradient is cross-checked against an Enzyme-through-KA reference
(`refine_logdet_grad_vals_enzyme`) and the JAX f64 oracle.

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
