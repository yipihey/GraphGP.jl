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
| gradients (`cov_vals`) | 🚧 planned (Enzyme) |
| `refine` (forward generation, sequential scan) | 🚧 planned |

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
