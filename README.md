# GraphGP.jl

[![CI](https://github.com/yipihey/GraphGP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/yipihey/GraphGP.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Scalable Gaussian processes via a Vecchia / nearest-neighbor-graph approximation — a standalone
Julia rewrite of [graphgp](https://github.com/yipihey/graphgp) (JAX). One fused per-point kernel
runs on **CPU and GPU** from a single code path ([KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)),
the full pipeline (k-d tree → neighbor query → depth ordering → generation) is **pure Julia with
no Python/JAX/C++ dependency**, and every step is **differentiable** through hand-written analytic
adjoints exposed as [ChainRules](https://github.com/JuliaDiff/ChainRulesCore.jl) `rrule`s.

The `(M, k+1, k+1)` covariance tensor is never materialized — each `(k+1)×(k+1)` block is built,
factorized, and consumed in registers — so the method scales to billions of points where the
materializing JAX path OOMs.

## 📊 Benchmarks & comparison

**Full numbers, methodology, and correctness cross-checks:**
[**`docs/benchmarks.md`**](docs/benchmarks.md) · [**`docs/comparison_with_jax.md`**](docs/comparison_with_jax.md)

Headline, on one **identical** graph, RTX A6000, Float32 (lower is better):

| 1 M points, k=8 | GraphGP.jl | graphgp CUDA ext | pure JAX |
| --- | --- | --- | --- |
| forward `generate` | **6.2 ms** | 13.7 ms | OOM-prone |
| ∂/∂xi | **6.5 ms** | 15.7 ms | autodiff (slow) |
| ∂/∂cov_vals | **41 ms** | 84 ms | autodiff (slow) |

- **Same answer.** Reproduces the JAX reference element-wise: ~1e-13 (Float64), ~1e-5 (Float32 GPU).
- **`build_graph` is byte-identical to Python's** `gp.build_graph` (permutation, neighbors, offsets).
- **Faster than the hand-written CUDA extension at 1 M** on all three (2–2.4×); competitive at 10 M
  (wins ∂/∂xi). The CUDA extension has **no autodiff**; GraphGP.jl differentiates on CPU *and* GPU.
- **CPU:** the same fused kernel does ~20 M points/s multithreaded — the CUDA extension is GPU-only.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/yipihey/GraphGP.jl")
```

## Quickstart

```julia
using GraphGP, Random

N, D, n0, k = 10_000, 2, 100, 10
points = randn(MersenneTwister(99), N, D)                       # (N, D)

# Discretized covariance table (RBF), then build the dependency graph.
bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 10.0, 1000; jitter = 1e-5)   # variance, scale, r_min, r_max, n_bins
prob = build_graph(points, n0, k, bins, vals)                  # k-d tree + k-NN + depth order, in Julia

xi = randn(MersenneTwister(7), N)                              # unit-normal parameters
values = generate(prob, xi)                                    # draw a GP realization
@assert !any(isnan, values)

xi_back = generate_inv(prob, values)                           # exact inverse
logdet  = generate_logdet(prob)                                # log-determinant of the implied covariance
```

## Coming from Python `graphgp`?

The API mirrors the JAX package, with one deliberate difference: **the covariance table is folded
into the graph object** (`build_graph`), so `generate`/`generate_inv`/`generate_logdet` don't take
a separate `covariance` argument.

| Python `graphgp` | GraphGP.jl |
| --- | --- |
| `gp.extras.rbf_kernel(variance=, scale=, r_min=, r_max=, n_bins=, jitter=)` | `rbf_kernel(variance, scale, r_min, r_max, n_bins; jitter=)` |
| `graph = gp.build_graph(points, n0=, k=)` | `prob = build_graph(points, n0, k, bins, vals)` |
| `gp.generate(graph, cov, xi)` | `generate(prob, xi)` |
| `gp.generate_inv(graph, cov, values)` | `generate_inv(prob, values)` |
| `gp.generate_logdet(graph, cov)` | `generate_logdet(prob)` |
| `gp.check_graph(graph)` | `check_graph(prob)` |
| `gp.build_graph(..., cuda=True)` (GPU) | pass `CuArray` points, or `to_backend(prob, CUDABackend())` |
| *(no gradient on the CUDA extension)* | `refine_logdet_grad_vals`, `generate_grad_xi`, `generate_grad_vals`, `hyperparam_grad`, … |

The same script, side by side:

```python
# Python (JAX)
import graphgp as gp
graph = gp.build_graph(points, n0=100, k=10)
cov   = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=10.0, n_bins=1000, jitter=1e-5)
values = gp.generate(graph, cov, xi)
```

```julia
# Julia (GraphGP.jl)
using GraphGP
bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 10.0, 1000; jitter = 1e-5)
prob   = build_graph(points, 100, 10, bins, vals)
values = generate(prob, xi)
```

Conventions: `points` is `(N, D)`; internally the graph stores `coords` as `(D, N)` `UInt32`
(a 21-bit-per-axis integer lattice) and `neighbors` as `(k, M)` 1-based indices. A `Matern` kernel
(`matern_kernel`) and an anisotropic kernel `K(Δspatial, Δz)` (`build_anisotropic_covariance`) are
also provided.

## Using GraphGP.jl from Python / JAX

GraphGP.jl is callable from Python through [juliacall](https://github.com/JuliaPy/PythonCall.jl),
which is the recommended way to use it from a JAX workflow.

```bash
pip install juliacall
```

```python
from juliacall import Main as jl
jl.seval('import Pkg; Pkg.add(url="https://github.com/yipihey/GraphGP.jl")')   # once
jl.seval("using GraphGP")
```

Two array conventions to remember:

- **2-D point arrays** must be handed over as a Julia `Matrix` —
  `jl.Matrix(np.asfortranarray(points))` (`build_graph` dispatches on a concrete column-major
  matrix; a raw NumPy/JAX 2-D array won't match).
- **Vectors** (`xi`, values) pass directly, but their dtype must match the problem's element type
  (`Float64` if you built the table with `rbf_kernel(1.0, …)`); wrap **outputs** in `np.asarray`.

```python
import numpy as np
import jax.numpy as jnp
from juliacall import Main as jl
jl.seval("using GraphGP")

N, D, n0, k = 10_000, 2, 100, 10
points = np.asarray(jnp.asarray(np.random.randn(N, D)))           # (N, D)

bins, vals = jl.rbf_kernel(1.0, 0.3, 1e-4, 10.0, 1000, jitter=1e-5)
prob = jl.build_graph(jl.Matrix(np.asfortranarray(points)), n0, k, bins, vals)

xi      = np.asarray(jnp.asarray(np.random.randn(N)), dtype=np.float64)  # match prob eltype
values  = np.asarray(jl.generate(prob, xi))                       # → NumPy, feed back into JAX
back    = np.asarray(jl.generate_inv(prob, jl.Vector(values)))    # exact inverse
logdet  = float(jl.generate_logdet(prob))
```

### Inside a JAX program

`jax.jit` / `jax.grad` cannot trace through the Julia call, so wrap it with `jax.pure_callback`:

```python
import jax

def gp_generate(xi):
    return jax.pure_callback(
        lambda x: np.asarray(jl.generate(prob, np.asarray(x, np.float64))).astype(x.dtype),
        jax.ShapeDtypeStruct((N,), xi.dtype), xi)
```

For gradients, use GraphGP.jl's **analytic adjoints** instead of autodiff-through-Julia: `generate`
is linear in `xi`, so its VJP is `generate_grad_xi(prob, cotangent)`; gradients w.r.t. the
covariance table / hyperparameters come from `refine_logdet_grad_vals` + `hyperparam_grad`. Wire
these into a `jax.custom_vjp` when you need GraphGP inside a differentiated JAX program.

## What it adds over the JAX stack

- **One fused kernel, two backends.** `src/kernels.jl` (one workitem per refined point; assemble
  the `(k+1)²` block, factorize in registers, never materialize the batch) + `src/linalg.jl`
  (per-point Cholesky / solve). The CPU backend uses a native threaded path (`src/cpu_native.jl`).
- **Analytic gradients the CUDA extension lacks**, on CPU **and** GPU: ∂/∂`cov_vals`,
  ∂/∂hyperparameters (via `hyperparam_grad`), ∂/∂`xi`, and ∂/∂points. `src/chainrules.jl` exposes
  them as ChainRules `rrule`s so Zygote composes an arbitrary scalar loss.
- **Full graph construction in Julia** — `build_graph` reproduces `gp.build_graph` byte-for-byte;
  `build_graph_ka` runs tree → k-NN → depths → reorder → quantize on the device.
- **Distributed (multi-node / multi-GPU)** — a `using MPI` extension lights up `distribute` and a
  distributed log-likelihood + gradient for fitting at scale (see `bench/distributed/`).

## Differentiability

Gradients are hand-written analytic adjoints (not generic autodiff), exposed as
`ChainRulesCore.rrule`s so any reverse-mode AD (e.g. Zygote) composes them for an arbitrary scalar
loss:

- **w.r.t. the discretized covariance `cov_vals`** — for `logdet` and the inverse quadratic form
  `0.5‖xi‖²`; chains to kernel hyperparameters via `hyperparam_grad`.
- **w.r.t. the white-noise `xi`** — `generate`/`refine` are linear in `xi`, so the VJP is exact.
- **w.r.t. point positions** — through a continuous (dequantized) coordinate path.

```julia
g_vals  = refine_logdet_grad_vals(prob)                        # ∂ logdet / ∂ cov_vals
g_hyper = hyperparam_grad(g_vals, (v, s) -> rbf_kernel(v, s, 1e-4, 10.0, 1000; jitter = 1e-3),
                          [1.0, 0.3])                          # → ∂ logdet / ∂ [variance, scale]
```

See [`examples/parity_and_autodiff.jl`](examples/parity_and_autodiff.jl) for an end-to-end tour.

## GPU

Pass `CuArray` inputs (and `CUDA` in your environment) and the same calls run on the device, or
move a built problem with `to_backend(prob, CUDABackend())`. The package has no hard CUDA
dependency; GPU-specific primitives are provided by a `CUDA` package extension.

## Documentation

- [`docs/benchmarks.md`](docs/benchmarks.md) — performance, memory, correctness vs the CUDA extension.
- [`docs/comparison_with_jax.md`](docs/comparison_with_jax.md) — design + throughput vs JAX, annotated.
- [`docs/scaling-100B.md`](docs/scaling-100B.md) — design notes + API sketch for distributed (multi-node/GPU) scaling to ~100 B points.
- [`FEATURE_COVERAGE.md`](FEATURE_COVERAGE.md) — function-by-function parity audit vs Python `graphgp`.
- [`bench/compare/`](bench/compare/) — the apples-to-apples comparison harness (one shared graph).

## Development

```bash
julia --project=. -e 'using Pkg; Pkg.test()'        # CPU suite (+ GPU testset auto-runs if CUDA is present)
```

Tests assert the Float32 kernels track JAX Float32 closely, stay within a bounded distance of the
Float64 oracle, and that the Float64 Julia path reproduces the JAX Float64 oracle tightly.
Reference fixtures can be regenerated from the Python package with `python test/dump_reference.py`.

## License

MIT — see [LICENSE](LICENSE). Derived from [graphgp](https://github.com/yipihey/graphgp)
by Benjamin Dodge and Philipp Frank.
