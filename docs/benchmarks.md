# GraphGP.jl — performance, memory, and correctness

A consolidated benchmark of GraphGP.jl (the backend-agnostic KernelAbstractions + CUDA rewrite
of the Vecchia / nearest-neighbour GP core) against the **graphgp CUDA extension** — the fair,
hand-written, production-grade GPU reference. (Naive pure-JAX is not used as a baseline; it
materialises the `(M, k+1, k+1)` tensor and OOMs past ~20 M points, so it is a strawman.)

Hardware: 1× NVIDIA RTX A6000 (48 GB), 2-socket AMD EPYC 7763 (128 cores, 16 NUMA nodes).
Precision: **Float32** is the production path (the inputs, the Vecchia approximation, and the
science carry no more); correctness is cross-checked in Float64 where it clarifies.

## TL;DR

- **Same answer.** GraphGP.jl reproduces the references element-wise: ~1e-13 in Float64, ~1e-5 in
  Float32 (GPU), and matches the Python anisotropic kernel to ~1e-15.
- **GPU forward: at parity with the CUDA extension at scale, faster at small/medium N** (the
  extension is launch-overhead-bound below ~5 M points).
- **Differentiation is the decisive difference.** The CUDA extension has **no autodiff**.
  GraphGP.jl provides analytic adjoints — w.r.t. the covariance table, hyperparameters, the
  white noise, the point positions, **and now the kernel derivative of the generated field** —
  at a small peak-GPU-memory footprint (~25 MB working set), things the extension cannot compute
  at all.
- **CPU too.** A native multithreaded CPU path (the same fused kernel) reaches ~20 M pts/s — the
  only fused CPU implementation of this core.
- **Robust.** Coincident / lattice-collision points (real catalogues) stay finite on the GPU and
  agree with the CPU; degenerate blocks contribute a regularised, near-zero-information
  conditional instead of NaN.

## 1. Correctness

| check | precision | result |
| --- | --- | --- |
| `refine_logdet` / `refine_inv` vs JAX fixtures | Float64 | rtol ~1e-6 |
| GPU vs CPU forward (real graph) | Float32 | 3.9e-6 |
| anisotropic `generate` vs Python `aniso.py` | Float64 | 1.3e-15 |
| `generate`/`generate_inv` original-order vs Python | Float64 | ~2e-15 |
| `generate_grad_vals` vs finite differences | Float64 | ~1e-9 |
| degenerate (coincident) graph, GPU vs CPU | Float32 | 4e-5, all finite |

## 2. GPU forward throughput vs the CUDA extension

One real `build_graph` graph (spatially-coherent neighbours, identical points for both),
A6000, K=10, D=3, Float32. Throughput in **M points·s⁻¹** (higher is better).

`refine_logdet`:

| N | GraphGP.jl | CUDA extension |
| --- | --- | --- |
| 200 K | **172** | 32 |
| 1 M | **166** | 94 |
| 5 M | **162** | 150 |
| 20 M | 158 | **169** |

`refine_inv`:

| N | GraphGP.jl | CUDA extension |
| --- | --- | --- |
| 200 K | **169** | 30 |
| 1 M | **148** | 90 |
| 5 M | **141** | 122 |
| 20 M | **136** | 129 |

GraphGP.jl is at parity with the extension at scale and 2–5× ahead at small/medium N (where the
extension is launch-overhead-bound); the extension edges ~7% ahead only on logdet at 20 M. Both
are fused — neither materialises the per-point tensor.

`refine_logdet_grad_vals` (gradient w.r.t. the covariance table): **30–37 M pts·s⁻¹** in
GraphGP.jl across 200 K–20 M. The CUDA extension **has no autodiff rule** for this op, so there
is nothing to compare against — it simply cannot produce the gradient.

## 3. Peak GPU memory

A6000, K=30, Float32. Both forward paths are fused (no `(M,k+1,k+1)` materialisation), so both
are low-memory. The headline is the **derivatives**, which only GraphGP.jl can compute.

| op | GraphGP.jl (device working set) | CUDA extension |
| --- | --- | --- |
| forward `generate` | ~32 MB @200 K, ~158 MB @1 M | ~90 / 210 MB †, fused |
| **d/dxi** (white-noise → field derivative) | **~26 MB @200 K, ~128 MB @1 M** | **unsupported** (no autodiff) |
| **d/dcov_vals** (kernel derivative of the field) | **~25 MB @200 K** | **unsupported** (no autodiff) |

† The extension's figure is JAX's `peak_bytes_in_use`; GraphGP.jl's is CUDA.jl's per-call
`@allocated`. These are not the identical metric, so read the forward row as "both low / no
materialisation", not a precise ratio. The derivative rows are the point: GraphGP.jl differentiates
the forward pass at a few tens of MB; the extension cannot at all. (Only `mean_vec`/`std` live on
the GPU for the adjoints — the reverse sweep is on host, see §6.)

## 4. The differentiability surface

The CUDA extension is forward-only. GraphGP.jl exposes analytic adjoints (hand-written, not
autodiff-through-materialisation), composable through ChainRules/Zygote:

| derivative | GraphGP.jl | CUDA extension |
| --- | --- | --- |
| ∂ log-det / ∂ cov_vals (and hyperparameters) | ✅ analytic | ❌ |
| ∂ ½‖generate_inv‖² / ∂ cov_vals | ✅ analytic | ❌ |
| ∂ generate / ∂ xi (white noise) | ✅ analytic | ❌ |
| **∂ generate / ∂ cov_vals (kernel derivative of the field)** | ✅ analytic (new) | ❌ |
| ∂ (log-det, inverse-loss) / ∂ point positions | ✅ analytic | ❌ |

All validated against finite differences and/or JAX autodiff; the field-vals adjoint
(`generate_grad_vals`) matches FD to ~1e-9 and composes through Zygote (with the d/dxi tangent).

## 5. CPU path (native multithreaded + NUMA)

The same fused kernel runs on the CPU backend via a native `@threads` path (the KernelAbstractions
CPU backend is 5–13× slower; bypassed). EPYC 7763, 2 M points, K=10, Float32, `pinthreads(:numa)`
(round-robin across the 16 NUMA nodes — +35–59% on the gradient over unpinned). Throughput in
M points·s⁻¹:

| threads | refine_logdet | refine_inv | grad (cov_vals) |
| --- | --- | --- | --- |
| 64 | 12.8 | 12.6 | 6.4 |
| 128 | 20.4 | 20.1 | 13.1 |

This is the only fused CPU implementation of the core; it lets the whole pipeline (build → refine
→ gradients) run on CPU with no Python or CUDA dependency. (The CUDA extension is GPU-only.)

## 6. Robustness — degenerate / coincident points

Real catalogues have coincident points (group members at identical sky+redshift), and the 21-bit
lattice quantisation can collide dense points into one cell → a rank-deficient `(k+1)` conditional
block. GraphGP.jl floors the Cholesky pivot and zeroes the degenerate column (inert) and its
conditional-mean weight, so a coincident point contributes a finite, regularised, near-zero-
information conditional — **identical on CPU and GPU** (the forward field and its gradient stay
finite and agree to ~4e-5 in f32, where naive paths return NaN).

## 7. Notes / honest caveats

- **Gradient time is host-bound.** The analytic adjoints (`generate_grad_xi`, `generate_grad_vals`)
  do their reverse depth-batch sweep on the host (it is inherently sequential across batches), so
  their *peak memory* is small but their *wall time* is CPU-bound (a GPU reverse sweep is the next
  lever). Forward ops and the cov_vals log-det/inverse-loss gradients are fully on-device.
- **Anisotropic kernel.** `K(Δspatial, Δz)` (observed-coordinate clustering) is a forward-only
  drop-in matching the Python fork to ~1e-15; same throughput characteristics as the isotropic path.

## Reproduce

```bash
# GPU forward throughput vs the CUDA extension (one shared real graph)
python julia/GraphGP/test/bench_realgraph.py 5000000 10 3 /tmp/rg.npz
julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_realgraph.jl /tmp/rg.npz

# peak GPU memory of forward + derivatives (baselined on the CUDA extension)
JAX_PLATFORMS=cuda python julia/GraphGP/bench/compare/run_gradmem.py 1000000 30   # CUDA-ext forward (no autodiff)
julia --project=julia/GraphGP/bench julia/GraphGP/bench/compare/run_gradmem.jl 200000 30  # GraphGP.jl fwd + d/dxi + d/dcov_vals

# CPU throughput (NUMA-pinned)
julia -t 128 --project=julia/GraphGP/bench julia/GraphGP/test/bench_cpu_pin.jl numa 2000000

# correctness: full suite (CPU + auto GPU testset) + opt-in Python parity
julia --project=julia/GraphGP -e 'using Pkg; Pkg.test()'
```

See also [`test/GPU_BENCHMARK_RESULTS.md`](../test/GPU_BENCHMARK_RESULTS.md) (full 0.2 M–20 M
sweep + graph construction) and [`bench/compare/`](../bench/compare/) (the harness).
