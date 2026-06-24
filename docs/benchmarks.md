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
- **GPU per-point ops: at parity with the CUDA extension at scale, faster at small/medium N** (the
  extension is launch-overhead-bound below ~5 M points). The *full* `generate` forward is the
  exception — its host-orchestrated sequential depth-batch apply is ~2.5× slower than the
  extension at 10 M (§2); moving that to the device is the next lever.
- **Differentiation: GraphGP.jl is portable, not faster.** Both the CUDA extension and pure-JAX
  differentiate the forward pass w.r.t. `xi` **and** `cov_vals` — and the CUDA extension does so
  on-GPU and fast (d/dxi 16 ms, d/dcov_vals 82 ms at 1 M). GraphGP.jl provides the same analytic
  adjoints (plus log-det/inverse-loss/point gradients), composable through Julia's ChainRules/
  Zygote and running on CPU **and** GPU — but its GPU reverse sweep is currently **host-bound**,
  so it is ~5× slower on d/dxi and much slower on d/dcov_vals than the extension (§4). GraphGP.jl's
  value is portability + analytic composability, **not** derivative speed; closing the host-bound
  reverse sweep is an open lever.
- **CPU.** A native multithreaded CPU path (the same fused kernel) reaches ~20 M pts/s — the CUDA
  extension is GPU-only, so this is GraphGP.jl's CPU story, not a head-to-head.
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

`refine_logdet_grad_vals` (gradient of the *log-det* w.r.t. the covariance table): **30–37
M pts·s⁻¹** in GraphGP.jl across 200 K–20 M. (The CUDA extension differentiates the forward
*generate* on-GPU — see §3/§4 — but its v0.0.4 had no autodiff rule for the log-det op
specifically; pure-JAX can but is ~1000× slower.)

### Scaling to 10 M points, K=8

K=8 has smaller `(k+1)` blocks → less private memory → higher GPU occupancy, so the fully-parallel
ops scale even better. A6000, Float32:

| op @ 10 M, K=8 | GraphGP.jl | CUDA extension |
| --- | --- | --- |
| `refine_logdet` | **398 M/s** | — |
| `refine_inv` | **327 M/s** | — |
| forward `generate` | 63 M/s (158 ms) | **159 M/s** (63 ms) |
| d/dxi (analytic) | 13 M/s (772 ms) ‡ | none (no autodiff) |
| d/dcov_vals (analytic) | 0.25 M/s (40 s) ‡ | none (no autodiff) |

Two honest findings at this scale: (1) the per-point parallel ops (`refine_logdet`/`refine_inv`)
are very fast; (2) **forward `generate` is ~2.5× *slower* than the CUDA extension** — it is bound
by the host-orchestrated sequential depth-batch apply (and the dense first layer), not the
per-point kernel. That host orchestration is the same thing that makes the analytic adjoints
(‡) host-bound: low memory but CPU-bound wall time. Moving the depth-batch apply and the reverse
sweep onto the device is the clear next performance lever.

Peak GPU memory at 10 M, K=8 (all sub-GB, comfortably within 48 GB): GraphGP.jl forward 700 MB,
d/dxi 400 MB, d/dcov_vals 360 MB; CUDA-ext forward 1.0 GB.

## 3. Forward-pass derivatives — speed and memory (A6000, 1 M, K=8)

The CUDA extension differentiates the forward pass on-GPU; GraphGP.jl differentiates it
analytically but with a **host-bound reverse sweep**. So the extension is faster here, while
GraphGP.jl uses less device memory (it keeps only `mean_vec`/`std` on the GPU). Time / peak:

| op | graphgp-cuda | GraphGP.jl |
| --- | --- | --- |
| d/dxi (white-noise → field) | **15.8 ms**, ~109 MB | 78 ms (host-bound), ~40 MB |
| d/dcov_vals (kernel derivative) | **82 ms**, ~118 MB | 4.1 s (host-bound), ~36 MB |

The memory metrics differ (extension = JAX `peak_bytes_in_use`; GraphGP.jl = CUDA.jl per-call
`@allocated`), so the MB are indicative, not a strict head-to-head. The clear story is *speed*:
GraphGP.jl's reverse depth-batch sweep runs on the host (it is sequential across the ~depth
batches), so the GPU sits idle — ~5× slower on d/dxi and far slower on d/dcov_vals than the
extension's on-GPU derivative. Moving the reverse sweep onto the device is the open lever (and
it interacts with the depth-batch count — see §6).

## 4. The differentiability surface

Both the CUDA extension and pure-JAX differentiate the forward pass w.r.t. `xi` and `cov_vals`.
GraphGP.jl matches that surface with hand-written **analytic** adjoints (plus log-det,
inverse-loss, and point-position gradients), composable through Julia's ChainRules/Zygote and
running on CPU and GPU. So GraphGP.jl's contribution is **portability + analytic composability in
Julia**, not a capability the others lack.

| derivative | GraphGP.jl | graphgp-cuda | pure-JAX |
| --- | --- | --- | --- |
| ∂ generate / ∂ xi | ✅ analytic | ✅ (on-GPU, fast) | ✅ autodiff |
| ∂ generate / ∂ cov_vals | ✅ analytic (new) | ✅ (on-GPU, fast) | ✅ autodiff |
| ∂ log-det / ∂ cov_vals (+ hyperparameters) | ✅ analytic | (forward-pass rules; log-det rule absent in v0.0.4) | ✅ autodiff |
| ∂ ½‖generate_inv‖² / ∂ cov_vals | ✅ analytic | — | ⚠️ inverse autodiff is `xi`-only |
| ∂ (log-det, inverse-loss) / ∂ point positions | ✅ analytic | — | ✅ autodiff |

GraphGP.jl's adjoints are validated against finite differences and JAX (`generate_grad_vals`
matches FD to ~1e-9, composes through Zygote). The honest standing on derivatives: same results,
**slower than the extension on GPU** (host-bound reverse sweep, §3), but available on CPU too and
composable with Julia AD. The pure-JAX inverse pass differentiates `xi` but not yet `cov_vals`.

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
