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
  extension is launch-overhead-bound below ~5 M points).
- **Forward + both derivatives now run on-device** (per-batch kernels in one CUDA stream — no
  per-batch host sync, no host-bound reverse sweep). On the shallow heap-order graph (§7),
  K=8, A6000: **at 1 M GraphGP.jl beats the CUDA extension on all three** — forward 6.2 ms vs
  13.7 ms (2.2×), d/dxi 6.5 ms vs 15.7 ms (2.4×), d/dcov_vals 41 ms vs 84 ms (2.0×). **At 10 M
  it wins d/dxi** (76 ms vs 120 ms) but trails on forward (70 ms vs 63 ms, ~12 %) and d/dcov_vals
  (689 ms vs 388 ms) — the extension fuses the heavier per-point reverse better at scale (§3).
- **Native, Python-free graph build — byte-identical to Python.** `build_graph` now constructs the
  JAX "special order" heap-layout k-d tree and preceding-neighbour query in pure Julia, producing
  a graph **identical** to `gp.build_graph` (permutation, split dims, every neighbour, depth
  offsets) on float clouds, and shallow (≈tens of depth batches, not thousands) — which is what
  makes the on-device sweeps fast (§7).
- **Differentiation is also portable + analytically composable.** The same analytic adjoints (xi,
  cov_vals, log-det, inverse-loss, point positions) compose through Julia's ChainRules/Zygote and
  run on CPU **and** GPU — a surface the GPU-only extension does not fully cover (§4).
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
| `generate_grad_vals` GPU vs CPU (device reverse sweep) | Float32 | 6.4e-6, all finite |
| `build_graph` vs Python `gp.build_graph` (perm, split dims, neighbours, offsets) | — | exact (0/22976) |
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
| forward `generate` | 142 M/s (70.5 ms) | **158 M/s** (63.2 ms) |
| d/dxi (analytic, on-device) | **131 M/s (76.3 ms)** | 84 M/s (119.6 ms) |
| d/dcov_vals (analytic, on-device) | 14.5 M/s (689 ms) | **26 M/s (387.5 ms)** |

Three findings at this scale (all on-device now — the depth-batch apply and the reverse sweep
were moved off the host): (1) the per-point parallel ops are very fast; (2) GraphGP.jl **wins
d/dxi** (1.6×); (3) it trails the extension on forward `generate` (~12 %) and on d/dcov_vals
(1.8×) — the extension fuses the heavier per-point reverse (Cholesky-vals backward + histogram
scatter) better at 10 M. Closing those two at scale is the remaining lever; at 1 M GraphGP.jl is
ahead on all three (§3).

Peak GPU memory at 10 M, K=8 (all sub-GB, comfortably within 48 GB): GraphGP.jl forward 700 MB,
d/dxi 400 MB, d/dcov_vals 360 MB; CUDA-ext forward 1.0 GB.

## 3. Forward-pass derivatives — speed (A6000, 1 M, K=8, shallow heap-order graph)

Both forward derivatives run on-device (per-batch reverse kernels, latest-first, in one CUDA
stream; dense first layer on the host). At 1 M GraphGP.jl is **faster than the CUDA extension on
all three** measured on the same machine:

| op | graphgp-cuda | GraphGP.jl | speedup |
| --- | --- | --- | --- |
| forward `generate` | 13.7 ms | **6.2 ms** | 2.2× |
| d/dxi (white-noise → field) | 15.7 ms | **6.5 ms** | 2.4× |
| d/dcov_vals (kernel derivative) | 84.4 ms | **41.2 ms** | 2.0× |

GraphGP.jl also uses less device memory on the derivatives (it keeps only `mean_vec`/`std` and the
recomputed field on the GPU; ~40 MB vs the extension's ~110 MB at 1 M — but the two metrics are
measured differently, so treat MB as indicative). The crossover with the extension is at scale:
by 10 M the extension retakes forward and d/dcov_vals while GraphGP.jl keeps d/dxi (§2).

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
matches FD to ~1e-9, the device sweep matches the CPU path to 6.4e-6 in f32, composes through
Zygote). The honest standing on derivatives: same results, **on-device and faster than the
extension at 1 M** (§3), competitive-to-faster at 10 M (wins d/dxi, trails d/dcov_vals; §2), and
available on CPU too and composable with Julia AD. The pure-JAX inverse pass differentiates `xi`
but not yet `cov_vals`.

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

## 7. Graph build — native, Python-free, byte-identical, shallow

`build_graph` builds the dependency graph entirely in Julia (no JAX/Python). It ports the JAX
reference (`tree.py`): the "special order" heap-layout k-d tree (`build_tree_special`, where a
point's array position equals its heap node id, `left(c)=c+2^L`, `right(c)=c+2·2^L`) and the exact
stackless preceding-neighbour query (`query_preceding_neighbors_special`). The result is
**byte-identical to `gp.build_graph`** (pure-JAX path) on float clouds — permutation, split dims,
every neighbour entry, depth offsets and batch count all match (validated 0/22976 at N=3000, k=8).

Why it matters for the numbers above: the heap order puts the spread tree medians first, so the
Vecchia DAG is **shallow** — ≈tens of depth batches instead of the thousands a leaf-order tree
produces. The on-device forward and reverse sweeps issue one kernel per batch in a single stream,
so few batches ⇒ few launches ⇒ the 1 M wins in §3. The build is the global per-level form
(matching JAX's even-by-id level fill, which does not decompose into recursive median); it is a
one-time CPU cost (~74 s at 1 M, D=3) — moving it on-device is a possible future lever, but it is
off the hot path. `build_graph_ka` remains the fully-on-device build (valid Vecchia graph, but not
byte-identical to JAX — sort tie-breaking differs).

## 8. Notes / honest caveats

- **Derivatives are on-device.** `generate_grad_xi` / `generate_grad_vals` run their reverse
  depth-batch sweep on the GPU (per-batch kernels, latest-first, one stream; dense first layer on
  the host). Faster than the extension at 1 M on all three; at 10 M GraphGP.jl wins d/dxi but the
  extension retakes forward (~12 %) and d/dcov_vals (1.8×) — fusing the heavier per-point reverse
  at scale is the open lever.
- **Anisotropic kernel.** `K(Δspatial, Δz)` (observed-coordinate clustering) is a forward-only
  drop-in matching the Python fork to ~1e-15; same throughput characteristics as the isotropic path.

## Reproduce

```bash
# GPU forward throughput vs the CUDA extension (one shared real graph)
python test/bench_realgraph.py 5000000 10 3 /tmp/rg.npz
julia --project=bench test/bench_realgraph.jl /tmp/rg.npz

# forward + d/dxi + d/dcov_vals: CUDA-extension baseline (time + peak GPU memory), K=8
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  .venv-gpu/bin/python bench/compare/run_gradmem.py 1000000 8
julia --project=bench bench/compare/run_gradmem.jl 200000 8  # GraphGP.jl side

# build_graph byte-identity vs Python's pure-JAX gp.build_graph (needs jax + graphgp)
# python -c "...gp.build_graph(pts,n0=128,k=8); np.savez(...)"  then compare in Julia (see §7)

# CPU throughput (NUMA-pinned)
julia -t 128 --project=bench test/bench_cpu_pin.jl numa 2000000

# correctness: full suite (CPU + auto GPU testset) + opt-in Python parity
julia --project=. -e 'using Pkg; Pkg.test()'
```

See also [`test/GPU_BENCHMARK_RESULTS.md`](../test/GPU_BENCHMARK_RESULTS.md) (full 0.2 M–20 M
sweep + graph construction) and [`bench/compare/`](../bench/compare/) (the harness).
