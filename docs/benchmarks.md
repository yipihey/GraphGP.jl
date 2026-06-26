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
- **Forward pass + both derivatives, on-device, all the way to GPU-fill.** The CUDA extension *does*
  differentiate the forward `generate` (w.r.t. `xi` **and** `cov_vals`, on-GPU) — a real head-to-head.
  Across 10 M–240 M (K=8, A6000; §2) GraphGP.jl **wins both derivatives** — d/dxi ~1.4–1.6×,
  d/dcov_vals ~1.4–1.6× — while the extension wins forward `generate` by ~1.2–1.3×. **At 240 M
  (filling the 44 GB A6000) GraphGP.jl computes the full forward + both derivatives, while the
  extension's combined autodiff run OOMs** — its analytic adjoints are tape-free and reuse buffers.
  The extension also has **no** gradient rule for the log-det / inverse-loss ops, which GraphGP.jl
  adds (§4). (At 1–2 M, below target scale, GraphGP.jl leads all three — the launch-overhead regime.)
- **Native, Python-free graph build — byte-identical to Python; fast on GPU at scale.** The pure-Julia
  `build_graph` reproduces `gp.build_graph` exactly (permutation, split dims, neighbours, offsets) and
  shallow, but is CPU-only/gather-bound past ~10 M. The optional **`build_graph_cuda`** (the
  hand-written shallow pipeline via the accelerator library) closes that gap — **at parity with JAX's
  `cuda=True` GPU build** (4.6 vs 4.1 s at 10 M, 55 vs 49 s at 80 M) and shallow (≈hundreds of
  batches), producing the same graph (§5). Portable fallbacks remain (`build_graph` CPU,
  `build_graph_ka` GPU); the distributed build for billions is future work (`docs/scaling-100B.md`).
- **Differentiation is also portable + analytically composable.** The same analytic adjoints (xi,
  cov_vals, log-det, inverse-loss, point positions) compose through Julia's ChainRules/Zygote and
  run on CPU **and** GPU — a surface the GPU-only extension does not fully cover (§4).
- **CPU: dramatically faster than pure-JAX.** On the same 64 cores and the same graph, GraphGP.jl
  beats pure-JAX-CPU by **46–84×** on the per-point ops and **16–23×** on `build_graph` (§5) — the
  fused per-point kernel avoids JAX's `(M,k+1,k+1)` materialisation and scales near-linearly with
  cores. The whole pipeline runs on CPU with no Python/CUDA dependency.
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

### At scale — the forward pass + its derivatives (the operating point we care about)

The interesting regime is **tens to hundreds of millions of points — graphs that fill the GPU** —
not 1–2 M (the CUDA extension's launch-overhead regime). The forward pass is also exactly where the
extension *has* derivatives (`jax.grad` through `generate(cuda=True)` w.r.t. `xi` **and** `cov_vals`),
so all three are a genuine head-to-head. A6000 (44 GB usable), K=8, D=3, Float32, wall-clock ms
(lower is better); both built on-GPU (GraphGP.jl `build_graph_cuda`, the extension `cuda=True`):

| N | forward `generate` (jl / cu) | d/dxi (jl / cu) | d/dcov_vals (jl / cu) |
| --- | --- | --- | --- |
| 10 M | 68 / **63** | **74** / 120 | **232** / 388 |
| 40 M | 281 / **234** | **317** / 475 | **872** / 1395 |
| 80 M | 569 / **462** | **653** / 949 | **1831** / 2687 |
| 160 M | 1155 / **920** | **1339** / 1905 | **3649** / 5236 |
| **240 M** (fills 44 GB) | 1693 / **1378** | **2021** / OOM† | **5487** / OOM† |

At every scale GraphGP.jl **wins both derivatives** — d/dxi by ~1.4–1.6× and d/dcov_vals by ~1.4–1.6×
— while the extension wins the forward `generate` by ~1.2–1.3× (it fuses the sequential depth-batch
apply tighter). The forward `generate`, d/dxi, and d/dcov_vals now use the same **fused** per-batch
form as the extension (assemble → factorise → solve → apply inline; no materialised `mean_vec`); at
GPU-fill scale this is at-parity in speed with the previous two-pass form (≤1 % at 80 M / 160 M) and
~10 % slower only at low N (≤40 M), where the small early depth-batches underfill the GPU.

**Footprint (per-point transient, measured at 40 M).** Fusing the mean/std computation into each
per-batch apply removes the materialised conditional-mean weights `mean_vec` (K×M = 32 B/pt at K=8)
plus `std`, roughly **halving every op's transient**:

| op | before (2-pass) | after (fused) |
| --- | --- | --- |
| forward `generate` | 2.61 GiB (65 B/pt) | **1.27 GiB (29 B/pt)** |
| d/dxi | 2.76 GiB (69 B/pt) | **1.42 GiB (35 B/pt)** |
| d/dcov_vals | 2.76 GiB (69 B/pt) | **1.42 GiB (35 B/pt)** |

The persistent graph (`prob`) is 44 B/pt (coords UInt32 + Int32 neighbours at K=8); the working set
during any op is now `prob + ~35 B/pt` (was `prob + ~69 B/pt`). Combined with a leaner `build_graph_cuda`
(below), this raises the **full-triplet GPU-fill ceiling from 240 M → 440 M (1.83×)** on a free 44 GB
A6000: 360 M / 400 M / 440 M all compute the full forward + both derivatives (free-after 11.6 / 9.0 /
5.5 GB), 480 M OOMs. At a fixed 240 M the run leaves **17.4 GB free (was 0.7 GB)** — the ~16.7 GB the
fused ops no longer hold. (Probe each N in its own process or with a full `CUDA.reclaim()` between
runs; a back-to-back sweep in one process can OOM early purely from pool fragmentation.)

Two changes got there, both leaving the per-point gather/compute kernels — and thus throughput —
untouched:
1. **Fused ops** (above): no materialised `mean_vec`, so each op's transient roughly halves.
2. **Leaner build.** `build_graph_cuda` no longer duplicates the 32 B/pt `neighbors` array for the
   0→1-based shift (`neighbors .+= 1` in place — was a full 9.6 GB copy at 300 M), permutes the input
   in place (`permutedims!`, no temp), and frees the input/scratch copies (`pts_dn`, `temp`, `depths`,
   intermediate coords) as soon as they are dead instead of letting them coexist with the persistent
   arrays. All build-time only.

Shrinking the *persistent* `neighbours` further is not free: Int32 is already the minimum width for
global indices at this N, and delta/Int16 encodings would add indirection to the `_gather_joint!`
hot path, so they are deliberately not done.

**At GPU-fill, GraphGP.jl computes the full forward + both derivatives, but the
extension's combined run OOMs on the derivatives.** † A *single* extension derivative fits at 240 M
in isolation, but running forward + d/dxi + d/dcov_vals in one process accumulates memory (jax.grad
materialises autodiff tapes) and OOMs; GraphGP.jl's **tape-free analytic adjoints** reuse buffers, so
the whole triplet fits. So GraphGP.jl reaches higher usable N for the gradient workflow that fitting
actually needs. (Forward-only, the extension's lighter footprint — ~88 B/pt — would go past 240 M.)

The fully-parallel per-point ops scale even better at K=8 (smaller `(k+1)` blocks → higher
occupancy): `refine_logdet` ~398 M/s, `refine_inv` ~327 M/s at 10 M.

## 3. Small-N reference (1 M, K=8) — below target scale

For completeness at small N (the extension's launch-overhead regime — **not** the operating point
we optimise for; see §2 for the at-scale comparison). Both forward derivatives run on-device
(per-batch reverse kernels, latest-first, one CUDA stream; dense first layer on the host). At 1 M
GraphGP.jl leads on all three because the extension is launch-bound here:

| op | graphgp-cuda | GraphGP.jl | speedup |
| --- | --- | --- | --- |
| forward `generate` | 13.7 ms | **6.2 ms** | 2.2× |
| d/dxi (white-noise → field) | 15.7 ms | **6.5 ms** | 2.4× |
| d/dcov_vals (kernel derivative) | 84.4 ms | **41.2 ms** | 2.0× |

These small-N margins shrink and partly invert as N grows into the launch-amortised regime (§2):
at 10–40 M GraphGP.jl keeps the win on **both derivatives** (~1.5–1.6×) but the extension edges
ahead on the forward `generate` (~1.1–1.2×).

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

## 5. CPU path — GraphGP.jl (multithreaded) vs JAX (CPU)

The same fused kernel runs on the CPU backend via a native `@threads` path (the KernelAbstractions
CPU backend is 5–13× slower; bypassed). Apples-to-apples vs pure-JAX on **one shared graph**,
2-socket EPYC 7763, **2 M points, K=10, D=3, Float32**, both confined to the **same 64 cores**
(`taskset -c 0-63`). Per-point throughput in M points·s⁻¹ (higher is better):

| op | jax-cpu (64) | GraphGP.jl (64) | speedup vs JAX |
| --- | --- | --- | --- |
| `refine_logdet` | 0.2 | **13–19** | ~65–95× |
| `refine_inv` | 0.1 | **12–18** | ~120–180× |
| grad (∂logdet/∂cov_vals) | 0.1 | **6.5–9.6** | ~65–95× |

GraphGP.jl-CPU throughput at this N has **run-to-run variance** (NUMA placement + machine load give
~13–19 M/s for `refine_logdet` @64 cores across runs); the robust result is the order-of-magnitude
win over pure-JAX. 128 cores adds ~1.3–1.5× more. Reproduce with `bench/compare/run_all.sh`.

The gap is structural, not tuning: pure-JAX materialises the full `(M, k+1, k+1)` tensor and calls a
batched Cholesky (memory-bound, poorly multithreaded on CPU — it scales only ~5× across cores),
whereas GraphGP.jl keeps each `(k+1)²` block in registers in a fused per-point kernel and scales
near-linearly with cores. (Same answer to f64 round-off — see §1.)

**Graph build (CPU), `build_graph` vs JAX `gp.build_graph`** (steady-state, excludes JAX JIT
compile), seconds (lower is better):

| N | jax-cpu | GraphGP.jl (128 threads) | speedup |
| --- | --- | --- | --- |
| 1 M | 41.1 s | **2.5 s** | 16× |
| 2 M | 79.6 s | **3.5 s** | 23× |

The Julia build is parallelized (threaded per-level passes + a parallel merge-sort) and is
**byte-identical** to `gp.build_graph` (§7). This is the only fused CPU implementation of the core;
the whole pipeline (build → refine → gradients) runs on CPU with no Python/CUDA dependency.

### Build at high N — fast shallow on-GPU build (`build_graph_cuda`)

The fast path at scale is **`build_graph_cuda`** (optional accelerator, `using CUDA` + the built
`csrc/libgraphgpcapi.so`): the same hand-written shallow pipeline as `gp.build_graph(cuda=True)`
(build_tree heap order → preceding k-NN → depths → order-by-depth), returning a GPU
`GraphGPProblem` ready for `generate`. Build to a usable graph, A6000 / EPYC 7763, K=8, seconds:

| N | **`build_graph_cuda`** (GPU, shallow) | JAX `cuda=True` (GPU, shallow) | `build_graph` (CPU, shallow) | `build_graph_ka` (GPU) |
| --- | --- | --- | --- | --- |
| 10 M | **4.6** (112 b) | 4.1 | 29 (109 b) | 13 (6954 b) |
| 20 M | **10.1** (120 b) | 9.5 | 82 | 31 (8844 b) |
| 40 M | **22.7** (126 b) | 22 | 1735 ⚠ | 77 (11273 b) |
| 80 M | **54.8** (132 b) | 49 | — (CPU-bound) | 190 (14407 b) |

`build_graph_cuda` is **at parity with JAX's GPU build** (~10 % wrapper overhead from the Julia-side
quantize/transpose) and **shallow** (≈hundreds of batches, not thousands) — it produces the *same*
graph as the CPU `build_graph` (identical batch count, neighbours and logdet to f32). This **closes
the build gap**: previously the only shallow build was CPU-only and gather-bound (fine to ~10 M,
then a memory cliff by 40 M), and the on-GPU `build_graph_ka` was both ~3× slower *and* deep
(sort-based leaf order → thousands of batches). Recipe at scale: `build_graph_cuda` to build, then
the winning GraphGP.jl forward + derivatives (§2) — all on-device. (For billions of points across
multiple GPUs, the distributed build remains future work; see `docs/scaling-100B.md`.) The portable
fallbacks — `build_graph` (CPU, no deps) and `build_graph_ka` (GPU, KA-only) — stay available when
the accelerator library isn't built.

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

## 8. Optional custom-CUDA path — and why it is not a win here

GraphGP.jl can call the original `graphgp_cuda` hand-written kernels directly (an `extern "C"`
bridge in `csrc/`, the `GraphGPCUDAExt` extension, `refine_logdet_custom` / `refine_inv_custom`).
We added it to chase the ~2× that hand-CUDA bought on the hydro solvers — it does **not** transfer.
A6000, Float32, D=3, custom-kernel-only vs KA throughput:

| op | k=8 | k=10 | k=16 |
| --- | --- | --- | --- |
| `refine_logdet` KA | **415 M/s** | **257 M/s** | 74 M/s |
| `refine_logdet` custom | 365 M/s | 219 M/s | 74 M/s |
| `refine_inv` KA | **352 M/s** | **210 M/s** | 65 M/s |
| `refine_inv` custom | 291 M/s | 177 M/s | 62 M/s |

Same answer (validated ~1–7e-6 f32). These per-point kernels are register/occupancy-bound, so there
is little for a hand-written kernel to exploit, and the KA path **specializes on the exact `k`**
(`Val(K)`) while the vendored kernel pads to `MAX_K ∈ {4,8,16,32,64}` (hence KA's edge at
non-power-of-two `k`, parity at large `k`). The portable path is already at/above the hand-written
reference. The bridge stays as a cross-check + a drop-in point for future hand-tuned kernels — see
[`csrc/README.md`](../../csrc/README.md).

## 9. Notes / honest caveats

- **Derivatives are on-device.** `generate_grad_xi` / `generate_grad_vals` run their reverse
  depth-batch sweep on the GPU (per-batch kernels, latest-first, one stream; dense first layer on
  the host). At target scale (10–40 M) GraphGP.jl wins **both** derivatives (d/dxi ~1.5–1.6×,
  d/dcov_vals ~1.6×); the CUDA extension wins only the forward `generate` (~1.1–1.2×). The
  extension differentiates the forward generate (xi, cov_vals) too, but has no log-det/inverse-loss
  gradient rule (§4).
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

# §5 CPU vs JAX — per-point ops on ONE shared graph (correctness + throughput), matched cores.
# Portable: set PY to a jax+graphgp python; CORES/NTHREADS to your machine. GPU=off for CPU-only.
PY=/path/to/jax-python GPU=off ./bench/compare/run_all.sh 2000000 10 3 64   # prints results/report.md

# §5 CPU vs JAX — build_graph wall-time (GraphGP.jl parallel vs gp.build_graph), matched cores.
PY=/path/to/jax-python ./bench/compare/run_build_compare.sh 1000000 10 3 64

# CPU throughput (NUMA-pinned, GraphGP.jl only)
julia -t 128 --project=bench test/bench_cpu_pin.jl numa 2000000

# correctness: full suite (CPU + auto GPU testset) + opt-in Python parity
julia --project=. -e 'using Pkg; Pkg.test()'
```

See also [`test/GPU_BENCHMARK_RESULTS.md`](../test/GPU_BENCHMARK_RESULTS.md) (full 0.2 M–20 M
sweep + graph construction) and [`bench/compare/`](../bench/compare/) (the harness).
