# What limits the per-point GPU kernels (and how to go faster)

A bottleneck analysis of the hot per-point ops (`refine_logdet`, `refine_inv`, and by extension the
forward/gradient sweeps, which run the same per-point work). Measured on an A6000 (Ampere GA102,
sm_86, 84 SMs, ~768 GB/s, ~38.7 FP32 TFLOPS), Float32, D=3. `ncu`/`nsys` are not installed here, so
this triangulates from three independent sources: the CUDA compiler report, `CUDA.@profile`, and
throughput-scaling microbenchmarks.

## TL;DR

The per-point kernels are **bound by the serial, ~O(k³) Cholesky operating on the `(k+1)×(k+1)`
covariance block that lives in *local memory*** (because it is dynamically indexed, `mat[tri(i,j)]`,
so it cannot sit in registers). They are **not** limited by global-memory bandwidth, and **not** by
register count / occupancy. The clear lever is to get that matrix off local memory — warp-cooperative
Cholesky in **shared memory** (large k), and full unrolling to a **register-resident** matrix (small
k). At small k a secondary limiter is the *uncoalesced neighbor-coordinate gather*.

## Evidence

**1. It scales like the O(k³) Cholesky, not like memory.** `refine_logdet`, N=5 M, varying k:

| k | M pts/s | ns/pt | ns/pt ÷ (k+1)² | est. global GB/s |
| --- | --- | --- | --- | --- |
| 4 | 1542 | 0.65 | 0.026 | 142 |
| 8 | 421 | 2.38 | 0.029 | 72 |
| 10 | 257 | 3.89 | 0.032 | 55 |
| 16 | 74 | 13.5 | 0.047 | 25 |
| 32 | 12 | 83.4 | 0.077 | 8 |

ns/pt grows **faster than (k+1)²** (the ratio rises 0.026→0.077) and tracks ~(k+1)³ at large k —
i.e. the per-thread Cholesky (≈(k+1)³/3 flops) dominates. The estimated global traffic *falls* to
8 GB/s — ~100× below the A6000's ~768 GB/s — so global bandwidth is nowhere near the limit.

**2. The working matrix is in local memory, and that is the cost.** Compiler report (`ptxas -v`)
for the vendored kernel, all instantiations: **40 registers, 0 spills**, but a per-thread **stack
frame that grows with MAX_K** — 328 B (MAX_K=8) → 888 B (16) → 2776 B (32), D=3. The `(k+1)²` `mat`
(and `pts`) arrays are dynamically indexed, so ptxas places them in **local memory** (off-chip,
L1/L2-cached) rather than registers. The Cholesky then streams that matrix from local memory O(k³)
times in a dependent chain.

**3. Less local memory ⇒ faster — the smoking gun.** At k=10 the portable KA kernel specializes on
the *exact* k (`Val(K)`, smaller `mat`) and hits **257 M/s**, while the vendored kernel padded to
MAX_K=16 (bigger `mat`, more local traffic) gets **219 M/s** — a ~17 % penalty purely from carrying a
larger local-memory matrix. At k=16 (both at MAX_K=16, same local size) they tie (~74 M/s). Shrinking
the local footprint directly buys speed → local-memory traffic is the binding resource.

**4. Occupancy is not the limiter.** 40 registers/thread is below Ampere's ~42.7-reg threshold for
100 % occupancy (65536 regs ÷ 1536 threads), and the kernels use no shared memory today (the matrix
is `@private`/local, not `@localmem`). So there are plenty of resident warps; the problem is that
each warp's work is a long dependent chain of local-memory loads/stores, not a shortage of warps.

**5. The kernel is ~all of it.** `CUDA.@profile` (k=10, 2 M): the per-point kernel is 99.7 % of
device time; the logdet reduction is 23 µs. Optimizing the per-point kernel is the whole game.

## Regimes

- **k ≲ 6** (1500+ M/s): short Cholesky; time shifts toward the **uncoalesced gather** of neighbor
  coordinates (scattered point indices) and fixed per-point overhead. Estimated global BW is highest
  here (142 GB/s) and — once the ~2–4× sector-overfetch from uncoalesced 12-byte gathers is folded
  in — plausibly approaches DRAM limits. Memory-latency / gather bound.
- **k ≈ 8–32** (the science range): firmly **compute/local-memory bound** by the O(k³) Cholesky on
  the local-resident `(k+1)²` matrix. This is where the headroom is.

## Optimization levers (ranked by expected payoff in the k≈8–32 range)

1. **Warp-cooperative Cholesky with the matrix in shared memory.** Assign one warp (32 lanes) per
   point; hold the `(k+1)²` block in **shared memory** (on-chip, ~100× faster than local) and
   parallelize the column updates across lanes. Removes the local-memory bottleneck *and* the serial
   chain. Shared-mem budget: (k+1)²/2 floats/point × points-per-block bounds occupancy, but at k≤31
   one-warp-per-point is natural. This is the classic batched-small-Cholesky design and the most
   promising path to the ~2× we hoped custom CUDA would give — it requires a genuinely different
   kernel, not the vendored one (which is one-thread-per-point, local-memory).
2. **Register-resident matrix for small k via full unrolling.** Fully unroll the assemble/Cholesky
   with compile-time indices so ptxas can keep small `mat` in registers (no dynamic indexing → no
   local memory). Best for k ≤ 8; complements (1).
3. **Bin on r² to drop the O(k²) `sqrt`s.** `cov_lookup` does one `sqrt` per matrix entry
   ((k+1)(k+2)/2 SFU ops). Pre-square the covariance bins and look up on squared distance to remove
   the `sqrt`s — helps the assembly-heavy small/medium-k regime (SFU runs at ¼ rate).
4. **Coalesce / stage the neighbor gather** (shared-memory staging of neighbor coords) — targets the
   small-k gather regime; modest at large k.
5. **Exact-k instantiation everywhere** (already what the KA path does; the lesson for any custom
   kernel: instantiate `MAX_K = k`, never pad to a power of two).

## Optimization results

### Lever 1 — warp-cooperative Cholesky (prototyped, benchmarked, not adopted)

Built a one-warp-per-point kernel (lane = matrix row; the `(k+1)` block in registers, shared by
`__shfl`; O(k²) critical path) in `csrc/` and benchmarked it against the KA path (A6000, N=5 M,
Float32, D=3, `refine_logdet`):

| k | KA (M/s) | warp (M/s) | warp / KA |
| --- | --- | --- | --- |
| 8 | 417 | 148 | 0.35 |
| 10 | 256 | 116 | 0.45 |
| 16 | 74 | 55 | 0.74 |
| 30 | 14 | 19 | **1.29** |

It **only wins at large k** (~1.3× at k=30) and loses badly at the common k≈8–16 because with one
warp per point only `k+1` of 32 lanes do work — the rest idle. So warp-per-point trades the
local-memory bottleneck for a lane-utilization bottleneck, and for the science range it is a net
loss. (It also needs to replicate KA's degenerate-column handling — `pivmin = eps(f32)·variance`,
floor + zero — to stay correct on the coincident points that 2²¹ lattice quantization produces in a
dense cloud; doing so faithfully tanked throughput further, confirming this is not the path.)

**Conclusion:** the KA per-point kernel is already at/above the hand-written reference for this
register/local-memory-bound workload, and warp-per-point does not beat it in the regime that
matters. A genuine win for k≈8–16 would require **multiple points per warp** (pack 32/(k+1) points
so no lane idles) — a substantially more complex kernel — and even then the ceiling is modest given
KA already matches the reference. The experiment was reverted; the thread bridge (`refine_*_custom`)
remains as the validated cross-check.

### Lever 3 — drop the O(k²) sqrt (r²-space lookup): no speedup

Added a covariance assembly that searches the *squared* bins on r² and interpolates in r²-space,
removing one `sqrt` per matrix entry. Benchmarked vs the with-sqrt thread kernel (A6000, N=5 M):

| k | thread (M/s) | no-sqrt (M/s) | no-sqrt / thread |
| --- | --- | --- | --- |
| 4 | 515 | 496 | 0.96 |
| 8 | 334 | 328 | 0.98 |
| 10 | 204 | 204 | 1.00 |
| 16 | 72 | 70 | 0.97 |

**No measurable speedup** — the kernel is local-memory/Cholesky-bound, so the `sqrt` (SFU) is already
hidden behind memory latency; removing it changes nothing. (Side result: the no-sqrt vs KA and
with-sqrt vs KA errors are *identical*, so r²-space interpolation is accuracy-equivalent to r-space
here — the ~4 % drift is the separate float-position cancellation, not the interp scheme.) Reverted.

### Levers 2, 4 (not pursued)
Lever 1 and lever 3 both confirm the kernel is bound by the local-memory-resident `(k+1)²` matrix in
a serial Cholesky, not by peripheral costs. (2) register-resident matrix via full unrolling and
(4) gather coalescing only touch small-k / the gather, where we are already fast — given the two
negative results above, they are not expected to move the needle.

## Verdict

Two of the ranked levers were implemented and benchmarked; both are negative. The portable KA
per-point kernel is **already near-optimal** for this register/local-memory-bound, occupancy-rich
workload — it matches the f64 reference to ~2e-7 and meets or beats the hand-written `graphgp_cuda`
reference. The *only* approach that could plausibly beat it in the common k≈8–32 range is a
**multi-point-per-warp** kernel (pack `32/(k+1)` points so no warp lane idles, matrix in shared
memory) — a substantially more complex kernel with a modest expected ceiling. Recommendation: keep
the KA path as the production GPU path; revisit multi-point-per-warp only if profiling on the real
science workload shows the per-point kernel is the wall-clock bottleneck.

## Host-side audit (graph build) — a 12× quick win

A second audit covered the rest of the package (host orchestration, graph build, gradients). Stage
breakdown of `build_graph` (CPU, 500k, D=3, k=10): **`build_tree_special` was 90% of the time and
allocated ~16 GB**; `query_preceding_neighbors_special` 10%; everything else (`compute_depths`,
`order_by_depth`, `quantize`) <0.2%.

**Root cause (type-unstable closure).** The per-level sort comparator captured `nodes`/`pad`, but
`nodes` was *reassigned* each level (`nodes = nodes[p]`; `nodes = newnodes`) → Julia boxed it
(`Core.Box`) → every one of the billions of sort comparisons was type-unstable and heap-allocating.
16 GB of garbage, ~90% of build time.

**Fix.** Captured variables (`nodes`, `pad`) are assigned once and only mutated in place (scratch +
`copyto!`, never rebound); non-captured `pts`/`indices` are double-buffered by swapping instead of
`pts[p,:]` slicing; the sort uses `QuickSort` (total-order comparator, no temp buffer). The M
preceding-neighbour queries (independent, read-only traversal, disjoint output columns) are threaded.

**Result of the type-stability fix.** `build_graph` 30.8 s → 1.94 s @500k; 74 s → 12.1 s (1 thread)
@1M; allocations 16 GB → 0.22 GB.

**Then parallelized for large datasets.** The per-level passes moved into top-level threaded helpers
(`_bt_splitdims!`/`_bt_pad!`/`_bt_gather!`/`_bt_update!` via `Threads.@threads`) and the per-level
sort became a parallel merge-sort (`_bt_psort!`: parallel chunk sorts + parallel pairwise-merge
rounds, serial-QuickSort fallback below 8192/1 thread). These are *functions*, not closures, so the
swapped `pts`/`indices` buffers aren't captured-and-boxed. The total-order comparator makes the
merge deterministic = the serial order, so byte-identity holds (also verified vs brute-force
preceding-kNN, 0 mismatches). `query_preceding_neighbors_special` is likewise threaded (independent
queries, disjoint output columns); `seg_minmax` stays serial (keyed reduction).

**Final result (16 threads):** `build_tree_special` @1M 5.06 s → 1.88 s (2.7×, Amdahl-limited by
the serial `seg_minmax`/copies); query 7.11 s → 0.97 s (7.3×); **full `build_graph` @1M 74 s →
3.3 s = 22×** over the original, byte-identical. At ≥5M the build becomes **gather-memory-bound**
(the `pts[p[i]]` permutation gather is random access over a >100 MB working set), which threading
helps but cannot fully hide — the remaining lever there would be a cache-blocked/radix permute.

The smaller leftovers the audit flagged (`quantize` broadcasts, `order_by_depth` slicing,
`_compute_offsets` `push!`, scheme-B `Set`/`Dict`) are each <1% now and not worth chasing. The GPU
forward/gradient depth-batch sweeps are already on-device with a single sync; their
sequential-across-batches structure is intrinsic to Vecchia and not parallelizable.

## Method notes / caveats
- No `ncu`/`nsys` here, so achieved occupancy, DRAM throughput %, and stall reasons are inferred, not
  directly measured. Installing Nsight Compute would let us confirm the local-memory (LSU) and
  long-scoreboard stall picture and quantify (1)'s headroom precisely.
- Numbers are A6000-specific; the *shape* (O(k³), local-memory-bound, occupancy-rich) is
  architecture-independent for one-thread-per-point small-dense-Cholesky.
- The custom-CUDA bridge (`csrc/`) is the natural place to prototype lever (1): a new
  warp-cooperative `.cu` kernel slots in behind the same `refine_*_custom` API for A/B testing
  against the KA path (see `csrc/README.md`).
