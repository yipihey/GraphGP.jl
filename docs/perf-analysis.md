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

## Method notes / caveats
- No `ncu`/`nsys` here, so achieved occupancy, DRAM throughput %, and stall reasons are inferred, not
  directly measured. Installing Nsight Compute would let us confirm the local-memory (LSU) and
  long-scoreboard stall picture and quantify (1)'s headroom precisely.
- Numbers are A6000-specific; the *shape* (O(k³), local-memory-bound, occupancy-rich) is
  architecture-independent for one-thread-per-point small-dense-Cholesky.
- The custom-CUDA bridge (`csrc/`) is the natural place to prototype lever (1): a new
  warp-cooperative `.cu` kernel slots in behind the same `refine_*_custom` API for A/B testing
  against the KA path (see `csrc/README.md`).
