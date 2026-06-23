# GraphGP.jl (CUDA) vs JAX vs custom graphgp-cuda extension — GPU benchmark

**Hardware:** NVIDIA RTX A6000 (sm_86, 45 GiB), CUDA driver 13.3
**Software:**
- Julia 1.12.6 + CUDA.jl (CUDA_Runtime 13.5, KernelAbstractions 0.9.41)
- JAX 0.10.2 `jax[cuda13]`, f32, jit
- `graphgp-cuda` 0.0.4 — the **reference** custom CUDA kernels (built with nvcc 13.1),
  invoked via `cuda=True`

**Workload:** synthetic graph, `K=10` neighbors, `D=3`, `n0=1000`, `M = N − n0` refined
points. Float32 throughout. Median of repeated runs, device-synchronized.

## Performance tuning pass (A6000)

A benchmark-driven optimization pass. Before → after at N=1M (K=10, D=3, f32):

| Op | Before | After | Change |
| --- | --- | --- | --- |
| `refine_logdet` | 160 M pts/s | **180 M pts/s** | +12% |
| `refine_inv` | 142 M pts/s | **154 M pts/s** | +8% |
| `query_preceding_neighbors_ka` | 4.7 M pts/s (200K) | **25.9 M pts/s (200K)** | **5.5×** |
| `refine_logdet_grad_points` (GPU) | 31 M pts/s | **27 M pts/s** ¹ | wg=256 |
| `build_graph_ka` end-to-end | 1.38 s | **1.00 s** | −28% |

Levers (each measured):
- **Workgroup size 64 → 32** (one warp/block). The private-memory-heavy kernels (the
  `(k+1)²` matrix; the query's DFS stack) are occupancy-limited; the smallest block maximizes
  warps in flight. Lifted `refine_logdet`/`refine_inv` ~8–12% and the query *kernel* ~50%.
- **Bottom-up AABB in the query pack.** The old node-pack scanned each node's segment (one
  workitem/node → the root scanned all N in one thread, ~85% of query time). Replaced by a
  parallel leaf-init + deepest→root child-box merge — O(N). This is what makes the query 5.5×.
- **`_wgsize_scatter=256`** for the atomic-scatter point-gradient kernels (their scatter target
  is large, so atomic latency hides better with more threads).

¹ point-grad throughput is similar at wg=256 vs the previous 64; the win is over the wg=32
default the other kernels use (33.7→38 M/s at K=10), and it remains ~167× over the CPU host path.

Gradient w.r.t. `cov_vals` (~40 M pts/s) was diagnosed as ~50% analytic-Cholesky-pullback
compute and ~50% off-diagonal atomic scatter into the small `d_vals` (the diagonal is already
coalesced to one atomic/point). A shared-memory block histogram could recover much of the
scatter half (best case ~1.5×) — a documented, bounded future option.

Three implementations compared:
1. **GraphGP.jl** — KernelAbstractions/CUDA.jl, one workitem per point, matrix in private memory.
2. **JAX (pure)** — `cuda=False`; materializes the full `(M, K+1, K+1)` conditioning tensor.
3. **graphgp-cuda** — `cuda=True`; the hand-written reference CUDA extension to beat.

Reproduce:
```bash
julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_gpu.jl  <N> 10 3   # GraphGP.jl
python julia/GraphGP/test/bench_jax.py                         <N> 10 3   # pure JAX
python julia/GraphGP/test/bench_jax_cuda.py                    <N> 10 3   # reference CUDA
```

## `refine_logdet` — throughput (M pts·s⁻¹) / wall (ms)

| N | GraphGP.jl | graphgp-cuda (ref) | JAX (pure) |
| --- | --- | --- | --- |
| 200 K | **163** / 1.22 ms | 32 / 6.29 ms | 8.6 / 23.2 ms |
| 1 M   | **160** / 6.23 ms | 94 / 10.7 ms | 8.6 / 116 ms |
| 5 M   | 155 / 32.3 ms | 150 / 33.3 ms | 8.6 / 584 ms |
| 20 M  | 153 / 130 ms  | **170** / 118 ms | OOM (47 GiB) |

## `refine_inv` — throughput (M pts·s⁻¹) / wall (ms)

| N | GraphGP.jl | graphgp-cuda (ref) | JAX (pure) |
| --- | --- | --- | --- |
| 200 K | **158** / 1.26 ms | 30 / 6.63 ms | 7.8 / 25.7 ms |
| 1 M   | **142** / 7.04 ms | 85 / 11.8 ms | 7.8 / 128 ms |
| 5 M   | **132** / 37.9 ms | 122 / 40.9 ms | 7.7 / 648 ms |
| 20 M  | 127 / 158 ms  | **128** / 156 ms | OOM |

## `refine_logdet_grad_vals` (gradient w.r.t. cov)

| N | GraphGP.jl | graphgp-cuda (ref) | JAX (pure) |
| --- | --- | --- | --- |
| 200 K | 32 M/s — 6.17 ms  | **not implemented** ¹ | 12,018 ms |
| 1 M   | 37 M/s — 27.1 ms  | not implemented | 44,598 ms |
| 5 M   | 42 M/s — 119 ms   | not implemented | >>100 s |
| 20 M  | 43 M/s — 462 ms   | not implemented | OOM |

¹ `cuda=True` `refine_logdet` raises `NotImplementedError: Differentiation rule for
  'graphgp_cuda_refine_logdet_32' not implemented` — the reference extension (v0.0.4) has
  no autodiff rule for this op, so the gradient is only available from pure JAX (which is
  catastrophically slow) or GraphGP.jl.

## Real `build_graph`-derived graph (identical graph for all three)

The tables above use a synthetic random-neighbor graph. The numbers below instead build **one
real `gp.build_graph` graph** (spatially-coherent neighbors, CUDA tree builder), dump it, and
run all three implementations on that *same* graph. Points are quantized onto the 21-bit/axis
integer lattice and dequantized so JAX, the CUDA extension, and GraphGP.jl evaluate identical
points. Cross-check at N=200 K: reference CUDA logdet = −583769.7, GraphGP.jl = −583767.4
(relative diff 3.9e-6, f32 roundoff).

Reproduce:
```bash
python julia/GraphGP/test/bench_realgraph.py <N> 10 3 /tmp/rg.npz   # builds+dumps+benches JAX & CUDA
julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_realgraph.jl /tmp/rg.npz   # GraphGP.jl on same graph
```

`refine_logdet` — throughput (M pts·s⁻¹):

| N | GraphGP.jl | graphgp-cuda (ref) | JAX (pure) |
| --- | --- | --- | --- |
| 200 K | **172** | 32 | 8.6 |
| 1 M   | **166** | 94 | 8.6 |
| 5 M   | **162** | 150 | 8.6 |
| 20 M  | 158 | **169** | OOM |

`refine_inv` — throughput (M pts·s⁻¹):

| N | GraphGP.jl | graphgp-cuda (ref) | JAX (pure) |
| --- | --- | --- | --- |
| 200 K | **169** | 30 | 7.8 |
| 1 M   | **148** | 90 | 7.8 |
| 5 M   | **141** | 122 | 7.7 |
| 20 M  | **136** | 129 | OOM |

`refine_logdet_grad_vals` (GraphGP.jl; ref has no autodiff rule, pure JAX is 1000× slower):
200 K 6.75 ms (30 M/s) · 1 M 29.8 ms (34 M/s) · 5 M 138 ms (36 M/s) · 20 M 541 ms (37 M/s).

On the real graph GraphGP.jl is **on par with or faster than** the reference everywhere except
logdet at 20 M (ref ~7% ahead); it now *beats* the reference on both ops at 5 M and on inv at
20 M. Spatial coherence slightly helps GraphGP.jl (memory-access pattern) — the synthetic
graph was, if anything, mildly pessimistic for it. Conclusions are unchanged from the
synthetic case.

## Graph construction (GPU build_tree + GPU k-NN query)

The graph-build pipeline is implemented in Julia and now runs on the GPU:
- **`build_tree_ka`** — sort-based k-d tree build (one global sort per level via the GPU
  `sortperm`); a valid k-d tree (validated against brute-force preceding k-NN).
- **`query_preceding_neighbors_ka`** — k-NN query, pruned by the tight per-node AABB *and* an
  index-range skip (`seg_lo ≥ m` ⇒ subtree has no preceding points), validated to match the
  scalar reference's neighbor sets.

RTX A6000, `K=10`, `D=3`, `n0=1000`:

| N | build_tree (CPU, BFS) | build_tree_ka (GPU) | GPU query (bottom-up pack + wg=32) |
| --- | --- | --- | --- |
| 100 K | — | — | 4.1 ms (24.4 M pts/s) |
| 200 K | 4.4 s | **39 ms** (~110×) | 7.7 ms (25.9 M pts/s) |
| 1 M   | ~25 s | **0.19 s** | — |
| 5 M   | — | 1.3 s | — |
| 20 M  | — | 7.2 s | — |

`build_tree_ka` was sped up 3–5× by switching from a per-node widest-dimension split (one
workitem per node → a single thread scanning the whole segment at the top levels, ~83% of
build time) to a **round-robin** split dimension (cycled by level, as in cudaKDTree/bosque),
which needs no per-node min/max scan. The build is now sort-bound (≈78% is the per-level GPU
`sortperm`, the natural floor of a sort-based build; ≈10% kernels). Reproduce:
`julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_tree.jl 20000000`.

Reproduce: `julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_build.jl 200000 10 3`.

Query throughput evolution at 200 K: 0.4 M pts/s (initial, loose AABB over all points) → 3.1
(index-range skip: skip subtrees with no preceding points, the decisive Vecchia prune) → 4.7
(Float32 geometry in a packed per-node record) → **25.9** (bottom-up AABB pack, removing the
per-node segment scan, + workgroup size 32). The Float32 packed record reads each node as one
contiguous Int32 `[lo, hi, split_dim, split_val, min(D), max(D)]` (float fields
bit-reinterpreted; positions exact Int32; A6000 f64 throughput is ~1/32 of f32). The bottom-up
pack is the big one: the prior pack scanned each node's segment (root scanned all N in one
thread, ~85% of query time); replacing it with a parallel leaf-init + deepest→root box merge
(O(N)) gave the final 4.7→25.9 jump. The GPU build is ~29× faster than the CPU BFS build at 200 K.

**Fully fused on-device build** (`build_graph_ka`: tree → query → depths → reorder → quantize,
all on the GPU, returning a device-resident `GraphGPProblem`):

| N | build_graph_ka (GPU, end-to-end) |
| --- | --- |
| 200 K | **0.18 s** |
| 1 M | **1.0 s** |

(The CPU `build_graph` is tens of seconds at 200 K — `build_tree` alone is 4.4 s plus the
scalar O(N²) query.) Validated by `check_graph` + generate/inverse roundtrip. So
`build_graph_ka(CuArray(points), …) → generate/refine/gradients` runs end-to-end on the GPU
with no Python and no host round-trip.

## Takeaways

- **vs the reference CUDA extension (the bar to beat):** GraphGP.jl **matches it at scale**
  (within ~10% at 5–20 M points; the reference edges ~11% ahead on logdet at 20 M, GraphGP.jl
  is marginally ahead on inv) and is **2–5× faster at small/medium N** (200 K–1 M), where the
  reference is launch/overhead-bound. A single backend-agnostic Julia kernel reaches
  parity with hand-tuned CUDA.
- **Gradient is GraphGP.jl's clear advantage:** the reference extension has no differentiation
  rule for `refine_logdet`, and pure-JAX autodiff is 3 orders of magnitude slower. GraphGP.jl's
  analytic Cholesky pullback runs in 6–460 ms across the whole range.
- **vs pure JAX:** 17–20× faster on forward ops, and the only two implementations (GraphGP.jl
  and the reference) that run at all at 20 M — pure JAX OOMs trying to allocate 47 GiB.
