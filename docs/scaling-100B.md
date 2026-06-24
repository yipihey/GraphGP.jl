# Scaling GraphGP.jl to 100 billion points

Design notes for running the Vecchia / nearest-neighbor-graph GP on N ≈ 10¹¹ points across a
multi-node, multi-GPU cluster. It states the memory/communication budget, what the **merged**
distributed layer (`src/distributed.jl`, `ext/GraphGPMPIExt.jl`) already does, and concrete API
sketches for the pieces still needed. Status tags: ✅ merged · 🟡 foundation present · ⬜ to build.

## 1. The budget

At N = 10¹¹ the state is dominated by two arrays:

| array | layout | bytes/pt | total @ 100B |
| --- | --- | --- | --- |
| `neighbors` (K×N) | Int64, K=10 | 80 | **8.0 TB** |
| `coords` (D×N) | UInt32, D=3 | 12 | 1.2 TB |
| `xi` / `values` / field | Float32 | 4 each | 0.4 TB each |

≈ 10 TB persistent. On 80 GB GPUs that is **~200–256 GPUs** to hold it with working headroom; at
256 GPUs each rank owns ~400 M points (~32 GB neighbors + a few GB coords/halo/field) — fits 80 GB.
The fused per-point kernels run at ~150–400 M pts/s/GPU, so a likelihood+gradient evaluation across
256 GPUs is **sub-second of compute**. The bottlenecks are therefore **data movement, graph
construction, and storage** — not arithmetic.

Two structural facts drive everything below:

1. **Fitting is a sum of independent per-point terms.** `refine_logdet = Σ log(stdₘ)`, and its
   gradient scatters into a tiny `nbins ≈ 1000` histogram. Distributing it is "partition the points,
   run the existing kernels, one `Allreduce`." This is the scientifically important target and it is
   the easy part.
2. **The forward sweep is sequential over depth batches.** `refine!` processes batches in order;
   batch *b* reads `values[neighbors]` written by earlier batches. Distributed, that is one halo
   exchange + barrier **per depth batch** — so the batch count must stay small (see §6).

## 2. What already works — scheme A (replicate coords) ✅

`distribute(prob, comm; scheme = :replicate_coords)` builds a `DistributedGraphGPProblem` in which
every rank holds the **full `coords`** and a balanced contiguous **column slice of `neighbors`**
(the dominant O(N) array), with `n0` shifted so local column *j* ↔ global self index
`n0 + m_lo - 1 + j`. The local kernels then run verbatim on `local_prob`; only the reduction
communicates:

- `generate_logdet(dprob)` — Float64 local partial + root-only dense first layer + sum-`Allreduce`. ✅
- `generate_logdet_and_grad_vals(dprob)` — local `nbins` histogram + dense block + sum-`Allreduce`. ✅
- `refine_inv_loss_grad_vals`, distributed inverse (`_dist_allgather_columns`). ✅
- `distributed_build_graph(points, comm, n0, k, …)` — scheme-A build (Phase 4). ✅
- `distributed_quantize(points_local, comm; bits)` — global lattice via `Allreduce` min/max. 🟡
- f64 partial sums before the `Allreduce` (reproducible across rank counts); CUDA-aware MPI with a
  host-staging fallback. ✅

**Ceiling:** `coords` is *replicated*, so scheme A stops where 1.2 TB of coords no longer fits one
GPU — roughly **1–3 B points**. 100B requires partitioning `coords` too.

## 3. Scheme B — partition coords + halo ✅ (fitting)

Past the replication ceiling, points are **spatially domain-decomposed**: each rank owns a region
plus a **ghost halo** of the neighbor points referenced from outside its region. Cosmological
clustering keeps halos small (neighbors are overwhelmingly local). The lattice foundation
(`distributed_quantize`) gives a globally consistent integer grid.

**Implemented for the fitting pipeline** (`src/distributed.jl`, `ext/GraphGPMPIExt.jl`):

```julia
struct PartitionedGraphGPProblem{P<:GraphGPProblem, DP, C}
    local_prob :: P            # compacted GraphGPProblem: [context/halo ; owned-self] coords,
                               #   neighbors remapped to LOCAL indices, local n0 = context size
    dense_prob :: DP           # global dense first layer (root), else nothing
    comm       :: C            # MPI.Comm
    n0         :: Int          # global dense first-layer size
    gids       :: Vector{Int}  # global point id of each local coord column (for data gather)
    owned_cols :: Vector{Int}  # global refined columns (1..M) this rank owns
    is_root    :: Bool
    indices    :: Union{Nothing, Vector{Int}}
end

distribute(prob, comm; scheme = :partition_coords) :: PartitionedGraphGPProblem
```

Each rank computes the same spatial (Morton) order of the refined points, takes its balanced
contiguous chunk as `owned_cols` (so the partition needs no communication), and `_scheme_b_local`
builds the compacted local problem holding **only owned + ghost coords** with neighbors remapped to
local indices. The existing per-point drivers then run **unchanged**, and the fitting reduction is
identical to scheme A. Implemented methods: `generate_logdet`, `generate_logdet_grad_vals`,
`generate_logdet_and_grad_vals`, `generate_inv_loss_grad_vals` — all order-independent reductions,
so the spatial partition is correctness-neutral.

**Validated** (`bench/distributed/test_scheme_b.jl`, `mpiexec -n 1/2/4`, CPU): every quantity matches
the serial oracle to f64 round-off (logdet relerr 0; grads ~1e-14), invariant to rank count, while
the coords held per rank fall **100 % → 57 % → 33 %** at 1 → 2 → 4 ranks (ideal 25 % at 4; the gap is
the Morton-bounded halo). The pure remap (`_scheme_b_local`) is also unit-tested without MPI in the
main suite (`test/test_scheme_b.jl`).

**Remaining for scheme B at 100B:** the input `prob` here is still replicated (as in scheme A) — the
*from-scratch* no-replication path (each rank reads only its slab from disk + a neighbor-to-neighbor
`Alltoallv` coords/values halo) rides on §4 (distributed build) + §5 (I/O); the remap and reduction
above are unchanged.

## 4. Distributed graph construction ⬜ — the real blocker

`build_graph` (root-relative, byte-identical to the JAX reference) is single-node CPU
(~74 s / 1 M) — hopeless at 100B. The distributed build is staged; each stage maps onto an existing
local routine:

```julia
function distributed_build_graph_B(points_local, comm; n0, k, bits=21, halo_width)
    coords_local, origin, scale = distributed_quantize(points_local, comm; bits)   # 🟡 exists
    # (a) spatial decomposition: redistribute points to spatially-contiguous rank slabs
    slab = spatial_partition!(coords_local, comm)                                  # ⬜ Morton/SFC split
    # (b) per-rank tree over [own slab ∪ ghost halo], then preceding k-NN with ghost zones
    ghosts = exchange_halo(slab, comm; width=halo_width)                           # ⬜
    tree   = build_tree_special(local_points(slab, ghosts))                        # ✅ reuse (root-relative)
    nbr    = query_preceding_neighbors_special(tree…, n0, k)                       # ✅ reuse, ghost-aware
    # (c) global Vecchia depths: boundary relaxation to fixpoint (neighbors cross ranks)
    depths = distributed_compute_depths(nbr, comm)                                 # ⬜ boundary exchange loop
    # (d) global order-by-depth: distributed sort + neighbor renumber  ← HARDEST
    return distributed_order_by_depth(tree, nbr, depths, comm)                     # ⬜
end
```

- **(b)** can reuse `build_tree_special` / `query_preceding_neighbors_special` per rank over
  `[slab ∪ ghost]`. Spike `~/codes/bosque` (Rust k-d tree, Julia bindings) for the ghost-zone k-NN.
- **(c)** `compute_depths` is already an iterative relaxation; the distributed form adds a boundary
  exchange of changed depths each round until a global fixpoint (`Allreduce` of a "changed" flag).
- **(d)** `distributed_order_by_depth` — a **distributed sort by depth + global neighbor renumber** —
  is the genuinely hard piece (the local `order_by_depth` is a sort + reindex). After it, `owned`
  becomes a contiguous global id range per rank (§3).

The build is a **once-per-dataset batch job**; serialize the result (§5) and never rebuild casually.
Validate the whole pipeline at a fat-node-buildable N (a few B) before scaling.

## 5. Compact indices + out-of-core I/O ⬜

- **Indexing.** Global ids exceed 2³¹ past 2.1 B points, forcing Int64 `neighbors` (the 8 TB row).
  Scheme B's **rank-local 32-bit index + owner-rank tag (~5 B/pt)** drops `neighbors` from ~8 TB to
  **~5 TB** and falls out of the partition for free.
- **Persistence.** Build once → write `neighbors` column-slices with MPI-IO; each rank `mmap`s or
  scatter-loads its slice on restart; broadcast the small coords-slab metadata + lattice. Checkpoint
  the graph as a first-class multi-TB artifact.

```julia
save_graph(path, dprob, comm)               # parallel write: per-rank neighbor slabs + manifest
load_graph(path, comm) :: PartitionedGraphGPProblem   # mmap each rank's slab; broadcast metadata
```

## 6. Forward field generation ⬜/✅

- **Ensembles of fields (mocks)** — embarrassingly parallel: the same distributed graph, different
  `xi` seeds per rank-group. The common cosmology need; nearly free once the graph is distributed. ✅
- **One single 100B field** — distributed forward sampling: process depth batches in global order,
  with a **halo exchange of newly-generated `values` + a barrier per batch** (`offsets` is already
  carried on the distributed problem). This is where the **shallow heap-order `build_graph`** is
  decisive: it yields ~tens-to-low-hundreds of depth batches (33 at 3 K, 86 at 1 M, ~99 at 10 M)
  instead of thousands, so the sweep needs ~10²–10² communication rounds, not 10⁴. Without it,
  distributed single-field sampling is latency-dead. ⬜ (only needed if a science case requires one
  field that doesn't fit a node).

## 7. Hardware sizing

| N | scheme | GPUs (80 GB) | pts/GPU | per-GPU neighbors |
| --- | --- | --- | --- | --- |
| 1 B | A (replicate coords) | ~16 | 64 M | ~5 GB |
| 10 B | B (partition coords) | ~32–64 | ~160–320 M | ~13–26 GB |
| 100 B | B + compact idx + I/O | ~200–256 | ~400 M | ~20–32 GB |

A Perlmutter / Frontier / Leonardo-class allocation; GH200's large unified memory reduces the GPU
count further. f64 reduction keeps cross-rank-count results reproducible to ~1e-15.

## 8. Phased plan

1. **Scheme B fitting** ✅ — `:partition_coords` constructor + compacted local problem +
   order-independent reductions. Unblocks per-rank coords past the replication ceiling; validated
   multi-rank against the serial oracle (§3).
2. **Compact indexing** ⬜ — store `local_prob.neighbors` as rank-local 32-bit + owner tag (the
   remap already produces local indices); ~35 % off the dominant array.
3. **From-scratch no-replication path** ⬜ — each rank reads only its slab; a neighbor-to-neighbor
   `Alltoallv` coords/values halo replaces the replicated input (the remap/reduction are unchanged).
4. **Distributed build** ⬜ — spatial decomp + ghost k-NN (spike bosque) + distributed depth
   relaxation + the distributed sort/renumber. Validate at a few-B N on a fat node, then scale.
5. **Parallel graph I/O + checkpointing** ⬜ (`save_graph`/`load_graph`).
6. **Distributed single-field sampling** ⬜ (per-batch halo) — only if a single field must exceed one
   node; otherwise ensembles are free.

**Net:** fitting at 100B is essentially solved by the merged scheme-A reduction plus scheme-B coords
partitioning; the real investment is **distributed graph construction and its TB-scale storage**,
with single-field sampling as an optional add-on that the shallow build has already de-risked.

## References in this repo
- `src/distributed.jl` — scheme-A + **scheme-B** distributed problems, reductions, entry points.
- `ext/GraphGPMPIExt.jl` — `distribute` (both schemes), MPI shims, GPU binding, balanced ranges.
- `bench/distributed/test_scheme_b.jl` — multi-rank scheme-B fitting vs serial + coords-compaction.
- `test/test_scheme_b.jl` — pure `_scheme_b_local` remap unit test (no MPI).
- `src/graph_build.jl`, `src/tree_special.jl` — the local build stages the distributed build reuses.
