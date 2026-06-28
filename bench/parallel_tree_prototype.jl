# ============================================================================================
# PARALLEL TREE BUILD — validated prototype (byte-exact). Banked 2026-06-28.
# ============================================================================================
# Goal: build the `build_tree_special` k-d tree faster on a large shared-memory (NUMA) box by
# decomposing the work, WITHOUT changing the resulting graph (so generate/logdet/clustering are
# bit-for-bit the global build — no approximation, no boundary seam).
#
# WHY decomposition and not a cleverer sort:
#   * Phase profile of build_graph: tree build is ~66–87% of the cost; inside it the per-level
#     GLOBAL parallel sort `_bt_psort!` is ~87% and is memory-bandwidth-bound.
#   * The heap is a SPECIAL order (`_sp_left(c)=c+2^level`; the median point BECOMES the split
#     node), so `nodes` is NOT value-sorted between levels — a "segment-local sort by contiguity"
#     is INVALID (verified: it breaks at level 1). Hence the coarse/fine decomposition below.
#
# APPROACH (this file):
#   1. coarse_phase(points, L): run build_tree_special's level loop for the top L levels (incl.
#      level L's sort/gather, not its update) → points grouped (sorted) by their level-L node,
#      i.e. partitioned into 2^L subtrees, each a contiguous slice.
#   2. For each subtree: build it STANDALONE (build_tree_serial) and GRAFT into the global heap via
#      paired_graft (paired BFS: local node 0 ↔ global node g, children via _sp_left/_sp_right in
#      lockstep). De-risked separately: a standalone-rebuilt subtree == the global subtree under the
#      graft (points + split_dims + bijection), at multiple roots and N.
#   3. Subtrees are independent → the outer loop is parallel; each subtree build must be SERIAL
#      (build_tree_serial) because nesting @threads inside build_tree_special's own
#      `@threads :static` ERRORS ("cannot be used concurrently or nested").
#
# STATUS:
#   * BYTE-IDENTICAL to build_tree_special (sorted_pts + split_dims + indices) at 10k…40M. ✔
#   * Single-process speedup is MODEST (~1.2–1.34× at 20–40M) and obscured by large run-to-run
#     variance: the build is memory-bandwidth-bound (~8 of 64 cores active) and a single process
#     can't escape the shared-memory bandwidth ceiling. `numactl --interleave=all` does NOT help.
#   * The REAL win (DEFERRED, NOT BUILT): a NUMA-process layer — coarse in one process, then
#     distribute the 2^L subtree point-sets to one PINNED builder process per NUMA node (node-LOCAL
#     memory via first-touch) → ~16× aggregate memory bandwidth (cf. echoes/parallel.py numa_map,
#     which gives 15× at 95% on bandwidth-bound ensemble work), then assemble → query → depths.
#   * Reached for a single ≳1e9-point field (LSST/Euclid-era). The ECHOES ensemble workload (K
#     moderate LGCP fields) needs NONE of this — numa_map over WHOLE fields is seam-free + 15×.
#
# DEAD END (do not retry): independent block-Vecchia sampling. A disjoint-brick field has perfect
# ξ(r) WITHIN a brick but ξ(r)≈0 ACROSS brick boundaries (hard seam; 15% of close pairs cross-brick)
# → suppresses small-scale clustering. A Vecchia field is a specific sqrt of K tied to the GLOBAL
# ordering, so independently-built bricks are different realizations. THIS file avoids that by
# producing the IDENTICAL global tree.
#
# Run:  julia -t <N> --project=bench bench/parallel_tree_prototype.jl   (prints byte-identity + speedup)
# ============================================================================================
using GraphGP, Printf, Random, Base.Threads
const G = GraphGP

# Run build_tree_special's level loop for levels 0..L, doing level L's sort+gather but NOT its
# update — so `nodes` holds the level-L assignment and points are grouped (sorted) by it.
function coarse_phase(points::Matrix{Float64}, L::Int)
    N, D = size(points)
    pts = copy(points); pts_s = similar(pts)
    nodes = zeros(Int, N); nodes_s = Vector{Int}(undef, N)
    indices = collect(1:N); indices_s = Vector{Int}(undef, N)
    split_dims = fill(-1, N)
    seg_max = Matrix{Float64}(undef, N, D); seg_min = Matrix{Float64}(undef, N, D)
    pad = Vector{Float64}(undef, N); p = Vector{Int}(undef, N); p_s = Vector{Int}(undef, N)
    nt = Threads.nthreads()
    lt = (a, b) -> begin
        @inbounds begin
            na = nodes[a]; nb = nodes[b]; na != nb && return na < nb
            pa = pad[a];  pb = pad[b];  pa != pb && return pa < pb
            return a < b
        end
    end
    for level in 0:L
        n_above = (1 << level) - 1; n_level = 1 << level
        G._bt_segminmax!(seg_max, seg_min, pts, nodes, N, D)
        G._bt_splitdims!(split_dims, seg_max, seg_min, n_above, N, D)
        G._bt_pad!(pad, pts, split_dims, nodes, N)
        @inbounds for i in 1:N; p[i] = i; end
        G._bt_psort!(p, p_s, lt, nt)
        G._bt_gather!(pts_s, pts, nodes_s, nodes, indices_s, indices, p, N, D)
        pts, pts_s = pts_s, pts; indices, indices_s = indices_s, indices
        copyto!(nodes, nodes_s)
        if level < L
            n_remaining = N - n_above; q = div(n_remaining, n_level); r = rem(n_remaining, n_level)
            G._bt_update!(nodes_s, nodes, n_above, n_level, q, r, N)
            copyto!(nodes, nodes_s)
        end
    end
    return pts, nodes, indices, split_dims
end

# SERIAL build_tree_special (no internal @threads) so it can run inside a parallel outer loop
# without nesting. Byte-identical to build_tree_special (the per-level sort is a total order, so a
# serial MergeSort gives the same permutation as the parallel merge sort).
function build_tree_serial(points::Matrix{Float64})
    N, D = size(points)
    n_levels = N > 0 ? (G._flevel(N) + 1) : 0
    pts = copy(points); pts_s = similar(pts)
    nodes = zeros(Int, N); nodes_s = Vector{Int}(undef, N)
    indices = collect(1:N); indices_s = Vector{Int}(undef, N)
    split_dims = fill(-1, N)
    seg_max = Matrix{Float64}(undef, N, D); seg_min = Matrix{Float64}(undef, N, D)
    pad = Vector{Float64}(undef, N); p = Vector{Int}(undef, N)
    lt = (a, b) -> begin
        @inbounds begin
            na = nodes[a]; nb = nodes[b]; na != nb && return na < nb
            pa = pad[a];  pb = pad[b];  pa != pb && return pa < pb
            return a < b
        end
    end
    for level in 0:(n_levels - 1)
        n_above = (1 << level) - 1; n_level = 1 << level
        G._bt_segminmax!(seg_max, seg_min, pts, nodes, N, D)          # already serial
        @inbounds for i in 1:N                                        # serial splitdims
            (i - 1) < n_above && continue
            bestd = 1; bestr = -Inf
            for d in 1:D
                r = seg_max[i, d] - seg_min[i, d]; r > bestr && (bestr = r; bestd = d)
            end
            split_dims[i] = bestd
        end
        @inbounds for i in 1:N; pad[i] = pts[i, split_dims[nodes[i] + 1]]; end   # serial pad
        @inbounds for i in 1:N; p[i] = i; end
        sort!(p; lt = lt, alg = MergeSort)                            # serial sort = same order
        @inbounds for i in 1:N                                        # serial gather
            pp = p[i]
            for d in 1:D; pts_s[i, d] = pts[pp, d]; end
            nodes_s[i] = nodes[pp]; indices_s[i] = indices[pp]
        end
        pts, pts_s = pts_s, pts; indices, indices_s = indices_s, indices; copyto!(nodes, nodes_s)
        n_remaining = N - n_above; q = div(n_remaining, n_level); r = rem(n_remaining, n_level)
        @inbounds for i in 1:N                                        # serial update
            idx = i - 1; s = nodes[i]; ii = s - n_above
            mid = (ii < r ? ii * (q + 1) + div(q + 1, 2) :
                            r * (q + 1) + (ii - r) * q + div(q, 2)) + n_above
            nodes_s[i] = (idx < n_above || idx == mid) ? s : (idx < mid ? s + n_level : s + 2 * n_level)
        end
        copyto!(nodes, nodes_s)
    end
    return pts, split_dims, indices
end

# local heap node l (0-based) -> global heap node, rooting local 0 at global g (paired BFS).
function paired_graft(g::Int, m::Int)
    gmap = Vector{Int}(undef, m); st = Tuple{Int,Int}[(0, g)]
    @inbounds while !isempty(st)
        (l, gg) = pop!(st); l >= m && continue
        gmap[l + 1] = gg
        push!(st, (G._sp_left(l),  G._sp_left(gg)))
        push!(st, (G._sp_right(l), G._sp_right(gg)))
    end
    gmap
end

function build_tree_parallel(points::Matrix{Float64}, L::Int)
    N, D = size(points)
    pts, nodes, indices, split_dims = coarse_phase(points, L)
    n_above = (1 << L) - 1
    sorted_pts = Matrix{Float64}(undef, N, D); out_sd = fill(-1, N); out_idx = Vector{Int}(undef, N)
    @inbounds for c in 0:(n_above - 1)                          # coarse internal nodes (one point each)
        for d in 1:D; sorted_pts[c + 1, d] = pts[c + 1, d]; end
        out_sd[c + 1] = split_dims[c + 1]; out_idx[c + 1] = indices[c + 1]
    end
    seglo = Int[]; seghi = Int[]; segg = Int[]                  # level-L segments (nodes sorted asc)
    i = n_above + 1
    @inbounds while i <= N
        j = i
        while j < N && nodes[j + 1] == nodes[i]; j += 1; end
        push!(seglo, i); push!(seghi, j); push!(segg, nodes[i]); i = j + 1
    end
    nseg = length(seglo)
    Threads.@threads :static for si in 1:nseg                  # parallel over subtrees; each subtree
        lo = seglo[si]; hi = seghi[si]; g = segg[si]; m = hi - lo + 1   # build is SERIAL (no nesting).
        spl, sdl, idxl = build_tree_serial(pts[lo:hi, :])      # STANDALONE subtree (cache-local)
        gmap = paired_graft(g, m)
        @inbounds for l in 0:(m - 1)
            gg = gmap[l + 1]
            for d in 1:D; sorted_pts[gg + 1, d] = spl[l + 1, d]; end
            out_sd[gg + 1]  = sdl[l + 1]
            out_idx[gg + 1] = indices[lo - 1 + idxl[l + 1]]
        end
    end
    return sorted_pts, out_sd, out_idx
end

if abspath(PROGRAM_FILE) == @__FILE__
    let pts = randn(2000, 3); build_tree_parallel(pts, 3); end     # warm
    Random.seed!(0)
    @printf("threads=%d\n", Threads.nthreads()); flush(stdout)
    for (N, L) in ((10_000, 4), (200_000, 6), (20_000_000, 8), (40_000_000, 8))
        pts = randn(N, 3)
        tref = @elapsed ref = G.build_tree_special(pts)
        tpar = @elapsed got = build_tree_parallel(pts, L)
        ok = ref[1] == got[1] && ref[2] == got[2] && ref[3] == got[3]
        @printf("N=%9d L=%d  build_tree_special=%6.1fs  parallel=%6.1fs  speedup=%.2fx  byte-identical=%s\n",
                N, L, tref, tpar, tref / tpar, ok); flush(stdout)
        pts = nothing; ref = nothing; got = nothing; GC.gc()
    end
end
