# Backend-agnostic (KernelAbstractions) k-d tree neighbor query — Workstream C, phase C2.
#
# The scalar CPU query in tree.jl recomputes a per-node bounding box by scanning the node's
# segment at *every* visited node (O(N) at the root → O(N²) overall). That is a fine
# correctness reference but does not scale. Here we instead:
#   1. precompute two compact per-node record arrays — integer positions `inodes`
#      `[lo, hi, split_dim]` (Int32) and Float32 geometry `fnodes` `[split_val, min[D], max[D]]`
#      — read with two small contiguous fetches per node. The geometry is Float32: the A6000's
#      f64 throughput is ~1/32 of f32, and the distance/AABB math dominates, so f32 ≈ 1.7×
#      faster (positions stay exact Int32). For a leaf, min == max == the point coordinates, so
#      the query needs no separate `spts` read for leaves, and
#   2. run one workitem per query point with a private fixed-size k-best buffer and an explicit
#      DFS stack, pruning by (a) the per-node AABB distance and (b) an index-range skip.
# Two prunes, both exact:
#   * AABB distance — the box over a node's contiguous tree-order segment [seg_lo, seg_hi] is
#     the tight bounding box of that subtree; its min-distance to the query is the exact lower
#     bound for the subtree, so a subtree whose box is farther than the current k-th neighbor
#     cannot improve and is skipped.
#   * index-range skip — `seg_lo[node] >= m` means every point in the subtree post-dates the
#     query point m, so the subtree holds no admissible *preceding* neighbor and is skipped
#     wholesale. This is the decisive prune for the Vecchia constraint (the AABB box otherwise
#     includes future points and prunes weakly for early query points).
# The exact k-NN set is identical to the scalar reference.
#
# Pure KernelAbstractions: the same kernels run on CPU and GPU; the backend is taken from the
# array type, so a CuArray tree runs the query on the GPU automatically.

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize, zeros as ka_zeros

const _QSTACK = 64   # max DFS stack depth (supports trees over up to ~2^63 points)

# Pack each node into one contiguous Int32 record column `rec` (RW = 4 + 2D rows):
#   [1]=lo  [2]=hi  [3]=split_dim  [4]=split_val  [5:4+D]=min  [5+D:4+2D]=max
# Positions (lo, hi, split_dim) are stored as Int32 (exact up to ~2.1e9 points); the geometry
# fields (split_val, min, max) are Float32 stored via bit-reinterpret into the same Int32 array.
# This gives one coalesced fetch per node *and* exact positions *and* fast f32 geometry math.
# `rec[1]==0` marks an empty implicit node.
#
# The subtree bounding box (min/max) is built BOTTOM-UP rather than by scanning each node's
# segment: a per-node scan is one workitem per node, so the root scans all N points in a single
# thread (the same bottleneck that dominated build_tree). Instead `_node_pack_init!` sets each
# leaf box to its point and each internal box to a placeholder (O(1) per node, fully parallel),
# then `_node_merge!` sweeps levels deepest→root merging child boxes — O(N) total, parallel.

# Per-node init: positions + split + leaf boxes (placeholder box for internal nodes).
@kernel function _node_pack_init!(rec, @Const(seg_lo), @Const(seg_hi), @Const(split_dim),
        @Const(spts), ::Val{D}) where {D}
    node = @index(Global)
    @inbounds begin
        lo = seg_lo[node]
        if lo == 0
            rec[1, node] = Int32(0)
        else
            hi = seg_hi[node]
            rec[1, node] = Int32(lo)
            rec[2, node] = Int32(hi)
            if lo < hi
                sd = split_dim[node]
                rec[3, node] = Int32(sd)
                mid = (lo + hi) >>> 1
                rec[4, node] = reinterpret(Int32, Float32(spts[sd, mid]))
                for d in 1:D
                    rec[4 + d, node] = reinterpret(Int32, Inf32)      # filled by merge
                    rec[4 + D + d, node] = reinterpret(Int32, -Inf32)
                end
            else
                rec[3, node] = Int32(1)
                rec[4, node] = Int32(0)
                for d in 1:D
                    v = Float32(spts[d, lo])                          # leaf box = the point
                    rec[4 + d, node] = reinterpret(Int32, v)
                    rec[4 + D + d, node] = reinterpret(Int32, v)
                end
            end
        end
    end
end

# Merge child boxes into each internal node at level `lvl_lo .. 2·lvl_lo-1` (one level).
@kernel function _node_merge!(rec, lvl_lo, max_node, ::Val{D}) where {D}
    g = @index(Global)
    node = lvl_lo + g - 1
    @inbounds begin
        if node <= max_node && rec[1, node] != 0 && rec[1, node] < rec[2, node]
            lc = 2 * node
            rc = lc + 1
            if rc <= max_node
                for d in 1:D
                    mn = min(reinterpret(Float32, rec[4 + d, lc]),
                        reinterpret(Float32, rec[4 + d, rc]))
                    mx = max(reinterpret(Float32, rec[4 + D + d, lc]),
                        reinterpret(Float32, rec[4 + D + d, rc]))
                    rec[4 + d, node] = reinterpret(Int32, mn)
                    rec[4 + D + d, node] = reinterpret(Int32, mx)
                end
            end
        end
    end
end

# One workitem per refined point t = 1..M (query tree position m = n0 + t). Finds the k
# nearest preceding (position < m) points and writes their 1-based tree indices, ascending
# by distance, into neighbors[:, t].
@kernel function _query_kernel!(neighbors, @Const(rec), @Const(spts),
        n0, m_base, max_node, ::Val{K}, ::Val{D}) where {K, D}
    t = @index(Global)
    m = n0 + m_base + t            # global query point; `m_base` offsets a distributed slice
    bd = @private Float32 (K,)     # k-best squared distances
    bi = @private Int (K,)         # k-best point indices (1-based)
    qp = @private Float32 (D,)     # query coordinates, cached once
    stack = @private Int (_QSTACK,)
    @inbounds begin
        for j in 1:K
            bd[j] = Inf32
            bi[j] = 0
        end
        for d in 1:D
            qp[d] = Float32(spts[d, m])
        end
        cnt = 0
        worst = Inf32
        sp = 1
        stack[1] = 1
        while sp >= 1
            node = stack[sp]
            sp -= 1
            (node < 1 || node > max_node) && continue
            lo = Int(rec[1, node])
            lo == 0 && continue
            # Index-range skip: a node's subtree occupies the contiguous tree-order range
            # [lo, hi], so lo >= m means the whole subtree post-dates the query point m → no
            # admissible preceding neighbor. The decisive prune for the Vecchia constraint.
            lo >= m && continue
            hi = Int(rec[2, node])

            # AABB pruning (only once the buffer is full). Geometry fields are Float32 bitcast.
            if cnt == K
                msq = 0.0f0
                for d in 1:D
                    q = qp[d]
                    nmn = reinterpret(Float32, rec[4 + d, node])
                    nmx = reinterpret(Float32, rec[4 + D + d, node])
                    c = q < nmn ? nmn : (q > nmx ? nmx : q)
                    dl = c - q
                    msq += dl * dl
                end
                msq >= worst && continue
            end

            if lo == hi
                # Leaf: min == max == the point's coordinates (from the node record).
                sq = 0.0f0
                for d in 1:D
                    dv = reinterpret(Float32, rec[4 + d, node]) - qp[d]
                    sq += dv * dv
                end
                if cnt < K
                    cnt += 1
                    bd[cnt] = sq
                    bi[cnt] = lo
                    if cnt == K
                        w = -Inf32
                        for j in 1:K
                            bd[j] > w && (w = bd[j])
                        end
                        worst = w
                    end
                elseif sq < worst
                    wp = 1
                    w = bd[1]
                    for j in 2:K
                        if bd[j] > w
                            w = bd[j]
                            wp = j
                        end
                    end
                    bd[wp] = sq
                    bi[wp] = lo
                    w2 = -Inf32
                    for j in 1:K
                        bd[j] > w2 && (w2 = bd[j])
                    end
                    worst = w2
                end
                continue
            end

            # Internal node: push children, nearer one last so it is popped first.
            left = 2 * node
            right = 2 * node + 1
            sd = Int(rec[3, node])
            splitv = reinterpret(Float32, rec[4, node])
            qsd = qp[sd]
            if qsd <= splitv
                if right <= max_node && sp < _QSTACK
                    sp += 1
                    stack[sp] = right
                end
                if left <= max_node && sp < _QSTACK
                    sp += 1
                    stack[sp] = left
                end
            else
                if left <= max_node && sp < _QSTACK
                    sp += 1
                    stack[sp] = left
                end
                if right <= max_node && sp < _QSTACK
                    sp += 1
                    stack[sp] = right
                end
            end
        end

        # Insertion-sort the cnt found neighbors ascending by distance, then emit.
        for a in 2:cnt
            dv = bd[a]
            iv = bi[a]
            b = a - 1
            while b >= 1 && bd[b] > dv
                bd[b + 1] = bd[b]
                bi[b + 1] = bi[b]
                b -= 1
            end
            bd[b + 1] = dv
            bi[b + 1] = iv
        end
        for j in 1:K
            neighbors[j, t] = j <= cnt ? bi[j] : 0
        end
    end
end

"""
    query_preceding_neighbors_ka(spts, seg_lo, seg_hi, split_dim, n0, k; backend) -> (K, M) Int

Backend-agnostic k-NN query over a prebuilt k-d tree. `spts` is `(D, N)` `Float64` points in
tree order; `seg_lo`/`seg_hi`/`split_dim` are the implicit-tree node arrays from `build_tree`
(all on the same backend as `spts`). Returns 1-based tree-position neighbor indices in a
`(k, M)` matrix (`M = N - n0`).

Distances are computed in Float32 (the geometry is packed Float32; positions stay exact
Int32). This matches the scalar `query_preceding_neighbors` neighbor set for well-separated
points; on near-equidistant ties the choice among them may differ (immaterial for the Vecchia
approximation). Runs on CPU or GPU depending on the array backend (pass `CuArray`s for GPU).
"""
function query_preceding_neighbors_ka(spts::AbstractMatrix{Float64},
        seg_lo::AbstractVector{<:Integer}, seg_hi::AbstractVector{<:Integer},
        split_dim::AbstractVector{<:Integer}, n0::Int, k::Int;
        backend = get_backend(spts), mrange::Union{Nothing, UnitRange{Int}} = nothing)
    D, N = size(spts)
    # `mrange` selects a contiguous slice of refined points (1-based, within 1:N-n0) to query —
    # used to distribute the query across ranks. Defaults to all refined points.
    rng = mrange === nothing ? (1:(N - n0)) : mrange
    m_base = first(rng) - 1
    M = length(rng)
    max_node = length(seg_lo)
    # One contiguous Int32 record per node (RW = 4 + 2D): positions exact, geometry Float32 via
    # bit-reinterpret → a single coalesced fetch per node with fast f32 math.
    RW = 4 + 2D
    rec = ka_zeros(backend, Int32, RW, max_node)
    spts32 = similar(spts, Float32)
    copyto!(spts32, spts)
    wgs = _wgsize(backend)
    # Init (leaf boxes + positions/split), then merge child boxes bottom-up (deepest → root).
    _node_pack_init!(backend)(rec, seg_lo, seg_hi, split_dim, spts, Val(D);
        ndrange = max_node, workgroupsize = wgs)
    synchronize(backend)
    Lmax = floor(Int, log2(max_node))
    for L in (Lmax - 1):-1:0
        lvl_lo = 1 << L
        _node_merge!(backend)(rec, lvl_lo, max_node, Val(D);
            ndrange = lvl_lo, workgroupsize = wgs)
        synchronize(backend)
    end
    neighbors = ka_zeros(backend, Int, k, M)
    _query_kernel!(backend)(neighbors, rec, spts32, n0, m_base, max_node, Val(k), Val(D);
        ndrange = M, workgroupsize = _wgsize(backend))
    synchronize(backend)
    return neighbors
end

# ── GPU/KA k-d tree build (sort-based, level by level) ────────────────────────────────────
#
# The CPU `build_tree` (tree.jl) does a sequential BFS median split. The GPU build instead
# mirrors the data-parallel JAX `_build_tree`: at each level, every node splits at the median
# of a round-robin dimension (`(level mod D)+1`, cycled by depth — as in cudaKDTree/bosque),
# realized as a single global sort of all points by a composite key (node-id in the integer
# part, normalized split coordinate in the fraction). One sort per level → O(N log²N) total.
#
# Round-robin avoids a per-node widest-dimension min/max scan, which was the build bottleneck:
# that scan used one workitem per node, so at the top levels a single thread scanned the whole
# segment (level 0 = one thread over all N points) — ~83% of build time. The split dimension
# is now uniform per level, needs no computation, and is stored per node in `set_children`.
#
# Points keep descending so all live positions stay at the current level (singletons descend
# the left spine). `sortperm`/`minimum` dispatch to the GPU backend at run time, so this needs
# no compile-time CUDA dependency; it also runs on the KA CPU backend. The resulting tree is a
# valid k-d tree (not byte-identical to the CPU build) and yields an equivalent Vecchia graph.

# Composite sort key per position: node id (integer part) + this level's normalized split
# coordinate (fraction). `sd` is the round-robin split dimension for the current level.
@kernel function _build_key_kernel!(key, @Const(node_of_pos), @Const(pts), sd, gmin, ginv)
    i = @index(Global)
    @inbounds begin
        node = node_of_pos[i]
        c = pts[sd, i]
        f = (c - gmin) * ginv
        f = f < 0.0 ? 0.0 : (f > 0.9999999999 ? 0.9999999999 : f)
        key[i] = Float64(node) + f
    end
end

# Set child segment ranges and record the split dimension for non-leaf nodes in this level.
@kernel function _build_setchildren_kernel!(seg_lo, seg_hi, split_dim, lvl_lo, sd, max_node)
    g = @index(Global)
    node = lvl_lo + g - 1
    @inbounds begin
        if node <= max_node
            lo = seg_lo[node]
            if lo != 0
                hi = seg_hi[node]
                if lo < hi && 2 * node + 1 <= max_node
                    mid = (lo + hi) >>> 1
                    split_dim[node] = sd
                    seg_lo[2 * node] = lo
                    seg_hi[2 * node] = mid
                    seg_lo[2 * node + 1] = mid + 1
                    seg_hi[2 * node + 1] = hi
                end
            end
        end
    end
end

# Reassign each position to its child node (left if position <= node median, else right).
@kernel function _build_assign_kernel!(node_of_pos, @Const(seg_lo), @Const(seg_hi), max_node)
    i = @index(Global)
    @inbounds begin
        node = node_of_pos[i]
        lo = seg_lo[node]
        hi = seg_hi[node]
        if lo < hi && 2 * node + 1 <= max_node
            mid = (lo + hi) >>> 1
            node_of_pos[i] = i <= mid ? 2 * node : 2 * node + 1
        end
    end
end

"""
    build_tree_ka(pts; backend) -> (spts, seg_lo, seg_hi, split_dim, perm)

Sort-based k-d tree build. `pts` is `(D, N)` (any element type; promoted to `Float64`), on the
given backend. Returns the reordered points `spts` `(D, N)`, the implicit-tree node arrays
`seg_lo`/`seg_hi`/`split_dim` (length `max_node`), and `perm` (tree position → original index,
1-based). Runs on GPU when `pts` is a GPU array (sort-based, one global sort per level).
"""
function build_tree_ka(pts::AbstractMatrix; backend = get_backend(pts))
    D, N = size(pts)
    spts = similar(pts, Float64, D, N)
    copyto!(spts, pts)
    Lmax = max(1, ceil(Int, log2(max(N, 2))))
    max_node = 1 << (Lmax + 1)            # > deepest leaf id (≈ 2N..4N)
    seg_lo = ka_zeros(backend, Int, max_node)
    seg_hi = ka_zeros(backend, Int, max_node)
    split_dim = ka_zeros(backend, Int, max_node)
    node_of_pos = ka_zeros(backend, Int, N)
    perm = similar(node_of_pos)
    copyto!(perm, collect(1:N))
    fill!(node_of_pos, 1)
    seg_lo[1:1] .= 1                      # broadcast assignment (no scalar indexing on GPU)
    seg_hi[1:1] .= N

    gmin = Float64(minimum(spts))
    gmax = Float64(maximum(spts))
    ginv = gmax > gmin ? 1.0 / (gmax - gmin) : 0.0
    key = ka_zeros(backend, Float64, N)
    wgs = _wgsize(backend)

    for L in 0:(Lmax - 1)
        lvl_lo = 1 << L
        nlvl = lvl_lo                      # node ids [2^L, 2^(L+1)-1] → 2^L of them
        sd = (L % D) + 1                   # round-robin split dimension for this level
        _build_key_kernel!(backend)(key, node_of_pos, spts, sd, gmin, ginv;
            ndrange = N, workgroupsize = wgs)
        synchronize(backend)
        sp = sortperm(key)                 # dispatches to the GPU sort when key is on device
        spts = spts[:, sp]
        perm = perm[sp]
        node_of_pos = node_of_pos[sp]
        _build_setchildren_kernel!(backend)(seg_lo, seg_hi, split_dim, lvl_lo, sd, max_node;
            ndrange = nlvl, workgroupsize = wgs)
        synchronize(backend)
        _build_assign_kernel!(backend)(node_of_pos, seg_lo, seg_hi, max_node;
            ndrange = N, workgroupsize = wgs)
        synchronize(backend)
    end

    return spts, seg_lo, seg_hi, split_dim, perm
end
