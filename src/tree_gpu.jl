# Backend-agnostic (KernelAbstractions) k-d tree neighbor query — Workstream C, phase C2.
#
# The scalar CPU query in tree.jl recomputes a per-node bounding box by scanning the node's
# segment at *every* visited node (O(N) at the root → O(N²) overall). That is a fine
# correctness reference but does not scale. Here we instead:
#   1. precompute a static per-node AABB once (one workitem per node), and
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

# Per-node axis-aligned bounding box over the node's full segment (Float64 coords, D×N layout).
@kernel function _node_aabb_kernel!(node_min, node_max, @Const(spts),
        @Const(seg_lo), @Const(seg_hi), ::Val{D}) where {D}
    node = @index(Global)
    @inbounds begin
        lo = seg_lo[node]
        if lo == 0
            for d in 1:D
                node_min[d, node] = Inf
                node_max[d, node] = -Inf
            end
        else
            hi = seg_hi[node]
            for d in 1:D
                mn = Inf
                mx = -Inf
                for i in lo:hi
                    v = spts[d, i]
                    v < mn && (mn = v)
                    v > mx && (mx = v)
                end
                node_min[d, node] = mn
                node_max[d, node] = mx
            end
        end
    end
end

# One workitem per refined point t = 1..M (query tree position m = n0 + t). Finds the k
# nearest preceding (position < m) points and writes their 1-based tree indices, ascending
# by distance, into neighbors[:, t].
@kernel function _query_kernel!(neighbors, @Const(spts), @Const(seg_lo), @Const(seg_hi),
        @Const(split_dim), @Const(node_min), @Const(node_max), n0, max_node,
        ::Val{K}, ::Val{D}) where {K, D}
    t = @index(Global)
    m = n0 + t
    bd = @private Float64 (K,)     # k-best squared distances
    bi = @private Int (K,)         # k-best point indices (1-based)
    stack = @private Int (_QSTACK,)
    @inbounds begin
        for j in 1:K
            bd[j] = Inf
            bi[j] = 0
        end
        cnt = 0
        worst = Inf
        sp = 1
        stack[1] = 1
        while sp >= 1
            node = stack[sp]
            sp -= 1
            (node < 1 || node > max_node) && continue
            lo = seg_lo[node]
            lo == 0 && continue
            # Index-range skip: points are in tree order and a node's subtree occupies the
            # contiguous range [seg_lo, seg_hi], so seg_lo >= m means the whole subtree
            # post-dates the query point m → no admissible preceding neighbor. This is the
            # decisive prune for the Vecchia "preceding neighbors" constraint.
            lo >= m && continue
            hi = seg_hi[node]

            # AABB pruning (only once the buffer is full).
            if cnt == K
                msq = 0.0
                for d in 1:D
                    q = spts[d, m]
                    nmn = node_min[d, node]
                    nmx = node_max[d, node]
                    c = q < nmn ? nmn : (q > nmx ? nmx : q)
                    dl = c - q
                    msq += dl * dl
                end
                msq >= worst && continue
            end

            if lo == hi
                (lo < m) || continue
                sq = 0.0
                for d in 1:D
                    dv = spts[d, lo] - spts[d, m]
                    sq += dv * dv
                end
                if cnt < K
                    cnt += 1
                    bd[cnt] = sq
                    bi[cnt] = lo
                    if cnt == K
                        w = -Inf
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
                    w2 = -Inf
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
            mid = (lo + hi) ÷ 2
            sd = split_dim[node]
            splitv = spts[sd, mid]
            qsd = spts[sd, m]
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
`(k, M)` matrix (`M = N - n0`), matching the set produced by the scalar `query_preceding_neighbors`.

Runs on CPU or GPU depending on the array backend (e.g. pass `CuArray`s for the GPU path).
"""
function query_preceding_neighbors_ka(spts::AbstractMatrix{Float64},
        seg_lo::AbstractVector{<:Integer}, seg_hi::AbstractVector{<:Integer},
        split_dim::AbstractVector{<:Integer}, n0::Int, k::Int;
        backend = get_backend(spts))
    D, N = size(spts)
    M = N - n0
    max_node = length(seg_lo)
    node_min = ka_zeros(backend, Float64, D, max_node)
    node_max = ka_zeros(backend, Float64, D, max_node)
    _node_aabb_kernel!(backend)(node_min, node_max, spts, seg_lo, seg_hi, Val(D);
        ndrange = max_node, workgroupsize = _wgsize(backend))
    synchronize(backend)
    neighbors = ka_zeros(backend, Int, k, M)
    _query_kernel!(backend)(neighbors, spts, seg_lo, seg_hi, split_dim, node_min, node_max,
        n0, max_node, Val(k), Val(D); ndrange = M, workgroupsize = _wgsize(backend))
    synchronize(backend)
    return neighbors
end

# ── GPU/KA k-d tree build (sort-based, level by level) ────────────────────────────────────
#
# The CPU `build_tree` (tree.jl) does a sequential BFS median split. The GPU build instead
# mirrors the data-parallel JAX `_build_tree`: at each level, every node splits at the median
# of its widest dimension, realized as a single global sort of all points by a composite key
# (node-id in the integer part, normalized split coordinate in the fraction). One sort per
# level → O(N log²N) total. Points keep descending so all live positions stay at the current
# level (singletons descend the left spine). `sortperm`/`minimum` dispatch to the GPU backend
# at run time, so this needs no compile-time CUDA dependency; it also runs on the KA CPU
# backend. The resulting tree is a valid k-d tree (not byte-identical to the CPU build —
# tie-breaking and the fractional key differ — but neighbor sets are equivalent).

# Per-node widest-dimension split selection for node ids in [lvl_lo, lvl_hi].
@kernel function _build_splitdim_kernel!(split_dim, @Const(seg_lo), @Const(seg_hi),
        @Const(pts), lvl_lo, max_node, ::Val{D}) where {D}
    g = @index(Global)
    node = lvl_lo + g - 1
    @inbounds begin
        if node <= max_node
            lo = seg_lo[node]
            if lo != 0
                hi = seg_hi[node]
                if lo == hi
                    split_dim[node] = 1
                else
                    best_d = 1
                    best_range = -Inf
                    for d in 1:D
                        mn = Inf
                        mx = -Inf
                        for i in lo:hi
                            v = pts[d, i]
                            v < mn && (mn = v)
                            v > mx && (mx = v)
                        end
                        rng = mx - mn
                        if rng > best_range
                            best_range = rng
                            best_d = d
                        end
                    end
                    split_dim[node] = best_d
                end
            end
        end
    end
end

# Composite sort key per position: node id (integer part) + normalized split coordinate (frac).
@kernel function _build_key_kernel!(key, @Const(node_of_pos), @Const(split_dim), @Const(pts),
        gmin, ginv)
    i = @index(Global)
    @inbounds begin
        node = node_of_pos[i]
        sd = split_dim[node]
        c = pts[sd, i]
        f = (c - gmin) * ginv
        f = f < 0.0 ? 0.0 : (f > 0.9999999999 ? 0.9999999999 : f)
        key[i] = Float64(node) + f
    end
end

# Set child segment ranges for non-leaf nodes in [lvl_lo, lvl_hi].
@kernel function _build_setchildren_kernel!(seg_lo, seg_hi, lvl_lo, max_node)
    g = @index(Global)
    node = lvl_lo + g - 1
    @inbounds begin
        if node <= max_node
            lo = seg_lo[node]
            if lo != 0
                hi = seg_hi[node]
                if lo < hi && 2 * node + 1 <= max_node
                    mid = (lo + hi) >>> 1
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
        _build_splitdim_kernel!(backend)(split_dim, seg_lo, seg_hi, spts, lvl_lo, max_node,
            Val(D); ndrange = nlvl, workgroupsize = wgs)
        synchronize(backend)
        _build_key_kernel!(backend)(key, node_of_pos, split_dim, spts, gmin, ginv;
            ndrange = N, workgroupsize = wgs)
        synchronize(backend)
        sp = sortperm(key)                 # dispatches to the GPU sort when key is on device
        spts = spts[:, sp]
        perm = perm[sp]
        node_of_pos = node_of_pos[sp]
        _build_setchildren_kernel!(backend)(seg_lo, seg_hi, lvl_lo, max_node;
            ndrange = nlvl, workgroupsize = wgs)
        synchronize(backend)
        _build_assign_kernel!(backend)(node_of_pos, seg_lo, seg_hi, max_node;
            ndrange = N, workgroupsize = wgs)
        synchronize(backend)
    end

    return spts, seg_lo, seg_hi, split_dim, perm
end
