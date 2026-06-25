# Faithful port of the JAX reference k-d tree (graphgp/tree.py): the "special order"
# heap-layout build (`_build_tree`) and the stackless preceding-neighbour query
# (`query_preceding_neighbors` / `_single_query_neighbors` / `_traverse_tree`).
#
# The point of this layout (vs the leaf-order tree in tree.jl) is the *array order*: a
# point's final array position equals its heap node id, so the first n0 points are the
# spatially-spread tree medians. That produces a SHALLOW Vecchia DAG (few depth batches),
# and — built operation-for-operation against the JAX code — a graph IDENTICAL to Python's
# `gp.build_graph(points, n0, k)` (the pure-JAX path), down to neighbour indices, offsets,
# and the tree permutation (for generic float point clouds, where k-NN ties are measure-zero).
#
# Heap layout (0-based node ids): root = 0; for a node `c` at level L (= floor(log2(c+1))),
# left(c)  = c + 2^L,  right(c) = c + 2·2^L. Node ids fill 0..N-1 contiguously and the final
# array is sorted by node id, so position i ↔ node id i-1 (Julia 1-based).

# floor(log2(x)) for x ≥ 1, exact via bit length.
@inline _flevel(x::Integer) = (8 * sizeof(Int) - leading_zeros(Int(x))) - 1

@inline function _sp_parent(c::Int)
    c == 0 && return -1
    level = _flevel(c + 1)
    n_above = (1 << level) - 1
    n_parent_level = 1 << (level - 1)
    return c < n_above + n_parent_level ? c - n_parent_level : c - 2 * n_parent_level
end
@inline function _sp_left(c::Int)
    level = _flevel(c + 1)
    return c + (1 << level)
end
@inline function _sp_right(c::Int)
    level = _flevel(c + 1)
    return c + 2 * (1 << level)
end

# ── Threaded per-level helpers (kept as top-level functions, NOT closures inside
# build_tree_special: the working `pts`/`indices` buffers are reassigned by buffer-swapping each
# level, and capturing a reassigned variable in a @threads closure would box it → type-unstable,
# allocating. Passing the arrays as arguments keeps every kernel type-stable.) ──

# Per-node segment max/min over the current points. Serial (it is a keyed reduction; cheap O(N·D)).
function _bt_segminmax!(seg_max, seg_min, pts, nodes, N, D)
    fill!(seg_max, -Inf)
    fill!(seg_min, Inf)
    @inbounds for i in 1:N
        s = nodes[i] + 1
        for d in 1:D
            v = pts[i, d]
            v > seg_max[s, d] && (seg_max[s, d] = v)
            v < seg_min[s, d] && (seg_min[s, d] = v)
        end
    end
end

function _bt_splitdims!(split_dims, seg_max, seg_min, n_above, N, D)
    Threads.@threads :static for i in 1:N
        (i - 1) < n_above && continue
        bestd = 1
        bestr = -Inf
        @inbounds for d in 1:D
            r = seg_max[i, d] - seg_min[i, d]
            r > bestr && (bestr = r; bestd = d)
        end
        @inbounds split_dims[i] = bestd
    end
end

function _bt_pad!(pad, pts, split_dims, nodes, N)
    Threads.@threads :static for i in 1:N
        @inbounds pad[i] = pts[i, split_dims[nodes[i] + 1]]
    end
end

# Apply the sort permutation p: gather pts/nodes/indices into scratch (disjoint output rows).
function _bt_gather!(pts_s, pts, nodes_s, nodes, indices_s, indices, p, N, D)
    Threads.@threads :static for i in 1:N
        @inbounds begin
            pp = p[i]
            for d in 1:D
                pts_s[i, d] = pts[pp, d]
            end
            nodes_s[i] = nodes[pp]
            indices_s[i] = indices[pp]
        end
    end
end

function _bt_update!(nodes_s, nodes, n_above, n_level, q, r, N)
    Threads.@threads :static for i in 1:N
        @inbounds begin
            idx = i - 1
            s = nodes[i]
            ii = s - n_above
            mid = (ii < r ? ii * (q + 1) + div(q + 1, 2) :
                            r * (q + 1) + (ii - r) * q + div(q, 2)) + n_above
            nodes_s[i] = (idx < n_above || idx == mid) ? s :
                         (idx < mid ? s + n_level : s + 2 * n_level)
        end
    end
end

# Merge sorted runs src[lo+1:mid] and src[mid+1:hi] into dst[lo+1:hi] by `lt` (left wins ties →
# deterministic with a total-order comparator, so the result is identical to a serial sort).
@inline function _bt_merge!(dst, src, lo, mid, hi, lt)
    i = lo + 1; j = mid + 1; ko = lo + 1
    @inbounds while i <= mid && j <= hi
        if lt(src[j], src[i])
            dst[ko] = src[j]; j += 1
        else
            dst[ko] = src[i]; i += 1
        end
        ko += 1
    end
    @inbounds while i <= mid; dst[ko] = src[i]; i += 1; ko += 1; end
    @inbounds while j <= hi;  dst[ko] = src[j]; j += 1; ko += 1; end
end

# Parallel merge-sort of the index vector `p` by `lt`: sort `nchunks` (power-of-two) chunks in
# parallel, then pairwise-merge in parallel rounds (ping-ponging p ↔ scratch). Falls back to a
# serial QuickSort below a size/thread threshold. Result lands back in `p`.
function _bt_psort!(p::Vector{Int}, scratch::Vector{Int}, lt, nt::Int)
    N = length(p)
    if nt <= 1 || N < 8192
        sort!(p; lt = lt, alg = QuickSort)
        return p
    end
    nchunks = 1 << min(floor(Int, log2(nt)), floor(Int, log2(max(N ÷ 4096, 1))))
    nchunks < 2 && (sort!(p; lt = lt, alg = QuickSort); return p)
    bnd = [(c * N) ÷ nchunks for c in 0:nchunks]            # 0-based run boundaries
    Threads.@threads :static for c in 1:nchunks
        lo = bnd[c] + 1; hi = bnd[c + 1]
        lo <= hi && sort!(view(p, lo:hi); lt = lt, alg = QuickSort)
    end
    src = p; dst = scratch; b = bnd; csz = nchunks
    while csz > 1
        half = csz ÷ 2
        Threads.@threads :static for jj in 1:half
            _bt_merge!(dst, src, b[2jj - 1], b[2jj], b[2jj + 1], lt)
        end
        b = b[1:2:end]
        src, dst = dst, src
        csz = half
    end
    src === p || copyto!(p, src)
    return p
end

"""
    build_tree_special(points::Matrix{Float64}) -> (sorted_points, split_dims, perm)

Build the JAX "special order" heap-layout k-d tree (matches `tree.py:_build_tree`).

- `sorted_points` — points in tree (heap) order, `(N, D)`. Position i is heap node id i-1.
- `split_dims`    — split dimension (1-based) of each heap node, length N (node-id indexed).
- `perm`          — `perm[tree_pos] = original 1-based index` of that point.

Implementation: faithful replica of the JAX `lax.scan` over levels — `n_levels` global passes,
each (a) picking each segment's split dim (widest range), (b) sorting all points by
`(node id, coord-along-split, position)`, and (c) relabeling via `_update_nodes`, which
distributes a level's points *evenly by node id* (the first `r` nodes get the extra point). That
even-by-id redistribution does not decompose into independent per-subtree recursion (children of
a node have non-adjacent ids, so the "extra point" membership does not compose), which is why the
global per-level form is kept rather than a recursive quickselect. For generic float clouds it is
byte-identical to JAX (split ties are measure-zero).
"""
function build_tree_special(points::Matrix{Float64})
    N, D = size(points)
    n_levels = N > 0 ? (_flevel(N) + 1) : 0     # N.bit_length()

    # `nodes` and `pad` are captured by the sort comparator below, so they are assigned EXACTLY
    # ONCE and only ever mutated in place (via the scratch buffers + copyto!). Reassigning a
    # captured variable would box it (`Core.Box`), making every sort comparison type-unstable and
    # heap-allocating — that was ~16 GB of garbage and ~90% of build time. `pts`/`indices` are not
    # captured, so they are cheaply double-buffered by swapping.
    pts = copy(points)                          # pts[i,:] = point currently at position i
    pts_s = similar(pts)                        # scratch for the permuted gather
    nodes = zeros(Int, N)                       # 0-based heap node id of point at position i
    nodes_s = Vector{Int}(undef, N)             # scratch (permute + _update_nodes)
    indices = collect(1:N)                      # original 1-based index of point at position i
    indices_s = Vector{Int}(undef, N)
    split_dims = fill(-1, N)                    # node-id-indexed (NEVER permuted by the sorts)

    seg_max = Matrix{Float64}(undef, N, D)
    seg_min = Matrix{Float64}(undef, N, D)
    pad = Vector{Float64}(undef, N)
    p = Vector{Int}(undef, N)
    p_s = Vector{Int}(undef, N)                 # scratch for the parallel merge-sort
    nt = Threads.nthreads()
    # Non-allocating lexicographic comparator for (nodes, points_along_dim, position). Total order
    # (the `a < b` tie-break), so the parallel merge-sort is deterministic = the serial order.
    # `nodes`/`pad` are assigned once (only mutated in place), so this closure is type-stable.
    lt = (a, b) -> begin
        @inbounds begin
            na = nodes[a]; nb = nodes[b]
            na != nb && return na < nb
            pa = pad[a]; pb = pad[b]
            pa != pb && return pa < pb
            return a < b
        end
    end

    for level in 0:(n_levels - 1)
        n_above = (1 << level) - 1
        n_level = 1 << level

        _bt_segminmax!(seg_max, seg_min, pts, nodes, N, D)
        _bt_splitdims!(split_dims, seg_max, seg_min, n_above, N, D)
        _bt_pad!(pad, pts, split_dims, nodes, N)

        @inbounds for i in 1:N
            p[i] = i
        end
        _bt_psort!(p, p_s, lt, nt)               # parallel sort by (nodes, pad, position)

        _bt_gather!(pts_s, pts, nodes_s, nodes, indices_s, indices, p, N, D)
        pts, pts_s = pts_s, pts                  # not captured → swap is free
        indices, indices_s = indices_s, indices  # not captured → swap is free
        copyto!(nodes, nodes_s)                  # captured → mutate in place, never rebind

        n_remaining = N - n_above
        q = div(n_remaining, n_level)
        r = rem(n_remaining, n_level)
        _bt_update!(nodes_s, nodes, n_above, n_level, q, r, N)
        copyto!(nodes, nodes_s)
    end

    return pts, split_dims, indices
end

# Exact replica of _single_query_neighbors + _traverse_tree for one query point.
# `cand_d` / `cand_i` are length-k scratch arrays (squared distances / node ids), reused per call.
function _sp_query_one!(cand_d::Vector{Float64}, cand_i::Vector{Int},
        sorted_pts::Matrix{Float64}, split_dims::Vector{Int},
        query::Int, max_index::Int, k::Int, D::Int)
    @inbounds for j in 1:k
        cand_d[j] = Inf
        cand_i[j] = -1
    end
    square_radius = Inf
    qrow = query + 1                              # query node id -> 1-based position

    current = 0
    previous = -1
    @inbounds while current != -1
        parent = _sp_parent(current)

        # Update candidates on first arrival (came from parent).
        if previous == parent
            crow = current + 1
            sq = 0.0
            for d in 1:D
                dv = sorted_pts[qrow, d] - sorted_pts[crow, d]
                sq += dv * dv
            end
            # evict the current worst (argmax) iff strictly closer — matches JAX exactly.
            wpos = 1
            wval = cand_d[1]
            for j in 2:k
                if cand_d[j] > wval
                    wval = cand_d[j]
                    wpos = j
                end
            end
            if sq < wval
                cand_d[wpos] = sq
                cand_i[wpos] = current
                # new square_radius = max over candidates
                m = cand_d[1]
                for j in 2:k
                    cand_d[j] > m && (m = cand_d[j])
                end
                square_radius = m
            else
                square_radius = wval
            end
        end

        sd = split_dims[current + 1]
        split_distance = sorted_pts[qrow, sd] - sorted_pts[current + 1, sd]
        near = split_distance < 0 ? _sp_left(current) : _sp_right(current)
        far = split_distance < 0 ? _sp_right(current) : _sp_left(current)
        far_in_range = split_distance * split_distance <= square_radius

        local nxt::Int
        if (previous == near) || ((previous == parent) && (near >= max_index))
            nxt = ((far < max_index) && far_in_range) ? far : parent
        else
            nxt = (previous == parent) ? near : parent
        end
        previous = current
        current = nxt
    end
    return nothing
end

"""
    query_preceding_neighbors_special(sorted_pts, split_dims, n0, k) -> Matrix{Int}

For each refined point m = n0+1 … N (1-based tree position), find its k nearest neighbours
among preceding tree positions (node id < m-1, i.e. `max_index = m-1` 0-based), via the exact
JAX stackless heap traversal. Returns a `(k, N-n0)` matrix of 1-based tree-position indices,
each column sorted by ascending distance (ties broken by index), matching `tree.py`.
"""
function query_preceding_neighbors_special(sorted_pts::Matrix{Float64}, split_dims::Vector{Int},
        n0::Int, k::Int)
    N = size(sorted_pts, 1)
    D = size(sorted_pts, 2)
    M = N - n0
    neighbors = zeros(Int, k, M)
    # Each query point is independent (read-only tree traversal, disjoint output column), so the
    # M queries run in parallel. Chunk across threads with per-thread scratch (the sort `by` closure
    # captures the per-thread cand_* — assigned once per task, so it stays type-stable).
    nt = max(1, min(Threads.nthreads(), M))
    chunks = collect(Iterators.partition((n0 + 1):N, cld(M, nt)))
    Threads.@threads :static for chunk in chunks
        cand_d = Vector{Float64}(undef, k)
        cand_i = Vector{Int}(undef, k)
        order = Vector{Int}(undef, k)
        for m in chunk
            query = m - 1                        # 0-based node id of this point
            max_index = query                    # preceding: nodes strictly < query
            _sp_query_one!(cand_d, cand_i, sorted_pts, split_dims, query, max_index, k, D)
            # sort candidates by (distance, index) ascending — JAX lax.sort num_keys=2.
            sortperm!(order, 1:k; by = j -> (cand_d[j], cand_i[j]))
            col = m - n0
            @inbounds for j in 1:k
                ci = cand_i[order[j]]
                neighbors[j, col] = ci + 1       # 0-based node id -> 1-based tree position
            end
        end
    end
    return neighbors
end
