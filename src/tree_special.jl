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

    pts = copy(points)                          # pts[i,:] = point currently at position i
    nodes = zeros(Int, N)                       # 0-based heap node id of point at position i
    indices = collect(1:N)                      # original 1-based index of point at position i
    split_dims = fill(-1, N)                    # node-id-indexed (NEVER permuted by the sorts)

    seg_max = Matrix{Float64}(undef, N, D)
    seg_min = Matrix{Float64}(undef, N, D)
    pad = Vector{Float64}(undef, N)
    p = Vector{Int}(undef, N)
    # Non-allocating lexicographic comparator for (nodes, points_along_dim, position).
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

        # segment_max / segment_min over current points, keyed by node id.
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

        # split_dims[i] = where(array_index < n_above, split_dims[i], argmax_d range(segment i))
        # (array_index is 0-based position i-1; segment id == array_index here).
        @inbounds for i in 1:N
            (i - 1) < n_above && continue
            bestd = 1
            bestr = -Inf
            for d in 1:D
                r = seg_max[i, d] - seg_min[i, d]
                if r > bestr
                    bestr = r
                    bestd = d
                end
            end
            split_dims[i] = bestd
        end

        # points_along_dim[i] = points[i, split_dims[nodes[i]]]
        @inbounds for i in 1:N
            sd = split_dims[nodes[i] + 1]
            pad[i] = pts[i, sd]
        end

        # Sort by (nodes, points_along_dim, array_index); array_index (= position) is the
        # stable 3rd key (total order, so the unstable sort is deterministic). Apply the
        # permutation to pts / nodes / indices (NOT to split_dims).
        @inbounds for i in 1:N
            p[i] = i
        end
        sort!(p; lt = lt)
        pts = pts[p, :]
        nodes = nodes[p]
        indices = indices[p]

        # _update_nodes: keep each segment's median at its node id, send earlier points to the
        # left child id and later points to the right child id.
        n_remaining = N - n_above
        q = div(n_remaining, n_level)
        r = rem(n_remaining, n_level)
        newnodes = Vector{Int}(undef, N)
        @inbounds for i in 1:N
            idx = i - 1                          # array_index (0-based position)
            s = nodes[i]
            ii = s - n_above
            mid = (ii < r ? ii * (q + 1) + div(q + 1, 2) :
                            r * (q + 1) + (ii - r) * q + div(q, 2)) + n_above
            if idx < n_above || idx == mid
                newnodes[i] = s
            elseif idx < mid
                newnodes[i] = s + n_level
            else
                newnodes[i] = s + 2 * n_level
            end
        end
        nodes = newnodes
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
    cand_d = Vector{Float64}(undef, k)
    cand_i = Vector{Int}(undef, k)
    order = Vector{Int}(undef, k)
    for m in (n0 + 1):N
        query = m - 1                            # 0-based node id of this point
        max_index = query                        # preceding: nodes strictly < query
        _sp_query_one!(cand_d, cand_i, sorted_pts, split_dims, query, max_index, k, D)
        # sort candidates by (distance, index) ascending — JAX lax.sort num_keys=2.
        sortperm!(order, 1:k; by = j -> (cand_d[j], cand_i[j]))
        col = m - n0
        @inbounds for j in 1:k
            ci = cand_i[order[j]]
            neighbors[j, col] = ci + 1           # 0-based node id -> 1-based tree position
        end
    end
    return neighbors
end
