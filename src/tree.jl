# Pure Julia k-d tree for graph build.
#
# The JAX implementation (tree.py) uses lax.scan/lax.while_loop purely to satisfy
# JAX's "no Python loops in JIT" constraint. Here we use plain Julia loops.
#
# `build_tree`  — level-by-level implicit binary k-d tree, matches tree.py semantics.
# `query_preceding_neighbors` — k-NN among preceding points via tree traversal.
#
# Convention: points are (N, D) Float64; all returned indices are 1-based.
# The implicit binary tree is 1-based: left(i)=2i, right(i)=2i+1, parent(i)=i÷2.

# ── Max-heap helpers (size k, tracks squared distances) ──────────────────────────────

mutable struct MaxHeap
    dists::Vector{Float64}   # squared distances
    idxs::Vector{Int}        # point indices
    size::Int
    capacity::Int
end

MaxHeap(k::Int) = MaxHeap(fill(Inf, k), zeros(Int, k), 0, k)

function _heap_sift_down!(h::MaxHeap, pos::Int)
    while true
        left = 2 * pos
        right = 2 * pos + 1
        largest = pos
        if left <= h.size && h.dists[left] > h.dists[largest]
            largest = left
        end
        if right <= h.size && h.dists[right] > h.dists[largest]
            largest = right
        end
        largest == pos && break
        h.dists[pos], h.dists[largest] = h.dists[largest], h.dists[pos]
        h.idxs[pos], h.idxs[largest] = h.idxs[largest], h.idxs[pos]
        pos = largest
    end
end

function _heap_sift_up!(h::MaxHeap, pos::Int)
    while pos > 1
        parent = pos ÷ 2
        h.dists[parent] >= h.dists[pos] && break
        h.dists[pos], h.dists[parent] = h.dists[parent], h.dists[pos]
        h.idxs[pos], h.idxs[parent] = h.idxs[parent], h.idxs[pos]
        pos = parent
    end
end

function heap_push!(h::MaxHeap, dist::Float64, idx::Int)
    if h.size < h.capacity
        h.size += 1
        h.dists[h.size] = dist
        h.idxs[h.size] = idx
        _heap_sift_up!(h, h.size)
    elseif dist < h.dists[1]
        h.dists[1] = dist
        h.idxs[1] = idx
        _heap_sift_down!(h, 1)
    end
end

heap_worst(h::MaxHeap) = h.dists[1]  # worst (largest) squared distance in heap

function heap_to_sorted(h::MaxHeap)
    n = h.size
    idxs = h.idxs[1:n]
    dists = h.dists[1:n]
    perm = sortperm(dists)
    return idxs[perm]
end

# ── Build tree ───────────────────────────────────────────────────────────────────────

"""
    build_tree(points::Matrix{Float64}) -> (sorted_points, seg_lo, seg_hi, split_dim, perm)

Build an implicit 1-based binary k-d tree over N points (N×D matrix).
Returns:
- `sorted_points`: points reordered into tree order (N×D)
- `seg_lo`, `seg_hi`: 1-based lo/hi into `perm` for each implicit tree node (length N)
- `split_dim`: split dimension for each tree node (1-based, length N)
- `perm`: permutation mapping tree position → original 1-based index

The root is node 1 covering all N points. Left child of node i = 2i, right = 2i+1.
At each internal node, we split at the median along the dimension with maximum range.
Matches the semantics of `tree.py:build_tree` (level-by-level median split).
"""
function build_tree(points::Matrix{Float64})
    N, D = size(points)
    perm = collect(1:N)        # tree_pos → orig_idx (1-based)
    seg_lo = zeros(Int, N)
    seg_hi = zeros(Int, N)
    split_dim = zeros(Int, N)

    # BFS queue: (node_id, range_lo, range_hi) in perm array
    queue = Vector{Tuple{Int, Int, Int}}()
    push!(queue, (1, 1, N))
    qi = 1

    while qi <= length(queue)
        node, lo, hi = queue[qi]
        qi += 1
        seg_lo[node] = lo
        seg_hi[node] = hi

        if lo == hi
            split_dim[node] = 1  # leaf: split_dim unused but set for completeness
            continue
        end

        # Find the dimension with maximum range within this segment.
        best_d = 1
        best_range = -Inf
        for d in 1:D
            mn = Inf
            mx = -Inf
            for pi in lo:hi
                v = points[perm[pi], d]
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

        # Sort perm[lo:hi] by points[:, best_d] (stable sort within segment).
        sort!(view(perm, lo:hi), by = pi -> points[pi, best_d])

        mid = (lo + hi) ÷ 2
        left = 2 * node
        right = 2 * node + 1
        if left <= N
            push!(queue, (left, lo, mid))
        end
        if right <= N
            push!(queue, (right, mid + 1, hi))
        end
    end

    sorted_points = points[perm, :]
    return sorted_points, seg_lo, seg_hi, split_dim, perm
end

# ── Preceding-neighbor query ─────────────────────────────────────────────────────────

# Traverse the implicit k-d tree for point at tree_pos `m` (1-based) to find its k
# nearest neighbors among tree positions 1..m-1.
function _query_one!(heap::MaxHeap, sorted_pts::Matrix{Float64}, seg_lo::Vector{Int},
        seg_hi::Vector{Int}, split_dim::Vector{Int}, m::Int, k::Int, N::Int)
    qpt = view(sorted_pts, m, :)  # query point (D-vector)
    D = size(sorted_pts, 2)

    # Stack-based DFS over tree nodes.
    stack = Vector{Int}()
    push!(stack, 1)  # start at root

    while !isempty(stack)
        node = pop!(stack)
        node > N && continue
        lo = seg_lo[node]
        hi = seg_hi[node]
        lo == 0 && continue  # uninitialized node (tree is not full binary)

        # Bounding-box pruning: compute min squared distance from qpt to the bounding box
        # of this node's point set.
        min_sq = 0.0
        for d in 1:D
            mn = Inf
            mx = -Inf
            for pi in lo:hi
                v = sorted_pts[pi, d]
                v < mn && (mn = v)
                v > mx && (mx = v)
            end
            delta = clamp(qpt[d], mn, mx) - qpt[d]
            min_sq += delta * delta
        end

        # Prune if bounding box is farther than current k-th neighbor.
        heap.size == k && min_sq >= heap_worst(heap) && continue

        if lo == hi
            # Leaf: check this point if it precedes m.
            lo < m || continue
            sq = 0.0
            for d in 1:D
                dv = sorted_pts[lo, d] - qpt[d]
                sq += dv * dv
            end
            heap_push!(heap, sq, lo)
            continue
        end

        # Internal node: push both children (right first so left is popped first).
        left = 2 * node
        right = 2 * node + 1
        mid = (lo + hi) ÷ 2

        # Check split dimension to decide which child to visit first.
        sd = split_dim[node]
        split_val = sorted_pts[mid, sd]
        if qpt[sd] <= split_val
            right <= N && push!(stack, right)
            left <= N && push!(stack, left)
        else
            left <= N && push!(stack, left)
            right <= N && push!(stack, right)
        end
    end
end

"""
    query_preceding_neighbors(sorted_pts, seg_lo, seg_hi, split_dim, n0, k) -> Matrix{Int}

For each refined point m from n0+1 to N (1-based), find its k nearest neighbors among
tree positions 1..m-1. Returns a (k, N-n0) matrix of 1-based tree-position indices.

Only points with tree position < m are considered ("preceding neighbors"), enforcing
the causal structure required by the Vecchia approximation.
"""
function query_preceding_neighbors(sorted_pts::Matrix{Float64}, seg_lo::Vector{Int},
        seg_hi::Vector{Int}, split_dim::Vector{Int}, n0::Int, k::Int)
    N = size(sorted_pts, 1)
    M = N - n0
    neighbors = zeros(Int, k, M)

    heap = MaxHeap(k)
    for m in (n0 + 1):N
        # Reset heap.
        heap.size = 0
        fill!(heap.dists, Inf)
        fill!(heap.idxs, 0)

        _query_one!(heap, sorted_pts, seg_lo, seg_hi, split_dim, m, k, N)

        # Store sorted neighbors (by distance) for this point.
        col = m - n0
        nb = heap_to_sorted(heap)
        for i in eachindex(nb)
            neighbors[i, col] = nb[i]
        end
    end
    return neighbors
end
