# Graph build pipeline: build_graph produces a GraphGPProblem directly from float points,
# removing the Python dependency for graph construction.
# Matches graphgp/graph.py: build_graph, compute_depths, order_by_depth, check_graph.

"""
    check_graph(prob::GraphGPProblem)

Validate that the graph satisfies the invariants the refinement kernels rely on:

1. `offsets[1] == n0` — the first batch is the dense/initial set, and `n0 == N - M`.
2. `offsets` is non-decreasing and `offsets[end] ≤ N`.
3. Topological order: every neighbor precedes its point (`max(neighbors[:,m]) < n0 + m`).
4. Batch causality: no point has a neighbor in its own (or a later) depth batch, so a batch
   can be generated fully in parallel.

Throws `ArgumentError` on the first violation, mirroring the asserts in
`graphgp/graph.py:check_graph`. `neighbors` is `(K, M)` 1-based; `offsets` are 1-based
exclusive batch ends with `offsets[1] = n0` and `offsets[end] = N`.
"""
function check_graph(prob::GraphGPProblem)
    N = npoints(prob)
    M = nrefined(prob)
    n0 = prob.n0
    offs = prob.offsets

    isempty(offs) && throw(ArgumentError("offsets is empty"))
    offs[1] == n0 || throw(ArgumentError(
        "offsets[1] ($(offs[1])) must equal n0 ($n0)"))
    n0 == N - M || throw(ArgumentError(
        "n0 ($n0) must equal npoints - nrefined ($(N - M))"))
    for b in 2:length(offs)
        offs[b] >= offs[b - 1] || throw(ArgumentError(
            "offsets must be non-decreasing (offsets[$b]=$(offs[b]) < offsets[$(b-1)]=$(offs[b-1]))"))
    end
    offs[end] <= N || throw(ArgumentError(
        "offsets[end] ($(offs[end])) must be ≤ npoints ($N)"))

    # Pull neighbors to the host for validation (this is an explicit, non-hot check).
    nb = Array(prob.neighbors)
    K = size(nb, 1)
    for m in 1:M
        p = n0 + m                          # 1-based position of this refined point
        mx = 0
        @inbounds for ki in 1:K
            v = nb[ki, m]
            v > mx && (mx = v)
        end
        mx < p || throw(ArgumentError(
            "point at position $p has neighbor $mx that does not precede it (not topological)"))
        bi = searchsortedfirst(offs, p)     # batch whose exclusive end is ≥ p
        prev_end = bi >= 2 ? offs[bi - 1] : 0
        mx <= prev_end || throw(ArgumentError(
            "point at position $p has a neighbor ($mx) in its own batch " *
            "(batch ends at $(offs[bi])); neighbors must be in earlier batches"))
    end
    return nothing
end

"""
    compute_depths(neighbors, n0) -> Vector{Int}

Compute the depth of each point in the Vecchia DAG. The first n0 points have depth 0.
Each refined point's depth is 1 + max(depth of its neighbors). Iterates until convergence.
Matches `graph.py:compute_depths`.

`neighbors` is (k, M) 1-based, column-major (as returned by `query_preceding_neighbors`).
Returns a length-N depth vector.
"""
function compute_depths(neighbors::AbstractMatrix{<:Integer}, n0::Int)
    backend = KernelAbstractions.get_backend(neighbors)
    if backend isa KernelAbstractions.CPU
        return _compute_depths_cpu(neighbors, n0)
    else
        return _compute_depths_ka(neighbors, n0, backend)
    end
end

function _compute_depths_cpu(neighbors::AbstractMatrix{<:Integer}, n0::Int)
    k, M = size(neighbors)
    N = n0 + M
    depths = fill(typemax(Int), N)
    for i in 1:n0
        depths[i] = 0
    end
    # Iterative relaxation: since neighbors always point backward, one pass in order
    # suffices for points already in topological order. Use repeated passes for safety.
    changed = true
    while changed
        changed = false
        for m in 1:M
            pt = n0 + m
            max_nb_depth = 0
            for ki in 1:k
                nb = neighbors[ki, m]
                nb > 0 || continue  # skip unset entries
                d = depths[nb]
                d != typemax(Int) && d > max_nb_depth && (max_nb_depth = d)
            end
            new_d = max_nb_depth + 1
            if new_d != depths[pt]
                depths[pt] = new_d
                changed = true
            end
        end
    end
    return depths
end

# One relaxation pass over all refined points; sets changed[1]=1 if any depth increased.
# Depths start at 0 and increase monotonically to the unique longest-path fixed point, so
# benign read/write races across passes only cost extra iterations (never wrong results).
@kernel function _depths_relax_kernel!(depths, @Const(neighbors), n0, changed,
        ::Val{K}) where {K}
    m = @index(Global)
    @inbounds begin
        mx = 0
        for ki in 1:K
            nb = neighbors[ki, m]
            if nb > 0
                d = depths[nb]
                d > mx && (mx = d)
            end
        end
        newd = mx + 1
        if depths[n0 + m] != newd
            depths[n0 + m] = newd
            changed[1] = 1
        end
    end
end

function _compute_depths_ka(neighbors::AbstractMatrix{<:Integer}, n0::Int, backend)
    K, M = size(neighbors)
    N = n0 + M
    depths = KernelAbstractions.zeros(backend, Int, N)   # all 0; first n0 stay 0
    changed = KernelAbstractions.zeros(backend, Int, 1)
    kernel = _depths_relax_kernel!(backend)
    iters = 0
    while true
        fill!(changed, 0)
        kernel(depths, neighbors, n0, changed, Val(K); ndrange = M,
            workgroupsize = _wgsize(backend))
        KernelAbstractions.synchronize(backend)
        Array(changed)[1] == 0 && break
        iters += 1
        iters > N && error("compute_depths did not converge (cycle in neighbor graph?)")
    end
    return depths
end

"""
    order_by_depth(points, perm, neighbors, depths) -> (points, perm, neighbors, depths)

Reorder all arrays by increasing depth. Neighbor indices are corrected to reflect the
new positions. Matches `graph.py:order_by_depth`.

`perm` maps current position → original index (1-based).
`neighbors` is (k, M) 1-based indices in the current ordering.
"""
function order_by_depth(points::Matrix{Float64}, perm::Vector{Int},
        neighbors::Matrix{Int}, depths::Vector{Int})
    N = length(perm)
    sort_perm = sortperm(depths)            # new_pos → old_pos (1-based)
    inv_sp = invperm(sort_perm)             # old_pos → new_pos

    sorted_points = points[sort_perm, :]
    sorted_perm = perm[sort_perm]
    sorted_depths = depths[sort_perm]

    k, M = size(neighbors)
    sorted_neighbors = similar(neighbors)
    n0_new = count(==(0), sorted_depths)    # number of depth-0 points
    for m in 1:M
        col_new = n0_new + m               # the column in sorted_neighbors corresponding
        # to what was col m in the old neighbors
        # Actually we need to re-map columns too.  The column index in neighbors
        # corresponds to old refined position n0+m. After reordering, that point moves
        # to new position inv_sp[n0+m] (where n0 = N-M in the old ordering).
        # Let's rebuild completely.
        break
    end
    # Rebuild neighbors from scratch in the new ordering.
    # The old n0 = N - M.
    n0_old = N - M
    n0 = n0_new
    # new_neighbors[ki, m_new] = inv_sp[old_neighbors[ki, m_old]]
    # where m_new = inv_sp[n0_old + m_old] - n0
    new_neighbors = zeros(Int, k, N - n0)
    for m_old in 1:M
        new_pos = inv_sp[n0_old + m_old]   # new 1-based position of this point
        m_new = new_pos - n0               # column in new_neighbors (must be ≥ 1)
        m_new < 1 && continue             # this point moved to the initial set (depth 0)
        for ki in 1:k
            old_nb = neighbors[ki, m_old]
            old_nb > 0 || continue
            new_neighbors[ki, m_new] = inv_sp[old_nb]
        end
    end
    return sorted_points, sorted_perm, new_neighbors, sorted_depths
end

"""
    _compute_offsets(depths) -> Vector{Int}

Compute batch offsets from a depth array (sorted in non-decreasing order).
`offsets[b]` is the 1-based exclusive end of depth-batch b-1 (so the first n0 points
at depth 0 end at offsets[1] = n0, and so on). Matches `graph.py`'s searchsorted-based
offsets computation.

Returns an `offsets` vector where `offsets[1] = n0` (count of depth-0 points) and
`offsets[end] = N`.
"""
function _compute_offsets(depths::Vector{Int})
    N = length(depths)
    max_depth = maximum(depths)
    offsets = Vector{Int}()
    for d in 0:max_depth
        idx = searchsortedlast(depths, d)
        push!(offsets, idx)
    end
    return offsets
end

"""
    quantize_to_lattice(points, bits=21) -> (coords::Matrix{UInt32}, origin, scale)

Map float points (N×D) isotropically onto a `bits`-bit integer lattice.
Returns UInt32 coordinates (N×D), the origin, and the lattice scale (physical distance
per lattice step, isotropic).
"""
function quantize_to_lattice(points::AbstractMatrix{<:Real}, bits::Int = 21)
    lmax = Float64((1 << bits) - 1)
    origin = vec(minimum(points; dims = 1))
    extents = vec(maximum(points; dims = 1)) .- origin
    extent = maximum(extents)
    scale = extent > 0 ? extent / lmax : 1.0 / lmax
    # Broadcast + reduction primitives are backend-generic, so a CuMatrix input quantizes on
    # the GPU and returns a CuMatrix{UInt32}.
    coords_f = (points .- origin') ./ scale
    coords = UInt32.(clamp.(round.(Int64, coords_f), 0, Int64(lmax)))
    return coords, origin, scale
end

"""
    build_graph(points, n0, k, bins, vals; lattice_bits=21) -> GraphGPProblem

Build a `GraphGPProblem` from scratch (float points, no Python dependency).
1. Build a k-d tree over all N points.
2. Query k preceding neighbors for each refined point (m ≥ n0+1 in tree order).
3. Compute Vecchia depths and reorder by depth.
4. Quantize to integer lattice.
5. Return a `GraphGPProblem` with the `indices` field set to the permutation.

`bins`/`vals` are the discretized covariance (from `rbf_kernel`/`matern_kernel`).
"""
function build_graph(points::Matrix{Float64}, n0::Int, k::Int,
        bins::AbstractVector, vals::AbstractVector;
        lattice_bits::Int = 21)
    N = size(points, 1)
    @assert n0 < N "n0 must be less than N"
    @assert k > 0 "k must be positive"

    # Step 1: build k-d tree (tree order).
    sorted_pts, seg_lo, seg_hi, split_dim, tree_perm = build_tree(points)

    # Step 2: query preceding neighbors in tree order. Use the KernelAbstractions query
    # (precomputed per-node AABBs) instead of the scalar O(N^2) reference: same neighbor sets,
    # far cheaper, and GPU-capable when the tree arrays live on a device.
    spts = permutedims(sorted_pts)  # (D, N)
    neighbors = Matrix{Int}(query_preceding_neighbors_ka(spts, seg_lo, seg_hi, split_dim, n0, k))

    # Step 3: depth ordering.
    depths = compute_depths(neighbors, n0)
    sorted_pts, tree_perm, neighbors, depths = order_by_depth(sorted_pts, tree_perm, neighbors, depths)
    n0_final = count(==(0), depths)

    # Step 4: quantize reordered points.
    coords_rowmajor, _, scale = quantize_to_lattice(sorted_pts, lattice_bits)
    coords = Matrix{UInt32}(permutedims(coords_rowmajor))  # (D, N) column-major

    # Step 5: build offsets.
    offsets = _compute_offsets(depths)

    # The permutation: tree_perm[new_pos] = original_1based_index.
    # For GraphGPProblem.indices semantics: indices[tree_pos] = orig_pos.
    return GraphGPProblem(coords, neighbors, offsets, n0_final, eltype(vals)(scale),
        bins, vals, tree_perm)
end
