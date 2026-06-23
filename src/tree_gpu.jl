# Backend-agnostic (KernelAbstractions) k-d tree neighbor query — Workstream C, phase C2.
#
# The scalar CPU query in tree.jl recomputes a per-node bounding box by scanning the node's
# segment at *every* visited node (O(N) at the root → O(N²) overall). That is a fine
# correctness reference but does not scale. Here we instead:
#   1. precompute a static per-node AABB once (one workitem per node), and
#   2. run one workitem per query point with a private fixed-size k-best buffer and an explicit
#      DFS stack, pruning against the static AABB.
# The static (full-segment) AABB is a looser-but-valid lower bound than the CPU's
# preceding-only box, so it never prunes a subtree that could hold a true neighbor — the exact
# k-NN set is identical to the scalar reference. Leaf acceptance still enforces `lo < m`.
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
