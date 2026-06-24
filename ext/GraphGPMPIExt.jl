module GraphGPMPIExt

# MPI backing for the distributed GraphGP layer (Phase 1: distributed log-likelihood + vals
# gradient for hyperparameter fitting). Activated by `using GraphGP, MPI`.
#
# Partition scheme :replicate_coords — every rank holds the full `coords` and a balanced
# contiguous column slice of `neighbors`. The per-rank COMPUTE runs on whatever backend the
# problem lives on (CPU or GPU); the reduced PAYLOAD is tiny (a Float64 scalar + an `nbins`
# vector) and is brought to the host in the core methods, so the Allreduce is a plain host
# reduction — no CUDA-aware MPI is required for Phase 1. (GPU device binding is a launch-time
# concern handled in the run scripts, before the problem is moved to the device.)

using GraphGP
using MPI
import GraphGP: distribute, _dist_allreduce_sum, _dist_allreduce_sum!, _dist_allgather_columns,
    _dist_allreduce_min!, _dist_allreduce_max!, distributed_build_graph, distributed_quantize,
    DistributedGraphGPProblem, GraphGPProblem, nrefined,
    build_tree_ka, query_preceding_neighbors_ka, quantize_to_lattice, _move_to_backend,
    KernelAbstractions

# Balanced contiguous split of 1:M across `nranks`: the first `rem` ranks get one extra column.
# Returns (m_lo, m_hi) 1-based inclusive; m_lo > m_hi means this rank owns no columns.
function _balanced_range(M::Int, rank::Int, nranks::Int)
    base = div(M, nranks)
    rem = mod(M, nranks)
    if rank < rem
        m_lo = rank * (base + 1) + 1
        m_hi = m_lo + base
    else
        m_lo = rem * (base + 1) + (rank - rem) * base + 1
        m_hi = m_lo + base - 1
    end
    return m_lo, m_hi
end

# A GraphGPProblem carrying only the global dense first layer (full coords + global n0). The
# dense gradient/logdet use coords[:,1:n0] / n0 / scale / bins / vals only — never neighbors —
# so an empty neighbor slice is fine.
function _dense_only_prob(prob::GraphGPProblem)
    return GraphGPProblem(prob.coords, prob.neighbors[:, 1:0], [prob.n0], prob.n0,
        prob.scale, prob.bins, prob.vals)
end

function distribute(prob::GraphGPProblem, comm::MPI.Comm; scheme::Symbol = :replicate_coords)
    scheme === :replicate_coords ||
        error("distribute: only scheme=:replicate_coords is implemented (Phase 1); " *
              "spatial coords partitioning is Phase 4.")
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    M = nrefined(prob)
    m_lo, m_hi = _balanced_range(M, rank, nranks)

    # This rank's neighbor column slice (same backend as prob.neighbors). Empty if m_lo > m_hi.
    nb_slice = prob.neighbors[:, m_lo:m_hi]
    n0g = prob.n0
    # Shift n0 so local column j (1-based) maps to global self index n0g + (m_lo-1) + j.
    local_n0 = n0g + m_lo - 1
    local_prob = GraphGPProblem(prob.coords, nb_slice, [local_n0], local_n0, prob.scale,
        prob.bins, prob.vals)

    is_root = rank == 0
    dense_prob = is_root ? _dense_only_prob(prob) : nothing
    return DistributedGraphGPProblem(local_prob, dense_prob, comm, n0g, m_lo, m_hi, is_root,
        prob.indices, copy(prob.offsets))
end

# --- reduction shims (host Float64 payloads) ---
_dist_allreduce_sum(x::Float64, comm::MPI.Comm) = MPI.Allreduce(x, MPI.SUM, comm)

function _dist_allreduce_sum!(v::Vector{Float64}, comm::MPI.Comm)
    MPI.Allreduce!(v, MPI.SUM, comm)
    return v
end

# Concatenate each rank's contiguous slice (in rank order) into the full vector.
function _dist_allgather_columns(xi_local::AbstractVector, comm::MPI.Comm)
    counts = MPI.Allgather(Cint(length(xi_local)), comm)
    out = Vector{eltype(xi_local)}(undef, sum(counts))
    MPI.Allgatherv!(collect(xi_local), MPI.VBuffer(out, counts), comm)
    return out
end

_dist_allreduce_min!(v::AbstractVector, comm::MPI.Comm) = (MPI.Allreduce!(v, MPI.MIN, comm); v)
_dist_allreduce_max!(v::AbstractVector, comm::MPI.Comm) = (MPI.Allreduce!(v, MPI.MAX, comm); v)

# === Phase 4: distributed graph construction ===

# Scheme A (fitting): replicated points, replicated tree, PARTITIONED query. Each rank queries
# only its slice of refined points, so the full neighbors is never materialised on one rank.
function distributed_build_graph(points::AbstractMatrix, comm::MPI.Comm, n0::Int, k::Int,
        bins::AbstractVector, vals::AbstractVector;
        backend = KernelAbstractions.get_backend(points), lattice_bits::Int = 21)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    N = size(points, 1)
    M = N - n0

    ptsT = permutedims(points)                                   # (D, N)
    spts, seg_lo, seg_hi, split_dim, perm = build_tree_ka(ptsT; backend = backend)

    m_lo, m_hi = _balanced_range(M, rank, nranks)
    rng = m_hi >= m_lo ? (m_lo:m_hi) : (1:0)
    nb_slice = query_preceding_neighbors_ka(spts, seg_lo, seg_hi, split_dim, n0, k;
        backend = backend, mrange = rng)                         # (k, M_local), global indices

    coords_nd, _, scale = quantize_to_lattice(permutedims(spts), lattice_bits)  # (N, D) UInt32
    coords = permutedims(coords_nd)                              # (D, N) replicated
    T = eltype(vals)
    bins_b = _move_to_backend(bins, backend)
    vals_b = _move_to_backend(vals, backend)

    local_n0 = n0 + m_lo - 1
    # offsets: tree order has no depth batches; a placeholder valid for fitting (logdet/inverse),
    # not for forward generate (use build_graph_ka for generation, or a distributed depth-sort).
    local_prob = GraphGPProblem(coords, nb_slice, [local_n0], local_n0, T(scale), bins_b, vals_b)
    is_root = rank == 0
    dense_prob = is_root ?
        GraphGPProblem(coords, nb_slice[:, 1:0], [n0], n0, T(scale), bins_b, vals_b) : nothing
    return DistributedGraphGPProblem(local_prob, dense_prob, comm, n0, m_lo, m_hi, is_root,
        Array(perm), [n0, N])
end

# Scheme B foundation: quantise spatially-partitioned points on a globally-consistent lattice.
function distributed_quantize(points_local::AbstractMatrix{<:Real}, comm::MPI.Comm; bits::Int = 21)
    # points_local is (N_local, D). Reduce the global bounding box, then quantise locally.
    D = size(points_local, 2)
    lo = Float64.(vec(minimum(points_local; dims = 1)))
    hi = Float64.(vec(maximum(points_local; dims = 1)))
    _dist_allreduce_min!(lo, comm)
    _dist_allreduce_max!(hi, comm)
    origin = lo
    extent = maximum(hi .- lo)
    lmax = (1 << bits) - 1
    scale = extent / lmax
    coords_local = Array{UInt32}(undef, size(points_local, 1), D)
    @inbounds for j in 1:D, i in 1:size(points_local, 1)
        q = round((Float64(points_local[i, j]) - origin[j]) / scale)
        coords_local[i, j] = UInt32(clamp(q, 0, lmax))
    end
    return coords_local, origin, scale
end

end # module
