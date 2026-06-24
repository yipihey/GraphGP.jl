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
    DistributedGraphGPProblem, GraphGPProblem, nrefined

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

end # module
