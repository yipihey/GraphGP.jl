# Distributed (multi-node / multi-GPU) GraphGP.
#
# The Vecchia log-likelihood and its gradient w.r.t. the covariance table are a SUM of
# independent per-point terms (`refine_logdet = Σ log(std_m)`; the gradient scatters into an
# `nbins` histogram `d_vals`). Distributing the fitting inner loop is therefore: partition the
# refined points across ranks, run the EXISTING local kernels on each rank's column slice, then
# one MPI `Allreduce`. No kernel is rewritten — this file only adds the partition + reduction
# layer.
#
# Partition scheme A (this file): every rank holds the full (replicated) `coords` and its own
# contiguous column slice of `neighbors` (the dominant O(N) array). Neighbour indices stay
# global and resolve in the replicated `coords`, so the per-point kernels run verbatim on a
# `local_prob` whose `n0` is shifted so local column j ↔ global self `n0 + m_lo - 1 + j`.
#
# Reductions are accumulated in Float64 before the Allreduce: the per-point terms are f32 but
# the SUM over a billion of them, combined across a runtime-chosen MPI reduction tree, must be
# f64 to stay reproducible across rank counts (the f32 sum is already marginal — see the JAX
# comparison note).
#
# The MPI-specific pieces (`distribute`, the `_dist_allreduce_*` shims) are provided by the
# package extension `ext/GraphGPMPIExt.jl`, loaded when `using MPI`. This file is MPI-free; a
# `DistributedGraphGPProblem` can only be constructed via `distribute`, so the methods below are
# never reachable without the extension.

"""
    DistributedGraphGPProblem

A rank-local view of a `GraphGPProblem` for distributed evaluation. Holds this rank's
`local_prob` (full replicated `coords`, the rank's `neighbors` column slice, shifted `n0`), the
global `n0`, the rank's global column range `[m_lo, m_hi]`, an MPI communicator, and — on the
root rank only — a `dense_prob` carrying the global first-layer (dense) block. Construct with
[`distribute`](@ref) (requires `using MPI`).
"""
struct DistributedGraphGPProblem{P <: GraphGPProblem, DP, C}
    local_prob::P          # GraphGPProblem: full coords, sliced neighbors, n0 shifted by m_lo-1
    dense_prob::DP         # GraphGPProblem for the global dense first layer (root), else nothing
    comm::C                # MPI.Comm (opaque to core)
    n0::Int                # GLOBAL n0 (dense first-layer size)
    m_lo::Int              # this rank's first global refined column (1-based)
    m_hi::Int              # this rank's last  global refined column
    is_root::Bool          # rank 0: owns the dense first-layer contribution
end

nrefined_local(d::DistributedGraphGPProblem) = d.m_hi - d.m_lo + 1
KernelAbstractions.get_backend(d::DistributedGraphGPProblem) = KernelAbstractions.get_backend(d.local_prob)

"""
    distribute(prob::GraphGPProblem, comm; scheme = :replicate_coords) -> DistributedGraphGPProblem

Partition `prob` across the ranks of MPI communicator `comm`. Requires `using MPI`. Each rank
keeps the full `coords` and its balanced contiguous slice of `neighbors`; on a GPU backend each
rank binds to its node-local GPU first. See `ext/GraphGPMPIExt.jl`.
"""
function distribute end

# --- internal reduction shims (implemented in the MPI extension) ---
# `_dist_allreduce_sum(x::Float64, comm) -> Float64`  : sum-allreduce of a scalar.
# `_dist_allreduce_sum!(v::Vector{Float64}, comm)`     : in-place sum-allreduce of a vector
#   (CUDA-aware when available; host-staged otherwise — the payload is a scalar + nbins vector).
function _dist_allreduce_sum end
function _dist_allreduce_sum! end

# --- distributed log-likelihood + gradient (these reuse the existing local drivers) ---

"""
    generate_logdet(dprob::DistributedGraphGPProblem; backend) -> Float64

Distributed `generate_logdet`: each rank sums `log(std)` over its refined-point slice in
Float64, the root adds the global dense first-layer logdet, then a sum-`Allreduce`.
"""
function generate_logdet(dprob::DistributedGraphGPProblem;
        backend = KernelAbstractions.get_backend(dprob))
    lp = dprob.local_prob
    terms = refine_logdet_terms(lp; backend = backend)
    local_sum = sum(Float64, Array(terms))                 # f64 partial over this rank's points
    if dprob.is_root
        n0 = dprob.n0
        local_sum += Float64(generate_dense_logdet(view(lp.coords, :, 1:n0), lp.scale,
            lp.bins, lp.vals, n0))
    end
    return _dist_allreduce_sum(local_sum, dprob.comm)
end

"""
    generate_logdet_grad_vals(dprob::DistributedGraphGPProblem; backend) -> Vector{Float64}

Distributed gradient of `generate_logdet` w.r.t. `vals`: each rank scatters its slice into a
local `nbins` histogram, the root adds the dense block, then a sum-`Allreduce`. Returns the
global gradient (length `nbins`) on every rank, ready for [`hyperparam_grad`](@ref).
"""
function generate_logdet_grad_vals(dprob::DistributedGraphGPProblem;
        backend = KernelAbstractions.get_backend(dprob))
    dv = _local_grad_vals_f64(dprob; backend = backend)
    _dist_allreduce_sum!(dv, dprob.comm)
    return dv
end

"""
    generate_logdet_and_grad_vals(dprob::DistributedGraphGPProblem; backend) -> (Float64, Vector{Float64})

Fused distributed logdet + `vals` gradient: a single sum-`Allreduce` of `[logdet; d_vals]`.
"""
function generate_logdet_and_grad_vals(dprob::DistributedGraphGPProblem;
        backend = KernelAbstractions.get_backend(dprob))
    lp = dprob.local_prob
    terms = refine_logdet_terms(lp; backend = backend)
    local_ld = sum(Float64, Array(terms))
    dv = _local_grad_vals_f64(dprob; backend = backend)
    if dprob.is_root
        n0 = dprob.n0
        local_ld += Float64(generate_dense_logdet(view(lp.coords, :, 1:n0), lp.scale,
            lp.bins, lp.vals, n0))
    end
    packed = Vector{Float64}(undef, length(dv) + 1)
    packed[1] = local_ld
    @inbounds packed[2:end] .= dv
    _dist_allreduce_sum!(packed, dprob.comm)
    return packed[1], packed[2:end]
end

# Local Float64 partial gradient histogram (refine slice + root dense block).
function _local_grad_vals_f64(dprob::DistributedGraphGPProblem; backend)
    dv = Float64.(Array(refine_logdet_grad_vals(dprob.local_prob; backend = backend)))
    if dprob.is_root
        dv .+= Float64.(_dense_logdet_grad_vals(dprob.dense_prob))
    end
    return dv
end
