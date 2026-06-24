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
    indices::Union{Nothing, Vector{Int}}  # GLOBAL tree-order permutation (for reordering data)
    offsets::Vector{Int}   # GLOBAL depth-batch offsets (for distributed forward generation)
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

"""
    distributed_build_graph(points, comm, n0, k, bins, vals; backend, lattice_bits) -> DistributedGraphGPProblem

Build the graph distributed for FITTING (Phase 4, scheme A). Points are replicated; each rank
builds the k-d tree and queries only its slice of refined points (the build's bottleneck), so the
full `neighbors` is never materialised on any one rank. The result is in tree order — valid for
`generate_logdet` / inverse-loss fitting (the Vecchia logdet is invariant to the depth reorder),
but NOT for forward `generate` (which needs depth batches; build that with `build_graph_ka` or a
later distributed depth-sort). Requires `using MPI`.
"""
function distributed_build_graph end

"""
    distributed_quantize(points_local, comm; bits = 21) -> (coords_local, origin, scale)

Quantise spatially-partitioned points onto a globally-consistent integer lattice: the bounding
box is reduced across ranks (`Allreduce` min/max), then each rank quantises its local points.
Foundation for scheme B (partitioned coords, 10B+). Requires `using MPI`.
"""
function distributed_quantize end

# --- internal reduction shims (implemented in the MPI extension) ---
# `_dist_allreduce_sum(x::Float64, comm) -> Float64`  : sum-allreduce of a scalar.
# `_dist_allreduce_sum!(v::Vector{Float64}, comm)`     : in-place sum-allreduce of a vector
#   (CUDA-aware when available; host-staged otherwise — the payload is a scalar + nbins vector).
# `_dist_allgather_columns(local_xi, comm) -> Vector` : concatenate each rank's slice in rank
#   order into the full vector (for distributed refine_inv).
function _dist_allreduce_sum end
function _dist_allreduce_sum! end
function _dist_allgather_columns end
# `_dist_allreduce_min!(v)`, `_dist_allreduce_max!(v)` : in-place min/max-allreduce (for the
#   global bounding box in distributed_quantize).
function _dist_allreduce_min! end
function _dist_allreduce_max! end

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

# === Phase 2: distributed inverse (refine_inv) + inverse-quadratic loss gradient ===
#
# `refine_inv` and the inverse-quadratic loss are per-point independent, but each point reads
# `data[neighbors]` — an N-vector. Under scheme A, `data` is small (8 B/pt) and replicated: pass
# the FULL data vector (original order); we reorder to tree order once on every rank, then the
# existing local kernels run on this rank's slice. The loss + its `vals` gradient reduce exactly
# like the logdet (sum of independent terms); the recovered `xi` is assembled by an allgather of
# the per-rank column slices.

# Reorder a full (original-order) data vector to tree/depth order on every rank.
_data_to_tree_order(dprob::DistributedGraphGPProblem, data) =
    dprob.indices === nothing ? data : data[dprob.indices]

"""
    refine_inv(dprob::DistributedGraphGPProblem, data; backend) -> Vector

Distributed `refine_inv`: each rank recovers `xi` for its refined-point slice from the
(replicated, original-order) `data`, then an allgather assembles the full length-`M` `xi`
(global refined order).
"""
function refine_inv(dprob::DistributedGraphGPProblem, data::AbstractVector;
        backend = KernelAbstractions.get_backend(dprob))
    lp = dprob.local_prob
    data_ord = _data_to_tree_order(dprob, data)
    data_local = _move_to_backend(eltype(lp.vals).(data_ord), backend)
    xi_local = Array(refine_inv(lp, data_local; backend = backend))    # this rank's columns
    # Columns are assigned contiguously in rank order, so an Allgatherv in rank order
    # reconstructs the full xi in global refined-column order.
    return _dist_allgather_columns(xi_local, dprob.comm)
end

"""
    generate_inv_loss_grad_vals(dprob::DistributedGraphGPProblem, data; backend) -> (Float64, Vector{Float64})

Distributed gradient of `0.5‖generate_inv(data)‖²` w.r.t. `vals`: each rank computes the loss +
`vals`-gradient over its slice, the root adds the dense first layer, then a single sum-`Allreduce`
of `[loss; d_vals]`. Returns the global `(loss, d_vals)` on every rank.
"""
function generate_inv_loss_grad_vals(dprob::DistributedGraphGPProblem, data::AbstractVector;
        backend = KernelAbstractions.get_backend(dprob))
    lp = dprob.local_prob
    T = eltype(lp.vals)
    data_ord = _data_to_tree_order(dprob, data)
    data_local = _move_to_backend(T.(data_ord), backend)
    ref_loss, g_ref = refine_inv_loss_grad_vals(lp, data_local; backend = backend)
    local_loss = Float64(ref_loss)
    dv = Float64.(Array(g_ref))
    if dprob.is_root
        n0 = dprob.n0
        dd = T.(data_ord[1:n0])
        xi_dense = generate_dense_inv(view(lp.coords, :, 1:n0), lp.scale, lp.bins, lp.vals, dd)
        local_loss += sum(abs2, xi_dense) / 2
        dv .+= Float64.(_dense_inv_loss_grad_vals(dprob.dense_prob, dd))
    end
    packed = Vector{Float64}(undef, length(dv) + 1)
    packed[1] = local_loss
    @inbounds packed[2:end] .= dv
    _dist_allreduce_sum!(packed, dprob.comm)
    return packed[1], packed[2:end]
end

# === Phase 3b: distributed forward generation of a SINGLE field ===
#
# `generate` is a sequential sweep over depth batches: batch b reads `values[neighbors]` produced
# by earlier batches. Distributed form (scheme A, replicated values): the parallel mean/std pass
# runs on each rank's column slice; then for each GLOBAL depth batch every rank applies its
# columns within that batch and a sum-Allreduce of the batch's self-values syncs all ranks before
# the next batch. Correctness-complete; the per-batch apply here is host-side (the field is
# replicated, so this distributes COMPUTE, not memory — partitioned values for memory scaling is
# Phase 4 / scheme B). For many independent fields prefer replicate-the-graph realizations.

# This rank's conditional-mean weights + std for its column slice (reuses the existing kernels).
function _local_meanvec_std(lp::GraphGPProblem{T}; backend) where {T}
    K = nneighbors(lp); D = ndims_space(lp); M = nrefined(lp)
    mean_vec = KernelAbstractions.zeros(backend, T, K, M)
    std = KernelAbstractions.zeros(backend, T, M)
    if _is_cpu(backend)
        _native_refine_meanvec_std!(mean_vec, std, lp, Val(K), Val(D))
    else
        refine_meanvec_std_kernel!(backend)(mean_vec, std, lp.coords, lp.neighbors, lp.n0,
            lp.scale, lp.bins, lp.vals, nbins(lp), Val(K), Val(D);
            ndrange = M, workgroupsize = _wgsize(backend))
        KernelAbstractions.synchronize(backend)
    end
    return mean_vec, std
end

"""
    generate(dprob::DistributedGraphGPProblem, xi; backend) -> Vector

Distributed forward generation of one field from full (replicated, tree-order) unit-normal
parameters `xi` (length N). Returns the field in the original point order on every rank.
"""
function generate(dprob::DistributedGraphGPProblem, xi::AbstractVector;
        backend = KernelAbstractions.get_backend(dprob))
    lp = dprob.local_prob
    T = eltype(lp.vals)
    n0g = dprob.n0
    offs = dprob.offsets
    N = npoints(lp)

    # Dense first layer (deterministic, replicated) + this rank's mean/std slice.
    xi_b = _move_to_backend(T.(xi), backend)
    v_dense = Array(generate_dense(view(lp.coords, :, 1:n0g), lp.scale, lp.bins, lp.vals,
        xi_b[1:n0g]))
    mv_d, std_d = _local_meanvec_std(lp; backend)
    mean_vec = Array(mv_d)                                 # K × M_local
    std = Array(std_d)                                     # M_local
    nbrs = Array(lp.neighbors)                             # K × M_local, global indices
    xi_h = T.(xi)                                          # host, tree order
    K = size(nbrs, 1)

    values = zeros(T, N)                                   # replicated host field (tree order)
    @inbounds values[1:n0g] .= v_dense

    # Sequential global depth-batch sweep with a per-batch sum-Allreduce of new self-values.
    for b in 2:length(offs)
        gc_lo = offs[b - 1] - n0g + 1     # first global refined column (1-based) in this batch
        gc_hi = offs[b] - n0g
        gc_hi < gc_lo && continue
        blen = gc_hi - gc_lo + 1
        contrib = zeros(Float64, blen)    # this rank's contributions (0 where another rank owns)
        lo = max(gc_lo, dprob.m_lo)
        hi = min(gc_hi, dprob.m_hi)
        @inbounds for c in lo:hi           # global columns this rank owns within the batch
            lc = c - dprob.m_lo + 1        # local column index
            acc = zero(T)
            for j in 1:K
                acc += mean_vec[j, lc] * values[nbrs[j, lc]]
            end
            contrib[c - gc_lo + 1] = Float64(acc + std[lc] * xi_h[n0g + c])
        end
        _dist_allreduce_sum!(contrib, dprob.comm)
        @inbounds values[(n0g + gc_lo):(n0g + gc_hi)] .= T.(contrib)
    end

    # Reorder to original point order if the graph was permuted.
    if dprob.indices !== nothing
        out = similar(values)
        @inbounds out[dprob.indices] = values
        return out
    end
    return values
end
