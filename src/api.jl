# Host-side drivers. These launch the per-point kernels and perform the (cheap) host-side
# reductions. They are the public, differentiable entry points.

# Workgroup size for GPU kernel launches. 32 (one warp/block) measured best for our
# private-memory-heavy kernels on an A6000 — the large per-thread scratch (the (k+1)² matrix,
# and the query's DFS stack) limits occupancy, so the smallest block maximizes warps-in-flight.
# Benchmarked vs 64/128/256: refine_logdet +10–12%, the k-NN query kernel +50%. The
# atomic-scatter *point*-gradient kernels are the exception (they prefer a larger block — see
# `_WG_SCATTER` — because the scatter target is large and atomic latency hides better with more
# threads). On CPU the KA backend uses this as a task-chunk hint; the value is not critical there.
_wgsize(backend) = backend isa KernelAbstractions.CPU ? nothing : 32

# Workgroup size for the atomic-scatter point-gradient kernels (scatter into a large d_points).
_wgsize_scatter(backend) = backend isa KernelAbstractions.CPU ? nothing : 256

"""
    refine_logdet(prob; backend) -> T

Log-determinant contribution of the GraphGP refinement (Vecchia approximation):
`sum(log(std))` over all refined points, where `std` is the last Cholesky diagonal of
each per-point (k+1)x(k+1) covariance. Fully parallel; matches `refine_logdet` in
`graphgp/refine.py`.
"""
function refine_logdet(prob::GraphGPProblem{T}; backend = KernelAbstractions.get_backend(prob)) where {T}
    terms = refine_logdet_terms(prob; backend = backend)
    return sum(terms)
end

# Per-point log(std) terms (kept separate so the host reduction is trivial to differentiate).
function refine_logdet_terms(prob::GraphGPProblem{T}; backend = KernelAbstractions.get_backend(prob)) where {T}
    M = nrefined(prob)
    K = nneighbors(prob)
    D = ndims_space(prob)
    terms = KernelAbstractions.zeros(backend, T, M)
    kernel = refine_logdet_kernel!(backend)
    kernel(terms, prob.coords, prob.neighbors, prob.n0, prob.scale, prob.bins, prob.vals,
        nbins(prob), Val(K), Val(D); ndrange = M, workgroupsize = _wgsize(backend))
    KernelAbstractions.synchronize(backend)
    return terms
end

"""
    refine_inv!(xi_out, prob, values; backend) -> xi_out

Inverse of the refinement: recover the unit-normal parameters `xi` for the refined points
from generated `values`. `xi[m] = (values[n0+m] - mean_vec . values[neighbors]) / std`.
Fully parallel; matches the refined-point branch of `refine_inv` in `graphgp/refine.py`.
"""
function refine_inv!(xi_out, prob::GraphGPProblem{T}, values;
        backend = KernelAbstractions.get_backend(prob)) where {T}
    M = nrefined(prob)
    K = nneighbors(prob)
    D = ndims_space(prob)
    kernel = refine_inv_kernel!(backend)
    kernel(xi_out, prob.coords, prob.neighbors, values, prob.n0, prob.scale, prob.bins,
        prob.vals, nbins(prob), Val(K), Val(D); ndrange = M, workgroupsize = _wgsize(backend))
    KernelAbstractions.synchronize(backend)
    return xi_out
end

"""
    refine_inv(prob, values; backend) -> xi

Allocating form of [`refine_inv!`](@ref).
"""
function refine_inv(prob::GraphGPProblem{T}, values;
        backend = KernelAbstractions.get_backend(prob)) where {T}
    xi = KernelAbstractions.zeros(backend, T, nrefined(prob))
    return refine_inv!(xi, prob, values; backend = backend)
end

"""
    refine!(values, prob, xi; backend) -> values

Forward GraphGP generation. `values` must be length `N` with `values[1:n0]` preset to the
initial (dense first-layer) values; the refined entries `values[n0+1:N]` are filled in.
Matches the `fast_jit` path of `refine` in `graphgp/refine.py`: a parallel mean/std pass
followed by a sequential sweep over the depth batches defined by `prob.offsets`.
"""
function refine!(values, prob::GraphGPProblem{T}, xi;
        backend = KernelAbstractions.get_backend(prob)) where {T}
    M = nrefined(prob)
    K = nneighbors(prob)
    D = ndims_space(prob)
    mean_vec = KernelAbstractions.zeros(backend, T, K, M)
    std = KernelAbstractions.zeros(backend, T, M)
    wgs = _wgsize(backend)
    refine_meanvec_std_kernel!(backend)(mean_vec, std, prob.coords, prob.neighbors, prob.n0,
        prob.scale, prob.bins, prob.vals, nbins(prob), Val(K), Val(D);
        ndrange = M, workgroupsize = wgs)
    KernelAbstractions.synchronize(backend)

    apply = refine_apply_kernel!(backend)
    offs = prob.offsets                      # 0-based exclusive batch ends; offs[1] == n0
    for b in 2:length(offs)
        m_lo = offs[b - 1] - prob.n0 + 1     # first refined column (1-based) in this batch
        len = offs[b] - offs[b - 1]
        len <= 0 && continue
        apply(values, mean_vec, std, prob.neighbors, xi, prob.n0, m_lo, Val(K);
            ndrange = len, workgroupsize = wgs)
        KernelAbstractions.synchronize(backend)
    end
    return values
end

"""
    refine(prob, initial_values, xi; backend) -> values

Allocating form of [`refine!`](@ref). `initial_values` are the first `n0` values.
"""
function refine(prob::GraphGPProblem{T}, initial_values, xi;
        backend = KernelAbstractions.get_backend(prob)) where {T}
    N = npoints(prob)
    values = KernelAbstractions.zeros(backend, T, N)
    copyto!(view(values, 1:prob.n0), initial_values)
    return refine!(values, prob, xi; backend = backend)
end
