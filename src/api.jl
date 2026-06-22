# Host-side drivers. These launch the per-point kernels and perform the (cheap) host-side
# reductions. They are the public, differentiable entry points.

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
        nbins(prob), Val(K), Val(D); ndrange = M)
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
        prob.vals, nbins(prob), Val(K), Val(D); ndrange = M)
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
