# Reverse-mode gradients w.r.t. the discretized covariance values `cov_vals` (the kernel
# hyperparameters) — the training-relevant derivatives. Point coordinates are fixed integer
# data, so `cov_vals` is the only active input.
#
# Differentiation goes straight through the KernelAbstractions kernels via the
# KernelAbstractions↔Enzyme integration. `set_runtime_activity` is required because each
# workitem keeps an integer coordinate scratch (`jc`) alongside the active covariance
# scratch (`A`), which defeats Enzyme's static activity analysis.

using Enzyme: Enzyme, Const, Duplicated, Reverse, set_runtime_activity

function _refine_logdet_launch!(terms, coords, neighbors, n0, scale, bins, vals, nbins,
        ::Val{K}, ::Val{D}, backend) where {K, D}
    kernel = refine_logdet_kernel!(backend)
    kernel(terms, coords, neighbors, n0, scale, bins, vals, nbins, Val(K), Val(D);
        ndrange = length(terms))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _refine_inv_launch!(xi_out, coords, neighbors, values, n0, scale, bins, vals, nbins,
        ::Val{K}, ::Val{D}, backend) where {K, D}
    kernel = refine_inv_kernel!(backend)
    kernel(xi_out, coords, neighbors, values, n0, scale, bins, vals, nbins, Val(K), Val(D);
        ndrange = length(xi_out))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    refine_logdet_grad_vals(prob; backend) -> dvals

Gradient of `refine_logdet(prob)` with respect to `prob.vals`. `dvals` has the same shape as
`prob.vals`.
"""
function refine_logdet_grad_vals(prob::GraphGPProblem{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    M = nrefined(prob)
    K = nneighbors(prob)
    D = ndims_space(prob)
    terms = KernelAbstractions.zeros(backend, T, M)
    dterms = KernelAbstractions.ones(backend, T, M)   # cotangent: d(sum terms)/d term = 1
    dvals = zero(prob.vals)
    Enzyme.autodiff(set_runtime_activity(Reverse), _refine_logdet_launch!, Const,
        Duplicated(terms, dterms),
        Const(prob.coords), Const(prob.neighbors), Const(prob.n0), Const(prob.scale),
        Const(prob.bins), Duplicated(prob.vals, dvals), Const(nbins(prob)),
        Const(Val(K)), Const(Val(D)), Const(backend))
    return dvals
end

"""
    refine_inv_loss_grad_vals(prob, values; backend) -> (loss, dvals)

Returns the inverse-half of the GP marginal likelihood, `loss = 0.5 * sum(xi.^2)` with
`xi = refine_inv(prob, values)`, and its gradient with respect to `prob.vals`.
"""
function refine_inv_loss_grad_vals(prob::GraphGPProblem{T}, values;
        backend = KernelAbstractions.get_backend(prob)) where {T}
    M = nrefined(prob)
    K = nneighbors(prob)
    D = ndims_space(prob)
    # Primal pass for the loss and the cotangent seed (d(0.5||xi||^2)/dxi = xi).
    xi = refine_inv(prob, values; backend = backend)
    loss = sum(abs2, xi) / 2
    dxi = copy(xi)
    xi_scratch = KernelAbstractions.zeros(backend, T, M)
    dvals = zero(prob.vals)
    Enzyme.autodiff(set_runtime_activity(Reverse), _refine_inv_launch!, Const,
        Duplicated(xi_scratch, dxi),
        Const(prob.coords), Const(prob.neighbors), Const(values), Const(prob.n0),
        Const(prob.scale), Const(prob.bins), Duplicated(prob.vals, dvals), Const(nbins(prob)),
        Const(Val(K)), Const(Val(D)), Const(backend))
    return loss, dvals
end
