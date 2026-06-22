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

# Accumulate one refined point's logdet-gradient contribution into `dv` (length nbins),
# using caller-provided scratch. Shared by the threaded CPU path and exercised in tests.
@inline function _accumulate_point_grad!(dv, A, Abar, Lbar, jc, prob::GraphGPProblem{T}, m,
        ::Val{K}, ::Val{D}, seed::T) where {T, K, D}
    nb = nbins(prob)
    _gather_joint!(jc, prob.coords, prob.neighbors, m, prob.n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), prob.scale, prob.bins, prob.vals, nb)
    chol_lower!(A, Val(K + 1))
    chol_logdet_pullback!(Abar, Lbar, A, Val(K + 1), seed)
    @inbounds for a in 1:(K + 1)
        dv[1] += Abar[a, a]
        for b in 1:(a - 1)
            sq = zero(Int64)
            for dd in 1:D
                di = Int64(jc[a, dd]) - Int64(jc[b, dd])
                sq += di * di
            end
            r = sqrt(T(sq)) * prob.scale
            lo, wlo, whi = cov_lookup_weights(r, prob.bins, nb)
            g = Abar[a, b]
            dv[lo] += g * wlo
            dv[lo + 1] += g * whi
        end
    end
    return nothing
end

# CPU gradient: partition points across tasks, each accumulating into a private histogram
# (no atomics), then reduce. Avoids the atomic contention that cripples the scatter on CPU.
function _refine_logdet_grad_vals_cpu(prob::GraphGPProblem{T}, ::Val{K}, ::Val{D}) where {T, K, D}
    M = nrefined(prob)
    nb = nbins(prob)
    nchunks = max(1, min(Threads.nthreads(), M))
    chunks = collect(Iterators.partition(1:M, cld(M, nchunks)))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            dv = zeros(T, nb)
            A = Matrix{T}(undef, K + 1, K + 1)
            Abar = Matrix{T}(undef, K + 1, K + 1)
            Lbar = Matrix{T}(undef, K + 1, K + 1)
            jc = Matrix{UInt32}(undef, K + 1, D)
            for m in chunk
                _accumulate_point_grad!(dv, A, Abar, Lbar, jc, prob, m, Val(K), Val(D), one(T))
            end
            dv
        end
    end
    dvals = zeros(T, nb)
    for tsk in tasks
        dvals .+= fetch(tsk)
    end
    return dvals
end

"""
    refine_logdet_grad_vals(prob; backend) -> dvals

Gradient of `refine_logdet(prob)` with respect to `prob.vals` via the hand-written
reverse-mode adjoint. On CPU a privatized threaded reduction is used; on GPU the atomic
scatter kernel. `dvals` has the same shape as `prob.vals`.
"""
function refine_logdet_grad_vals(prob::GraphGPProblem{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    K = nneighbors(prob)
    D = ndims_space(prob)
    if backend isa KernelAbstractions.CPU
        return _refine_logdet_grad_vals_cpu(prob, Val(K), Val(D))
    end
    dvals = KernelAbstractions.zeros(backend, T, length(prob.vals))
    kernel = refine_logdet_grad_kernel!(backend)
    kernel(dvals, prob.coords, prob.neighbors, prob.n0, prob.scale, prob.bins, prob.vals,
        nbins(prob), one(T), Val(K), Val(D); ndrange = nrefined(prob))
    KernelAbstractions.synchronize(backend)
    return dvals
end

"""
    refine_logdet_grad_vals_enzyme(prob; backend) -> dvals

Gradient of `refine_logdet(prob)` w.r.t. `prob.vals` via Enzyme-through-KernelAbstractions.
Kept as a cross-check / reference for the hand-written [`refine_logdet_grad_vals`](@ref).
"""
function refine_logdet_grad_vals_enzyme(prob::GraphGPProblem{T};
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
