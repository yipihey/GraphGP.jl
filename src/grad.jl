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
    @inbounds logdet_term = log(A[K + 1, K + 1])   # capture before pullback consumes Lbar
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
    return logdet_term
end

# CPU gradient: partition points across tasks, each accumulating into a private histogram
# (no atomics), then reduce. Returns (logdet_sum, dvals) — both computed in one pass.
function _refine_logdet_grad_vals_cpu(prob::GraphGPProblem{T}, ::Val{K}, ::Val{D}) where {T, K, D}
    M = nrefined(prob)
    nb = nbins(prob)
    nchunks = max(1, min(Threads.nthreads(), M))
    chunks = collect(Iterators.partition(1:M, cld(M, nchunks)))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            dv = zeros(T, nb)
            ld = zero(T)
            A = Matrix{T}(undef, K + 1, K + 1)
            Abar = Matrix{T}(undef, K + 1, K + 1)
            Lbar = Matrix{T}(undef, K + 1, K + 1)
            jc = Matrix{UInt32}(undef, K + 1, D)
            for m in chunk
                ld += _accumulate_point_grad!(dv, A, Abar, Lbar, jc, prob, m, Val(K), Val(D), one(T))
            end
            (ld, dv)
        end
    end
    logdet_sum = zero(T)
    dvals = zeros(T, nb)
    for tsk in tasks
        ld, dv = fetch(tsk)
        logdet_sum += ld
        dvals .+= dv
    end
    return logdet_sum, dvals
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
        _, dvals = _refine_logdet_grad_vals_cpu(prob, Val(K), Val(D))
        return dvals
    end
    dvals = KernelAbstractions.zeros(backend, T, length(prob.vals))
    kernel = refine_logdet_grad_kernel!(backend)
    kernel(dvals, prob.coords, prob.neighbors, prob.n0, prob.scale, prob.bins, prob.vals,
        nbins(prob), one(T), Val(K), Val(D);
        ndrange = nrefined(prob), workgroupsize = _wgsize(backend))
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
    generate_logdet_grad_vals(prob; backend) -> dvals

Gradient of `generate_logdet(prob)` (dense + refinement) w.r.t. `prob.vals`.
Combines the analytic hand-written refinement adjoint with the dense-layer Cholesky gradient.
"""
function generate_logdet_grad_vals(prob::GraphGPProblem{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    return refine_logdet_grad_vals(prob; backend = backend) .+ _dense_logdet_grad_vals(prob)
end

"""
    generate_inv_loss_grad_vals(prob, data; backend) -> (loss, dvals)

Returns `loss = 0.5 * ||generate_inv(prob, data)||^2` and its gradient w.r.t. `prob.vals`.
`data` should be in the original (not tree/depth) ordering; reordering is handled internally.
"""
function generate_inv_loss_grad_vals(prob::GraphGPProblem{T}, data::AbstractVector{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    data_ord = prob.indices !== nothing ? data[prob.indices] : data
    # Dense part
    xi_dense = generate_dense_inv(view(prob.coords, :, 1:n0), prob.scale, prob.bins, prob.vals,
        data_ord[1:n0])
    dense_loss = sum(abs2, xi_dense) / 2
    g_dense = _dense_inv_loss_grad_vals(prob, data_ord[1:n0])
    # Refinement part
    ref_loss, g_ref = refine_inv_loss_grad_vals(prob, data_ord; backend = backend)
    return dense_loss + ref_loss, g_dense .+ g_ref
end

# Per-point backward of refine_inv w.r.t. cov_vals.
# Re-runs the forward (assemble_cov → chol → mean_vec_solve) to recover A and mean_vec,
# then backpropagates d(0.5·xi[m]^2) using the pre-computed xi_m as the cotangent seed.
# The backward of mean_vec_solve uses a forward substitution with L[1:K,1:K], and the
# Cholesky pullback reuses the generic chol_pullback! (same recurrence as the logdet path).
@inline function _accumulate_inv_point_grad!(dv, A, Abar, Lbar, zbar, mv, jc,
        prob::GraphGPProblem{T}, m, values, ::Val{K}, ::Val{D}) where {T, K, D}
    nb = nbins(prob)
    n0 = prob.n0
    _gather_joint!(jc, prob.coords, prob.neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), prob.scale, prob.bins, prob.vals, nb)
    chol_lower!(A, Val(K + 1))
    mean_vec_solve!(mv, A, Val(K))
    std = A[K + 1, K + 1]

    # Compute xi_m inline, matching the refine_inv_kernel! formula.
    @inbounds begin
        mean_val = zero(T)
        for j in 1:K
            mean_val += mv[j] * values[prob.neighbors[j, m]]
        end
        xi_m = (values[n0 + m] - mean_val) / std
    end

    # Seed Lbar: d(0.5·xi^2)/d(std)  = xi · d(xi)/d(std) = xi·(-xi/std) = -xi²/std
    @inbounds for j in 1:K + 1, i in 1:K + 1
        Lbar[i, j] = zero(T)
    end
    @inbounds Lbar[K + 1, K + 1] = -xi_m * xi_m / std

    # Seed zbar ← mean_vec_bar: d(0.5·xi^2)/d(mv[j]) = xi·(-values[nb[j]]/std)
    # Forward-substitute L[1:K,1:K] · zbar = mean_vec_bar to get the Lbar contribution.
    @inbounds for j in 1:K
        zbar[j] = -xi_m * values[prob.neighbors[j, m]] / std
    end
    @inbounds for i in 1:K
        s = zbar[i]
        for p in 1:(i - 1)
            s -= A[i, p] * zbar[p]
        end
        zbar[i] = s / A[i, i]
    end
    # Scatter zbar → Lbar: from the adjoint of  L[1:K,1:K]^T · mv = L[K+1,1:K]
    @inbounds for j in 1:K
        Lbar[K + 1, j] += zbar[j]          # ← from bbar[j] = zbar[j]
        for i in 1:j
            Lbar[j, i] -= zbar[i] * mv[j]  # ← from U_bar scatter
        end
    end

    # Backprop through chol_lower! using the generic pullback.
    chol_pullback!(Abar, Lbar, A, Val(K + 1))

    # Scatter Abar into dv via cov_lookup_weights (same layout as logdet path).
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
    return xi_m
end

# CPU gradient: privatized per-thread reduction (no atomics), then sum.
# Computes xi_m inline for each point (no separate refine_inv pass needed).
function _refine_inv_loss_grad_vals_cpu(prob::GraphGPProblem{T}, values,
        ::Val{K}, ::Val{D}) where {T, K, D}
    M = nrefined(prob)
    nb = nbins(prob)
    nchunks = max(1, min(Threads.nthreads(), M))
    chunks = collect(Iterators.partition(1:M, cld(M, nchunks)))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            dv = zeros(T, nb)
            loss_chunk = zero(T)
            A = Matrix{T}(undef, K + 1, K + 1)
            Abar = Matrix{T}(undef, K + 1, K + 1)
            Lbar = Matrix{T}(undef, K + 1, K + 1)
            zbar = Vector{T}(undef, K)
            mv = Vector{T}(undef, K)
            jc = Matrix{UInt32}(undef, K + 1, D)
            for m in chunk
                xi_m = _accumulate_inv_point_grad!(dv, A, Abar, Lbar, zbar, mv, jc,
                    prob, m, values, Val(K), Val(D))
                loss_chunk += xi_m * xi_m / 2
            end
            (loss_chunk, dv)
        end
    end
    loss_sum = zero(T)
    dvals = zeros(T, nb)
    for tsk in tasks
        lc, dv = fetch(tsk)
        loss_sum += lc
        dvals .+= dv
    end
    return loss_sum, dvals
end

"""
    refine_inv_loss_grad_vals(prob, values; backend) -> (loss, dvals)

Returns the inverse-half of the GP marginal likelihood, `loss = 0.5 * sum(xi.^2)` with
`xi = refine_inv(prob, values)`, and its gradient with respect to `prob.vals`.
On CPU uses the hand-written privatized-thread adjoint; on GPU falls back to Enzyme.
"""
function refine_inv_loss_grad_vals(prob::GraphGPProblem{T}, values;
        backend = KernelAbstractions.get_backend(prob)) where {T}
    K = nneighbors(prob)
    D = ndims_space(prob)
    if backend isa KernelAbstractions.CPU
        loss, dvals = _refine_inv_loss_grad_vals_cpu(prob, values, Val(K), Val(D))
        return loss, dvals
    end
    # GPU path: hand-written kernel computes xi inline and scatters gradient atomically.
    M = nrefined(prob)
    xi_out = KernelAbstractions.zeros(backend, T, M)
    dvals = KernelAbstractions.zeros(backend, T, length(prob.vals))
    refine_inv_loss_grad_kernel!(backend)(
        dvals, xi_out, prob.coords, prob.neighbors, values, prob.n0, prob.scale,
        prob.bins, prob.vals, nbins(prob), Val(K), Val(D);
        ndrange = M, workgroupsize = _wgsize(backend))
    KernelAbstractions.synchronize(backend)
    loss = sum(abs2, xi_out) / 2
    return loss, dvals
end

"""
    generate_logdet_and_grad_vals(prob; backend) -> (logdet, dvals)

Fused forward+backward pass for the log-determinant part of the GP marginal likelihood.
Returns `generate_logdet(prob)` and its gradient w.r.t. `prob.vals` in a single
traversal of the M refined points (no separate forward kernel launch on CPU).
"""
function generate_logdet_and_grad_vals(prob::GraphGPProblem{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    K = nneighbors(prob)
    D = ndims_space(prob)
    n0 = prob.n0
    if backend isa KernelAbstractions.CPU
        ref_ld, ref_dvals = _refine_logdet_grad_vals_cpu(prob, Val(K), Val(D))
        dense_ld = generate_dense_logdet(view(prob.coords, :, 1:n0), prob.scale, prob.bins,
            prob.vals, n0)
        dense_dvals = _dense_logdet_grad_vals(prob)
        return T(dense_ld) + ref_ld, dense_dvals .+ ref_dvals
    end
    # GPU: single fused kernel emits logdet_terms and scatters d_vals in one Cholesky pass.
    M = nrefined(prob)
    logdet_terms = KernelAbstractions.zeros(backend, T, M)
    dvals = KernelAbstractions.zeros(backend, T, length(prob.vals))
    refine_logdet_and_grad_kernel!(backend)(
        logdet_terms, dvals, prob.coords, prob.neighbors, prob.n0, prob.scale,
        prob.bins, prob.vals, nbins(prob), one(T), Val(K), Val(D);
        ndrange = M, workgroupsize = _wgsize(backend))
    KernelAbstractions.synchronize(backend)
    ref_ld = sum(logdet_terms)
    dense_ld = generate_dense_logdet(view(prob.coords, :, 1:n0), prob.scale, prob.bins,
        prob.vals, n0)
    dense_dvals = _dense_logdet_grad_vals(prob)
    return T(dense_ld) + ref_ld, dense_dvals .+ dvals
end

