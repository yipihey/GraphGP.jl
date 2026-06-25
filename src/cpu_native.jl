# Native multithreaded CPU implementations of the per-point forward kernels.
#
# The KernelAbstractions CPU backend carries heavy per-workgroup launch + @private overhead,
# and its scheduler *degrades* past ~64 threads. A plain `@threads` loop over the SAME inner
# kernels (`_gather_joint!` / `assemble_cov!` / `chol_lower!` / `mean_vec_solve!`) with
# per-thread scratch is 5-10x faster and actually scales (measured 2.4 -> 13.3 M pts/s at 64
# threads on an EPYC 7763). The GPU path keeps the KA kernels; the cov_vals gradient already
# runs natively via `Threads.@spawn` in grad.jl. Results are bit-for-bit the KA results (same
# inner functions), so dispatch is purely a performance choice — see the `_is_cpu` branches in
# api.jl. Each thread owns its scratch (the (k+1)² matrix `A`, the joint coords `jc`, and the
# mean-vec `mv`), so there is no per-point allocation and no false sharing.

@inline _is_cpu(backend) = backend isa KernelAbstractions.CPU

# Run `1:M` split into one contiguous chunk per thread. `body!(m, A, jc, mv)` is called for each
# point with thread-private scratch. `A`/`jc`/`mv` are sized for (K+1, D, K).
@inline function _threaded_points!(body!::F, M::Int, ::Val{K}, ::Val{D}, ::Type{T}) where {F, K, D, T}
    nt = Threads.nthreads()
    chunk = cld(M, nt)
    Threads.@threads :static for t in 1:nt
        A = Matrix{T}(undef, K + 1, K + 1)
        jc = Matrix{UInt32}(undef, K + 1, D)
        mv = Vector{T}(undef, K)
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, M)
        @inbounds for m in lo:hi
            body!(m, A, jc, mv)
        end
    end
    return nothing
end

function _native_refine_logdet_terms!(terms, prob::GraphGPProblem{T}, ::Val{K}, ::Val{D}) where {T, K, D}
    coords = prob.coords; neighbors = prob.neighbors; n0 = prob.n0
    scale = prob.scale; bins = prob.bins; vals = prob.vals; nb = nbins(prob)
    M = nrefined(prob)
    _threaded_points!(M, Val(K), Val(D), T) do m, A, jc, _mv
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nb)
        chol_lower!(A, Val(K + 1))
        terms[m] = log(A[K + 1, K + 1])
    end
    return terms
end

function _native_refine_inv!(xi_out, prob::GraphGPProblem{T}, values, ::Val{K}, ::Val{D}) where {T, K, D}
    coords = prob.coords; neighbors = prob.neighbors; n0 = prob.n0
    scale = prob.scale; bins = prob.bins; vals = prob.vals; nb = nbins(prob)
    M = nrefined(prob)
    _threaded_points!(M, Val(K), Val(D), T) do m, A, jc, mv
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nb)
        chol_lower!(A, Val(K + 1))
        mean_vec_solve!(mv, A, Val(K))
        mean = zero(T)
        @inbounds for j in 1:K
            mean += mv[j] * values[neighbors[j, m]]
        end
        std = @inbounds A[K + 1, K + 1]
        @inbounds xi_out[m] = (values[n0 + m] - mean) / std
    end
    return xi_out
end

function _native_refine_meanvec_std!(mean_vec, std, prob::GraphGPProblem{T}, ::Val{K}, ::Val{D}) where {T, K, D}
    coords = prob.coords; neighbors = prob.neighbors; n0 = prob.n0
    scale = prob.scale; bins = prob.bins; vals = prob.vals; nb = nbins(prob)
    M = nrefined(prob)
    _threaded_points!(M, Val(K), Val(D), T) do m, A, jc, mv
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nb)
        chol_lower!(A, Val(K + 1))
        mean_vec_solve!(mv, A, Val(K))
        @inbounds begin
            for j in 1:K
                mean_vec[j, m] = mv[j]
            end
            std[m] = A[K + 1, K + 1]
        end
    end
    return nothing
end

# FUSED forward apply for one depth batch (no materialised mean_vec): assemble + factorize + solve
# + apply per point, threaded over the batch. Mirrors refine_apply_fused_kernel! on the GPU.
function _native_refine_apply_fused!(values, prob::GraphGPProblem{T}, xi, m_lo::Int, len::Int,
        ::Val{K}, ::Val{D}) where {T, K, D}
    coords = prob.coords; neighbors = prob.neighbors; n0 = prob.n0
    scale = prob.scale; bins = prob.bins; vals = prob.vals; nb = nbins(prob)
    nt = Threads.nthreads()
    chunk = cld(len, nt)
    Threads.@threads :static for t in 1:nt
        A = Matrix{T}(undef, K + 1, K + 1)
        jc = Matrix{UInt32}(undef, K + 1, D)
        mv = Vector{T}(undef, K)
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, len)
        @inbounds for tt in lo:hi
            m = m_lo + tt - 1
            _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
            assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nb)
            chol_lower!(A, Val(K + 1))
            mean_vec_solve!(mv, A, Val(K))
            acc = A[K + 1, K + 1] * xi[m]
            for j in 1:K
                acc += mv[j] * values[neighbors[j, m]]
            end
            values[n0 + m] = acc
        end
    end
    return nothing
end

# Step 2 of forward generation for one depth batch: fully parallel over the batch's points.
function _native_refine_apply!(values, mean_vec, std, neighbors, xi, n0::Int, m_lo::Int,
        len::Int, ::Val{K}) where {K}
    nt = Threads.nthreads()
    chunk = cld(len, nt)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, len)
        @inbounds for tt in lo:hi
            m = m_lo + tt - 1
            acc = zero(eltype(values))
            for j in 1:K
                acc += mean_vec[j, m] * values[neighbors[j, m]]
            end
            values[n0 + m] = acc + std[m] * xi[m]
        end
    end
    return nothing
end
