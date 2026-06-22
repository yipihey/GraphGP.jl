# KernelAbstractions kernels: one workitem per refined point. Each workitem gathers its
# (k neighbor + 1 fine) lattice coordinates into private memory, assembles the (k+1)x(k+1)
# covariance on the fly, factorizes it, and emits its scalar contribution. The full
# (M, k+1, k+1) tensor is never materialized (the memory win over the JAX path).
#
# `K` and `D` are passed as `Val` so the inner loops fully unroll and the private scratch
# is statically sized (required for GPU local memory).

@inline function _gather_joint!(jc, coords, neighbors, m, n0, ::Val{K}, ::Val{D}) where {K, D}
    @inbounds begin
        for j in 1:K
            nj = neighbors[j, m]
            for dd in 1:D
                jc[j, dd] = coords[dd, nj]
            end
        end
        fine = n0 + m
        for dd in 1:D
            jc[K + 1, dd] = coords[dd, fine]
        end
    end
    return nothing
end

@kernel function refine_logdet_kernel!(logdet_terms, @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(bins), @Const(vals), nbins,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A = @private T (K + 1, K + 1)
    jc = @private UInt32 (K + 1, D)
    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    @inbounds logdet_terms[m] = log(A[K + 1, K + 1])
end

@kernel function refine_inv_kernel!(xi_out, @Const(coords), @Const(neighbors), @Const(values),
        n0, scale::T, @Const(bins), @Const(vals), nbins,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A = @private T (K + 1, K + 1)
    jc = @private UInt32 (K + 1, D)
    mv = @private T (K,)
    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    mean_vec_solve!(mv, A, Val(K))
    @inbounds begin
        mean = zero(T)
        for j in 1:K
            mean += mv[j] * values[neighbors[j, m]]
        end
        std = A[K + 1, K + 1]
        xi_out[m] = (values[n0 + m] - mean) / std
    end
end
