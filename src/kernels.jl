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

# Forward generation, step 1 (fully parallel): per refined point compute and store the
# conditional-mean weights `mean_vec` (K x M) and `std` (M). These feed the sequential
# per-batch apply below. `M*K` storage replaces JAX's `(M, k+1, k+1)` materialization.
@kernel function refine_meanvec_std_kernel!(mean_vec, std, @Const(coords), @Const(neighbors),
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
        for j in 1:K
            mean_vec[j, m] = mv[j]
        end
        std[m] = A[K + 1, K + 1]
    end
end

# Forward generation, step 2 (per depth batch): `values[n0+m] = mean_vec.values[neighbors] +
# std*xi` for the refined columns `m_lo .. m_lo+ndrange-1`. Run one batch at a time with a
# host-side barrier between batches; neighbors of a batch lie strictly in earlier batches.
@kernel function refine_apply_kernel!(values, @Const(mean_vec), @Const(std), @Const(neighbors),
        @Const(xi), n0, m_lo, ::Val{K}) where {K}
    t = @index(Global)
    m = m_lo + t - 1
    @inbounds begin
        acc = zero(eltype(values))
        for j in 1:K
            acc += mean_vec[j, m] * values[neighbors[j, m]]
        end
        values[n0 + m] = acc + std[m] * xi[m]
    end
end

# FUSED forward apply (per depth batch): assemble the per-point (k+1) covariance, factorize, solve
# for the conditional-mean weights, and apply — all inline, so the full `mean_vec` (K×M) is never
# materialised (saves ~K·4 B/pt of device memory) and the separate full-M mean/std pass is dropped
# (one fused pass instead of two — like the CUDA extension's forward). Same answer as the
# meanvec_std + apply pair. Run one batch at a time (`m_lo .. m_lo+ndrange-1`).
@kernel function refine_apply_fused_kernel!(values, @Const(coords), @Const(neighbors), @Const(xi),
        n0, m_lo, scale::T, @Const(bins), @Const(vals), nbins, ::Val{K}, ::Val{D}) where {T, K, D}
    t = @index(Global)
    m = m_lo + t - 1
    A = @private T (K + 1, K + 1)
    jc = @private UInt32 (K + 1, D)
    mv = @private T (K,)
    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    mean_vec_solve!(mv, A, Val(K))
    @inbounds begin
        acc = A[K + 1, K + 1] * xi[m]            # std * xi
        for j in 1:K
            acc += mv[j] * values[neighbors[j, m]]
        end
        values[n0 + m] = acc
    end
end
