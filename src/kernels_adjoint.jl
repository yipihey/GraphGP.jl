# Hand-written reverse-mode kernels for the gradients of `refine_logdet` and
# `refine_inv_loss` w.r.t. `cov_vals`, plus a fused forward+backward for logdet.
#
# Each workitem recomputes its forward factorization (cheap, recompute-not-store), runs the
# analytic Cholesky pullback to get the cotangent of every covariance entry, then scatters
# those cotangents through the (linear) covariance lookup into the global `d_vals` with
# atomics. Fully parallel; avoids Enzyme's runtime-activity overhead.

using Atomix: Atomix

# Max bins for the shared-memory histogram scatter (see below). nbins beyond this falls back to
# the direct-atomic kernels. 2048 covers typical discretizations (300–1000 bins); the localmem
# cost is `_HIST_BINS * sizeof(T)` per block (8 KB f32 / 16 KB f64).
const _HIST_BINS = 2048

# Shared scatter helper — avoids code duplication across the three kernels below.
# Accumulates Abar cotangents (lower triangle) into d_vals using atomic adds.
# Diagonals (r=0) are coalesced into a single atomic to reduce contention on d_vals[1].
@inline function _scatter_Abar_atomic!(d_vals, Abar, jc, scale::T, bins, nbins,
        ::Val{KP1}, ::Val{D}) where {T, KP1, D}
    @inbounds begin
        diag_acc = zero(T)
        for a in 1:KP1
            diag_acc += Abar[a, a]
        end
        Atomix.@atomic d_vals[1] += diag_acc
        for a in 1:KP1
            for b in 1:(a - 1)
                sq = zero(Int64)
                for dd in 1:D
                    di = Int64(jc[a, dd]) - Int64(jc[b, dd])
                    sq += di * di
                end
                r = sqrt(T(sq)) * scale
                lo, wlo, whi = cov_lookup_weights(r, bins, nbins)
                g = Abar[a, b]
                Atomix.@atomic d_vals[lo] += g * wlo
                Atomix.@atomic d_vals[lo + 1] += g * whi
            end
        end
    end
    return nothing
end

# Scatter Abar cotangents into a block-local histogram `hist` (shared memory) instead of
# straight into the global `d_vals`. The off-diagonal scatter into the small `d_vals` is ~half
# the cov_vals-gradient cost (global atomic contention); accumulating per block in shared memory
# and flushing once cuts global atomics ~(block size)× and uses faster shared atomics
# (~+45% in f32). Same layout as `_scatter_Abar_atomic!`; the diagonal goes to `hist[1]`.
@inline function _scatter_Abar_hist!(hist, Abar, jc, scale::T, bins, nbins,
        ::Val{KP1}, ::Val{D}) where {T, KP1, D}
    @inbounds begin
        diag_acc = zero(T)
        for a in 1:KP1
            diag_acc += Abar[a, a]
        end
        Atomix.@atomic hist[1] += diag_acc
        for a in 1:KP1
            for b in 1:(a - 1)
                sq = zero(Int64)
                for dd in 1:D
                    di = Int64(jc[a, dd]) - Int64(jc[b, dd])
                    sq += di * di
                end
                r = sqrt(T(sq)) * scale
                lo, wlo, whi = cov_lookup_weights(r, bins, nbins)
                g = Abar[a, b]
                Atomix.@atomic hist[lo] += g * wlo
                Atomix.@atomic hist[lo + 1] += g * whi
            end
        end
    end
    return nothing
end

# Gradient of refine_logdet w.r.t. cov_vals via shared-memory histogram. One block of `W`
# threads shares a `hist` of length nbins; each thread scatters into it, then the block flushes
# to `d_vals`. `M` = number of refined points; `ndrange` must be a multiple of `W` (so every
# launched thread reaches the two barriers — the body is guarded by `m <= M`).
@kernel function refine_logdet_grad_hist_kernel!(d_vals, @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(bins), @Const(vals), nbins, M, seed::T, W,
        ::Val{K}, ::Val{D}) where {T, K, D}
    hist = @localmem T (_HIST_BINS,)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    jc   = @private UInt32 (K + 1, D)
    li = @index(Local, Linear)
    @inbounds begin
        i = li
        while i <= nbins
            hist[i] = zero(T)
            i += W
        end
    end
    @synchronize
    m = @index(Global)
    @inbounds if m <= M
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
        chol_lower!(A, Val(K + 1))
        chol_logdet_pullback!(Abar, Lbar, A, Val(K + 1), seed)
        _scatter_Abar_hist!(hist, Abar, jc, scale, bins, nbins, Val(K + 1), Val(D))
    end
    @synchronize
    @inbounds begin
        i = li
        while i <= nbins
            h = hist[i]
            h != zero(T) && (Atomix.@atomic d_vals[i] += h)
            i += W
        end
    end
end

# Fused forward+backward for logdet via shared-memory histogram (also emits logdet_terms[m]).
@kernel function refine_logdet_and_grad_hist_kernel!(logdet_terms, d_vals,
        @Const(coords), @Const(neighbors), n0, scale::T, @Const(bins), @Const(vals), nbins, M,
        seed::T, W, ::Val{K}, ::Val{D}) where {T, K, D}
    hist = @localmem T (_HIST_BINS,)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    jc   = @private UInt32 (K + 1, D)
    li = @index(Local, Linear)
    @inbounds begin
        i = li
        while i <= nbins
            hist[i] = zero(T)
            i += W
        end
    end
    @synchronize
    m = @index(Global)
    @inbounds if m <= M
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
        chol_lower!(A, Val(K + 1))
        logdet_terms[m] = log(A[K + 1, K + 1])
        chol_logdet_pullback!(Abar, Lbar, A, Val(K + 1), seed)
        _scatter_Abar_hist!(hist, Abar, jc, scale, bins, nbins, Val(K + 1), Val(D))
    end
    @synchronize
    @inbounds begin
        i = li
        while i <= nbins
            h = hist[i]
            h != zero(T) && (Atomix.@atomic d_vals[i] += h)
            i += W
        end
    end
end

# Gradient of 0.5·‖xi‖² (xi = refine_inv) w.r.t. cov_vals via shared-memory histogram.
@kernel function refine_inv_loss_grad_hist_kernel!(d_vals, xi_out,
        @Const(coords), @Const(neighbors), @Const(values), n0, scale::T, @Const(bins),
        @Const(vals), nbins, M, W, ::Val{K}, ::Val{D}) where {T, K, D}
    hist = @localmem T (_HIST_BINS,)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    mv   = @private T (K,)
    zbar = @private T (K,)
    jc   = @private UInt32 (K + 1, D)
    li = @index(Local, Linear)
    @inbounds begin
        i = li
        while i <= nbins
            hist[i] = zero(T)
            i += W
        end
    end
    @synchronize
    m = @index(Global)
    @inbounds if m <= M
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
        chol_lower!(A, Val(K + 1))
        mean_vec_solve!(mv, A, Val(K))
        std = A[K + 1, K + 1]
        mean_val = zero(T)
        for j in 1:K
            mean_val += mv[j] * values[neighbors[j, m]]
        end
        xi_m = (values[n0 + m] - mean_val) / std
        xi_out[m] = xi_m
        for j in 1:(K + 1)
            for i in 1:(K + 1)
                Lbar[i, j] = zero(T)
            end
        end
        Lbar[K + 1, K + 1] = -xi_m * xi_m / std
        for j in 1:K
            zbar[j] = -xi_m * values[neighbors[j, m]] / std
        end
        for i in 1:K
            s = zbar[i]
            for p in 1:(i - 1)
                s -= A[i, p] * zbar[p]
            end
            zbar[i] = s / A[i, i]
        end
        for j in 1:K
            Lbar[K + 1, j] += zbar[j]
            for i in 1:j
                Lbar[j, i] -= zbar[i] * mv[j]
            end
        end
        chol_pullback!(Abar, Lbar, A, Val(K + 1))
        _scatter_Abar_hist!(hist, Abar, jc, scale, bins, nbins, Val(K + 1), Val(D))
    end
    @synchronize
    @inbounds begin
        i = li
        while i <= nbins
            h = hist[i]
            h != zero(T) && (Atomix.@atomic d_vals[i] += h)
            i += W
        end
    end
end

# Shared scatter helper for the POINT gradients: accumulate Abar cotangents into d_points
# (D × N) through k'(r)·dr/dx with atomic adds. Diagonal entries (r=0) are position-independent
# and skipped. `ga`/`gb` are the global point indices for local rows a/b of the (k+1) block.
@inline function _scatter_Abar_points_atomic!(d_points, Abar, jc, neighbors, m, n0,
        scale::T, bins, vals, nbins, ::Val{KP1}, ::Val{D}) where {T, KP1, D}
    K = KP1 - 1
    @inbounds for a in 2:KP1
        ga = a <= K ? neighbors[a, m] : n0 + m
        for b in 1:(a - 1)
            gb = b <= K ? neighbors[b, m] : n0 + m
            sq = zero(Int64)
            for dd in 1:D
                di = Int64(jc[a, dd]) - Int64(jc[b, dd])
                sq += di * di
            end
            sq == 0 && continue
            r = sqrt(T(sq)) * scale
            dcov = cov_lookup_dr(r, bins, vals, nbins)
            dcov == zero(T) && continue
            factor = Abar[a, b] * dcov / r
            for dd in 1:D
                diff = scale * (T(Int64(jc[a, dd])) - T(Int64(jc[b, dd])))
                c = factor * diff
                Atomix.@atomic d_points[dd, ga] += c
                Atomix.@atomic d_points[dd, gb] -= c
            end
        end
    end
    return nothing
end

# Gradient of refine_logdet w.r.t. point positions (atomic scatter into d_points).
@kernel function refine_logdet_grad_points_kernel!(d_points, @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(bins), @Const(vals), nbins, seed::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    jc   = @private UInt32 (K + 1, D)

    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    chol_logdet_pullback!(Abar, Lbar, A, Val(K + 1), seed)
    _scatter_Abar_points_atomic!(d_points, Abar, jc, neighbors, m, n0, scale, bins, vals,
        nbins, Val(K + 1), Val(D))
end

# Gradient of 0.5·‖xi‖² (xi = refine_inv) w.r.t. point positions (atomic scatter into d_points).
@kernel function refine_inv_loss_grad_points_kernel!(d_points, @Const(coords), @Const(neighbors),
        @Const(values), n0, scale::T, @Const(bins), @Const(vals), nbins,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    mv   = @private T (K,)
    zbar = @private T (K,)
    jc   = @private UInt32 (K + 1, D)

    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    mean_vec_solve!(mv, A, Val(K))
    std = A[K + 1, K + 1]

    @inbounds begin
        mean_val = zero(T)
        for j in 1:K
            mean_val += mv[j] * values[neighbors[j, m]]
        end
        xi_m = (values[n0 + m] - mean_val) / std
    end

    @inbounds for j in 1:(K + 1)
        for i in 1:(K + 1)
            Lbar[i, j] = zero(T)
        end
    end
    @inbounds Lbar[K + 1, K + 1] = -xi_m * xi_m / std
    @inbounds for j in 1:K
        zbar[j] = -xi_m * values[neighbors[j, m]] / std
    end
    @inbounds for i in 1:K
        s = zbar[i]
        for p in 1:(i - 1)
            s -= A[i, p] * zbar[p]
        end
        zbar[i] = s / A[i, i]
    end
    @inbounds for j in 1:K
        Lbar[K + 1, j] += zbar[j]
        for i in 1:j
            Lbar[j, i] -= zbar[i] * mv[j]
        end
    end

    chol_pullback!(Abar, Lbar, A, Val(K + 1))
    _scatter_Abar_points_atomic!(d_points, Abar, jc, neighbors, m, n0, scale, bins, vals,
        nbins, Val(K + 1), Val(D))
end

# Gradient of refine_logdet w.r.t. cov_vals.
@kernel function refine_logdet_grad_kernel!(d_vals, @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(bins), @Const(vals), nbins, seed::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    jc   = @private UInt32 (K + 1, D)

    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    chol_logdet_pullback!(Abar, Lbar, A, Val(K + 1), seed)
    _scatter_Abar_atomic!(d_vals, Abar, jc, scale, bins, nbins, Val(K + 1), Val(D))
end

# Fused forward+backward for logdet: single kernel emits both log(std) and d_vals gradient.
# Saves one full Cholesky recomputation vs running refine_logdet_kernel! + refine_logdet_grad_kernel!.
@kernel function refine_logdet_and_grad_kernel!(logdet_terms, d_vals,
        @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(bins), @Const(vals), nbins, seed::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    jc   = @private UInt32 (K + 1, D)

    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    @inbounds logdet_terms[m] = log(A[K + 1, K + 1])
    chol_logdet_pullback!(Abar, Lbar, A, Val(K + 1), seed)
    _scatter_Abar_atomic!(d_vals, Abar, jc, scale, bins, nbins, Val(K + 1), Val(D))
end

# Gradient of 0.5·‖xi‖² (with xi = refine_inv) w.r.t. cov_vals.
# Also emits xi_out[m] so the caller can compute the loss = 0.5·sum(xi.^2) without a
# separate refine_inv launch. Replaces the Enzyme fallback for GPU.
@kernel function refine_inv_loss_grad_kernel!(d_vals, xi_out,
        @Const(coords), @Const(neighbors), @Const(values),
        n0, scale::T, @Const(bins), @Const(vals), nbins,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A    = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    mv   = @private T (K,)
    zbar = @private T (K,)
    jc   = @private UInt32 (K + 1, D)

    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))
    mean_vec_solve!(mv, A, Val(K))
    std = A[K + 1, K + 1]

    # Compute xi_m inline (matches refine_inv_kernel! formula).
    @inbounds begin
        mean_val = zero(T)
        for j in 1:K
            mean_val += mv[j] * values[neighbors[j, m]]
        end
        xi_m = (values[n0 + m] - mean_val) / std
    end
    @inbounds xi_out[m] = xi_m

    # Seed Lbar: d(0.5·xi^2)/d(std) = xi · d(xi)/d(std) = xi·(-xi/std) = -xi²/std
    @inbounds for j in 1:(K + 1)
        for i in 1:(K + 1)
            Lbar[i, j] = zero(T)
        end
    end
    @inbounds Lbar[K + 1, K + 1] = -xi_m * xi_m / std

    # Backward of mean_vec_solve: forward-substitute L[1:K,1:K] · zbar = mean_vec_bar,
    # where mean_vec_bar[j] = -xi_m · values[nb[j]] / std.
    @inbounds for j in 1:K
        zbar[j] = -xi_m * values[neighbors[j, m]] / std
    end
    @inbounds for i in 1:K
        s = zbar[i]
        for p in 1:(i - 1)
            s -= A[i, p] * zbar[p]
        end
        zbar[i] = s / A[i, i]
    end
    # Scatter zbar into Lbar: adjoint of L[1:K,1:K]^T · mv = L[K+1,1:K]
    @inbounds for j in 1:K
        Lbar[K + 1, j] += zbar[j]
        for i in 1:j
            Lbar[j, i] -= zbar[i] * mv[j]
        end
    end

    chol_pullback!(Abar, Lbar, A, Val(K + 1))
    _scatter_Abar_atomic!(d_vals, Abar, jc, scale, bins, nbins, Val(K + 1), Val(D))
end

# Transpose of refine_apply_kernel! for one depth batch (the reverse depth-batch sweep of
# generate_grad_xi, run on-device instead of host). Given the field cotangent `vb`, emit the xi
# cotangent xg[p] = std·vb[p] for this batch's refined points and scatter the conditional-mean
# contributions back to their neighbours (strictly-earlier points). Atomic because batch-mates can
# share a neighbour; batch-mates never neighbour each other (depth-batch invariant), so the reads
# vb[p] and the scatter targets vb[neighbour] are disjoint within a batch. Launch batches
# latest-first in one stream (stream ordering makes batch b+1's scatters visible to batch b).
@kernel function refine_reverse_apply_kernel!(xg, vb, @Const(mean_vec), @Const(std),
        @Const(neighbors), n0, m_lo, ::Val{K}) where {K}
    t = @index(Global)
    m = m_lo + t - 1
    @inbounds begin
        p = n0 + m
        vbp = vb[p]
        xg[p] = std[m] * vbp
        for j in 1:K
            nbj = neighbors[j, m]
            nbj > 0 || continue
            Atomix.@atomic vb[nbj] += mean_vec[j, m] * vbp
        end
    end
end
