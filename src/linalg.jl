# Tiny dense linear-algebra primitives for the fused per-point GraphGP core.
#
# Each refined point conditions on its `k` nearest preceding neighbors, producing a
# (k+1)x(k+1) symmetric covariance matrix. `k` is small (~10) so all of this runs
# scalarized in a single workitem, keeping the matrix in private/register memory.
#
# Buffers (`A`, `x`, `jc`) are caller-provided and may be plain `Array`s (CPU / tests)
# or KernelAbstractions `@private` arrays (GPU). They are indexed 2D where noted.

# Assemble the lower triangle (incl. diagonal) of the (KP1 x KP1) covariance matrix `A`
# from the integer lattice coordinates `jc` (KP1 x D, UInt32-like) of the joint point set.
#
# Distances are exact integer differences accumulated in Int64, converted to `T` only at
# the sqrt, then scaled to physical units by `scale` before the covariance lookup.
@inline function assemble_cov!(A, jc, ::Val{KP1}, ::Val{D}, scale::T, bins, vals, nbins::Int) where {KP1, D, T}
    @inbounds for a in 1:KP1
        A[a, a] = cov_lookup(zero(T), bins, vals, nbins)   # r = 0 -> vals[1] (incl. jitter)
        for b in 1:(a - 1)
            sq = zero(Int64)
            for dd in 1:D
                di = Int64(jc[a, dd]) - Int64(jc[b, dd])
                sq += di * di
            end
            r = sqrt(T(sq)) * scale
            A[a, b] = cov_lookup(r, bins, vals, nbins)
        end
    end
    return nothing
end

# In-place lower Cholesky (Cholesky-Banachiewicz, left-looking) of the symmetric matrix
# whose lower triangle is stored in `A`. Overwrites the lower triangle with `L`, `A = L*L'`.
# On a non-positive pivot (loss of positive-definiteness, e.g. in f32) the factor is filled
# with NaN, mirroring the NaN-propagation of `jnp.linalg.cholesky`.
@inline function chol_lower!(A, ::Val{KP1}) where {KP1}
    @inbounds for j in 1:KP1
        s = A[j, j]
        for p in 1:(j - 1)
            s -= A[j, p] * A[j, p]
        end
        if s <= zero(s)
            nan = oftype(s, NaN)
            for i in j:KP1
                A[i, j] = nan
            end
        else
            d = sqrt(s)
            A[j, j] = d
            for i in (j + 1):KP1
                t = A[i, j]
                for p in 1:(j - 1)
                    t -= A[i, p] * A[j, p]
                end
                A[i, j] = t / d
            end
        end
    end
    return nothing
end

# Solve L[1:K,1:K]' x = L[K+1, 1:K] for `x` (the conditional-mean weights `mean_vec`),
# an upper-triangular back-substitution. `L` is the Cholesky factor from `chol_lower!`.
@inline function mean_vec_solve!(x, L, ::Val{K}) where {K}
    @inbounds for i in K:-1:1
        s = L[K + 1, i]
        for j in (i + 1):K
            s -= L[j, i] * x[j]
        end
        x[i] = s / L[i, i]
    end
    return nothing
end

# Reverse-mode pullback of `logdet = log(L[KP1,KP1])` w.r.t. the original (lower-triangle)
# covariance entries, where `L = chol_lower!(A)`. Given the factor `L` and a zeroed scratch
# `Lbar` (KP1 x KP1), fills the lower triangle of `Abar` with the cotangents
# `Abar[i,j] = d logdet / d A[i,j]`. `seed` is the incoming cotangent on `logdet`.
#
# This is the exact reverse of the left-looking recurrence in `chol_lower!`, derived by
# reverse-accumulating each scalar operation; validated numerically against Enzyme / JAX.
@inline function chol_logdet_pullback!(Abar, Lbar, L, ::Val{KP1}, seed::T) where {KP1, T}
    @inbounds begin
        for j in 1:KP1, i in 1:KP1
            Lbar[i, j] = zero(T)
        end
        Lbar[KP1, KP1] = seed / L[KP1, KP1]   # d(seed*log(L[n,n]))/dL[n,n]
        for j in KP1:-1:1
            ljj = L[j, j]
            for i in KP1:-1:(j + 1)
                lbij = Lbar[i, j]
                tbar = lbij / ljj
                Lbar[j, j] -= lbij * L[i, j] / ljj
                Abar[i, j] = tbar
                for p in 1:(j - 1)
                    Lbar[i, p] -= tbar * L[j, p]
                    Lbar[j, p] -= tbar * L[i, p]
                end
            end
            sbar = Lbar[j, j] / (2 * ljj)
            Abar[j, j] = sbar
            for p in 1:(j - 1)
                Lbar[j, p] -= 2 * sbar * L[j, p]
            end
        end
    end
    return nothing
end
