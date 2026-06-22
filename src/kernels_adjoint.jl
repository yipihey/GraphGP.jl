# Hand-written reverse-mode kernel for the gradient of `refine_logdet` w.r.t. `cov_vals`.
#
# Each workitem recomputes its forward factorization (cheap, recompute-not-store), runs the
# analytic Cholesky pullback to get the cotangent of every covariance entry, then scatters
# those cotangents through the (linear) covariance lookup into the global `d_vals` with
# atomics. Fully parallel; avoids Enzyme's runtime-activity overhead.

using Atomix: Atomix

@kernel function refine_logdet_grad_kernel!(d_vals, @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(bins), @Const(vals), nbins, seed::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A = @private T (K + 1, K + 1)
    Abar = @private T (K + 1, K + 1)
    Lbar = @private T (K + 1, K + 1)
    jc = @private UInt32 (K + 1, D)

    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nbins)
    chol_lower!(A, Val(K + 1))                              # A now holds L
    chol_logdet_pullback!(Abar, Lbar, A, Val(K + 1), seed)  # Abar = d logdet / d A

    @inbounds begin
        # Diagonal entries all map to bin 1 (distance 0); coalesce into a single atomic to
        # avoid extreme contention on d_vals[1].
        diag_acc = zero(T)
        for a in 1:(K + 1)
            diag_acc += Abar[a, a]
        end
        Atomix.@atomic d_vals[1] += diag_acc
        for a in 1:(K + 1)
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
end
