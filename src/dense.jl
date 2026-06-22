# Dense first-layer operations for the initial n0 points.
# The first n0 points form a fully-connected GP (no sparse approximation); their
# covariance matrix is assembled and Cholesky-factored via LAPACK. n0 is typically
# small (100–1000), so BLAS/LAPACK overhead is negligible.
#
# All three functions share the same n0×n0 Cholesky: generate, invert, logdet.

"""
    _assemble_dense_cov(coords, scale, bins, vals, n) -> Matrix{T}

Build the n×n covariance matrix for the first n columns of `coords` (shape D×N)
using the same integer-distance + cov_lookup pipeline as the KA kernels.
`T` is inferred from `scale`.
"""
function _assemble_dense_cov(coords::AbstractMatrix{UInt32}, scale::T,
        bins::AbstractVector, vals::AbstractVector, n::Int) where {T}
    nb = length(bins)
    D = size(coords, 1)
    K = Matrix{T}(undef, n, n)
    for j in 1:n
        K[j, j] = cov_lookup(zero(T), bins, vals, nb)
        for i in j+1:n
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords[d, i]) - Int64(coords[d, j])
                sq += di * di
            end
            r = sqrt(T(sq)) * scale
            v = cov_lookup(r, bins, vals, nb)
            K[i, j] = v
            K[j, i] = v
        end
    end
    return K
end

"""
    generate_dense(coords, scale, bins, vals, xi) -> Vector{T}

Sample the first n0 = length(xi) points from the dense GP: returns `L * xi` where
`L` is the lower Cholesky of the n0×n0 covariance matrix. Matches `refine.py:generate_dense`.
"""
function generate_dense(coords::AbstractMatrix{UInt32}, scale::T,
        bins::AbstractVector, vals::AbstractVector, xi::AbstractVector{T}) where {T}
    n0 = length(xi)
    K = _assemble_dense_cov(coords, scale, bins, vals, n0)
    L = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(K, :L)).L
    return L * xi
end

"""
    generate_dense_inv(coords, scale, bins, vals, values) -> Vector{T}

Inverse of `generate_dense`: solve `L \\ values` to recover the unit-normal parameters.
Matches `refine.py:generate_dense_inv`.
"""
function generate_dense_inv(coords::AbstractMatrix{UInt32}, scale::T,
        bins::AbstractVector, vals::AbstractVector, values::AbstractVector{T}) where {T}
    n0 = length(values)
    K = _assemble_dense_cov(coords, scale, bins, vals, n0)
    L = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(K, :L)).L
    return LinearAlgebra.LowerTriangular(L) \ values
end

"""
    generate_dense_logdet(coords, scale, bins, vals, n0) -> T

Log-determinant contribution of the dense first layer: `sum(log(diag(L)))`.
Matches `refine.py:generate_dense_logdet`.
"""
function generate_dense_logdet(coords::AbstractMatrix{UInt32}, scale::T,
        bins::AbstractVector, vals::AbstractVector, n0::Int) where {T}
    K = _assemble_dense_cov(coords, scale, bins, vals, n0)
    L = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(K, :L)).L
    return sum(log, LinearAlgebra.diag(L))
end

# Gradient of generate_dense_logdet w.r.t. vals.
# Uses d(log|K|)/d(K[a,b]) = K^{-1}[a,b] (diagonal) or 2*K^{-1}[a,b] (off-diagonal),
# scattered through cov_lookup weights. K^{-1} = (L^{-1})^T * L^{-1}.
function _dense_logdet_grad_vals(prob::GraphGPProblem{T}) where {T}
    n0 = prob.n0
    D = ndims_space(prob)
    nb = nbins(prob)
    coords_n0 = view(prob.coords, :, 1:n0)
    K = _assemble_dense_cov(coords_n0, prob.scale, prob.bins, prob.vals, n0)
    L = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(K, :L)).L
    Linv = Matrix(LinearAlgebra.inv(LinearAlgebra.LowerTriangular(L)))
    dv = zeros(T, nb)
    for b in 1:n0
        for a in b:n0
            # K^{-1}[a,b] = sum_k Linv[k,a] * Linv[k,b]
            Kinv_ab = zero(T)
            for kk in 1:n0
                Kinv_ab += Linv[kk, a] * Linv[kk, b]
            end
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords_n0[d, a]) - Int64(coords_n0[d, b])
                sq += di * di
            end
            r = sqrt(T(sq)) * prob.scale
            lo, wlo, whi = cov_lookup_weights(r, prob.bins, nb)
            g = a == b ? Kinv_ab : 2 * Kinv_ab
            dv[lo] += g * wlo
            dv[lo + 1] += g * whi
        end
    end
    return dv
end

# Gradient of 0.5*||generate_dense_inv||^2 w.r.t. vals.
# With xi = L^{-1} y, d(0.5||xi||^2)/d(K[a,b]) = -0.5*xi[a]*xi[b] (diagonal)
# or -xi[a]*xi[b] (off-diagonal), from the matrix-calculus identity for y^T K^{-1} y.
function _dense_inv_loss_grad_vals(prob::GraphGPProblem{T},
        data_dense::AbstractVector{T}) where {T}
    n0 = prob.n0
    D = ndims_space(prob)
    nb = nbins(prob)
    coords_n0 = view(prob.coords, :, 1:n0)
    xi_d = generate_dense_inv(coords_n0, prob.scale, prob.bins, prob.vals, data_dense)
    dv = zeros(T, nb)
    for b in 1:n0
        xib = xi_d[b]
        for a in b:n0
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords_n0[d, a]) - Int64(coords_n0[d, b])
                sq += di * di
            end
            r = sqrt(T(sq)) * prob.scale
            lo, wlo, whi = cov_lookup_weights(r, prob.bins, nb)
            g = a == b ? -T(0.5) * xib * xi_d[a] : -xib * xi_d[a]
            dv[lo] += g * wlo
            dv[lo + 1] += g * whi
        end
    end
    return dv
end
