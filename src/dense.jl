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
