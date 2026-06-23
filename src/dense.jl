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
    # The dense first layer is small (n0 ≲ 1000) and uses host LAPACK; materialize the inputs
    # on the CPU so this path works regardless of whether the problem lives on a GPU.
    coords = Array(coords)
    bins = Array(bins)
    vals = Array(vals)
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
    compute_cov_matrix(coords_a, coords_b, scale, bins, vals) -> Matrix
    compute_cov_matrix(prob::GraphGPProblem) -> Matrix

Dense cross-covariance between two coordinate sets `coords_a` (D×nₐ) and `coords_b` (D×n_b)
on the integer lattice, using the same integer-distance + `cov_lookup` interpolation as the
KA kernels. Returns an `nₐ × n_b` matrix. The single-argument form returns the full `N × N`
covariance for all points in `prob` (debugging/reference path; `O(N²)`). Mirrors
`graphgp.compute_cov_matrix`.
"""
function compute_cov_matrix(coords_a::AbstractMatrix{UInt32}, coords_b::AbstractMatrix{UInt32},
        scale::T, bins::AbstractVector, vals::AbstractVector) where {T}
    nb = length(bins)
    D = size(coords_a, 1)
    size(coords_b, 1) == D || throw(ArgumentError("coordinate dimension mismatch"))
    na = size(coords_a, 2)
    nbpts = size(coords_b, 2)
    Kmat = Matrix{T}(undef, na, nbpts)
    @inbounds for j in 1:nbpts
        for i in 1:na
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords_a[d, i]) - Int64(coords_b[d, j])
                sq += di * di
            end
            r = sqrt(T(sq)) * scale
            Kmat[i, j] = cov_lookup(r, bins, vals, nb)
        end
    end
    return Kmat
end

function compute_cov_matrix(prob::GraphGPProblem)
    coords = Array(prob.coords)
    return compute_cov_matrix(coords, coords, prob.scale, Array(prob.bins), Array(prob.vals))
end

"""
    generate_dense(coords, scale, bins, vals, xi) -> Vector{T}

Sample the first n0 = length(xi) points from the dense GP: returns `L * xi` where
`L` is the lower Cholesky of the n0×n0 covariance matrix. Matches `refine.py:generate_dense`.
"""
function generate_dense(coords::AbstractMatrix{UInt32}, scale::T,
        bins::AbstractVector, vals::AbstractVector, xi::AbstractVector{T}) where {T}
    xi = Array(xi)                       # host LAPACK path (small dense block)
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
    values = Array(values)               # host LAPACK path (small dense block)
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

# Gradient of generate_dense_logdet = 0.5·log|K| w.r.t. vals.
# d(0.5 log|K|)/d(cov(r_ab)) = 0.5·K^{-1}[a,a] on the diagonal, and K^{-1}[a,b] off-diagonal
# (the off-diagonal scalar cov sets both K[a,b] and K[b,a], contributing 0.5·(Kinv_ab+Kinv_ba)
# = Kinv_ab). K^{-1} = (L^{-1})^T·L^{-1}. Scattered through cov_lookup weights.
function _dense_logdet_grad_vals(prob::GraphGPProblem{T}) where {T}
    n0 = prob.n0
    D = ndims_space(prob)
    nb = nbins(prob)
    coords_n0 = Array(view(prob.coords, :, 1:n0))   # host (small dense block)
    bins = Array(prob.bins)
    K = _assemble_dense_cov(coords_n0, prob.scale, bins, prob.vals, n0)
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
            lo, wlo, whi = cov_lookup_weights(r, bins, nb)
            g = a == b ? T(0.5) * Kinv_ab : Kinv_ab
            dv[lo] += g * wlo
            dv[lo + 1] += g * whi
        end
    end
    return dv
end

# Gradient of generate_dense_logdet w.r.t. the (dequantized) positions of the first n0 points.
# Same cotangent g = 2·K⁻¹[a,b] on each off-diagonal covariance as the cov_vals path, then
# propagated through k'(r)·dr/dx. Diagonal entries (r=0) are position-independent → skipped.
function _dense_logdet_grad_points(prob::GraphGPProblem{T}) where {T}
    n0 = prob.n0
    D = ndims_space(prob)
    nb = nbins(prob)
    N = npoints(prob)
    coords_n0 = Array(view(prob.coords, :, 1:n0))
    bins = Array(prob.bins)
    vals = Array(prob.vals)
    K = _assemble_dense_cov(coords_n0, prob.scale, bins, vals, n0)
    L = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(K, :L)).L
    Linv = Matrix(LinearAlgebra.inv(LinearAlgebra.LowerTriangular(L)))
    dpts = zeros(T, D, N)
    @inbounds for b in 1:n0
        for a in (b + 1):n0
            Kinv_ab = zero(T)
            for kk in 1:n0
                Kinv_ab += Linv[kk, a] * Linv[kk, b]
            end
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords_n0[d, a]) - Int64(coords_n0[d, b])
                sq += di * di
            end
            sq == 0 && continue
            r = sqrt(T(sq)) * prob.scale
            dcov = cov_lookup_dr(r, bins, vals, nb)
            dcov == 0 && continue
            factor = Kinv_ab * dcov / r          # off-diagonal cotangent = K⁻¹[a,b]
            for d in 1:D
                diff = prob.scale * (T(Int64(coords_n0[d, a])) - T(Int64(coords_n0[d, b])))
                c = factor * diff
                dpts[d, a] += c
                dpts[d, b] -= c
            end
        end
    end
    return dpts
end

# α = K⁻¹·y for the dense inverse loss 0.5‖xi‖² = 0.5·yᵀK⁻¹y (xi = L⁻¹y from generate_dense_inv,
# so α = L⁻ᵀ·xi). Shared by the cov_vals and points gradients below.
function _dense_inv_alpha(prob::GraphGPProblem{T}, data_dense) where {T}
    n0 = prob.n0
    coords_n0 = Array(view(prob.coords, :, 1:n0))
    bins = Array(prob.bins)
    vals = Array(prob.vals)
    K = _assemble_dense_cov(coords_n0, prob.scale, bins, vals, n0)
    L = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(K, :L)).L
    Lt = LinearAlgebra.LowerTriangular(L)
    xi = Lt \ Array(data_dense)          # L⁻¹ y
    alpha = transpose(Lt) \ xi           # L⁻ᵀ xi = K⁻¹ y
    return coords_n0, bins, vals, alpha
end

# Gradient of 0.5*‖generate_dense_inv‖² = 0.5·yᵀK⁻¹y w.r.t. vals.
# d/d(cov(r_ab)) = -α[a]·α[b] off-diagonal, -0.5·α[a]² on the diagonal, with α = K⁻¹y.
function _dense_inv_loss_grad_vals(prob::GraphGPProblem{T},
        data_dense::AbstractVector{T}) where {T}
    n0 = prob.n0
    D = ndims_space(prob)
    nb = nbins(prob)
    coords_n0, bins, _, alpha = _dense_inv_alpha(prob, data_dense)
    dv = zeros(T, nb)
    for b in 1:n0
        for a in b:n0
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords_n0[d, a]) - Int64(coords_n0[d, b])
                sq += di * di
            end
            r = sqrt(T(sq)) * prob.scale
            lo, wlo, whi = cov_lookup_weights(r, bins, nb)
            g = a == b ? -T(0.5) * alpha[a] * alpha[a] : -alpha[a] * alpha[b]
            dv[lo] += g * wlo
            dv[lo + 1] += g * whi
        end
    end
    return dv
end

# Gradient of 0.5*‖generate_dense_inv‖² w.r.t. the positions of the first n0 points.
# Off-diagonal cotangent -α[a]·α[b] propagated through k'(r)·dr/dx; diagonal (r=0) is
# position-independent and skipped.
function _dense_inv_loss_grad_points(prob::GraphGPProblem{T},
        data_dense::AbstractVector{T}) where {T}
    n0 = prob.n0
    D = ndims_space(prob)
    nb = nbins(prob)
    N = npoints(prob)
    coords_n0, bins, vals, alpha = _dense_inv_alpha(prob, data_dense)
    dpts = zeros(T, D, N)
    @inbounds for b in 1:n0
        for a in (b + 1):n0
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords_n0[d, a]) - Int64(coords_n0[d, b])
                sq += di * di
            end
            sq == 0 && continue
            r = sqrt(T(sq)) * prob.scale
            dcov = cov_lookup_dr(r, bins, vals, nb)
            dcov == 0 && continue
            factor = (-alpha[a] * alpha[b]) * dcov / r
            for d in 1:D
                diff = prob.scale * (T(Int64(coords_n0[d, a])) - T(Int64(coords_n0[d, b])))
                c = factor * diff
                dpts[d, a] += c
                dpts[d, b] -= c
            end
        end
    end
    return dpts
end
