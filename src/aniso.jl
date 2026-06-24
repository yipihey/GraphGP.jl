# Anisotropic covariance K(Δspatial, Δz) — feature-parity with the Python fork's graphgp/aniso.py.
#
# The default kernel is isotropic: a pair reduces to one Euclidean distance looked up in a 1-D
# table (cov_lookup). Survey clustering in OBSERVED coordinates is anisotropic — the correlation
# depends SEPARATELY on angular separation Δθ and redshift separation Δz. `AnisoCov` represents
# that as a 2-D table grid[i,j] = K(spatial_bins[i], z_bins[j]) with bilinear interpolation.
#
# Points are embedded with the LAST coordinate radial (alpha*z) and the first D-1 spatial (n̂).
# For a pair on the shared integer lattice (one `scale`):
#     d_spatial = sqrt(Σ_{d=1}^{D-1} (jc[a,d]-jc[b,d])²) * scale
#     d_z       = |jc[a,D]-jc[b,D]| * scale / alpha          (divide the embedding scale back out)
# matching aniso.py's  d_spatial = ‖Δ[:-1]‖,  d_z = |Δ[-1]|/alpha.
#
# Like the fork, the change is localized to the covariance assembly (assemble_cov!); the dense
# block and the KA/native forward kernels gain anisotropic variants, and generate / refine /
# generate_inv / generate_logdet dispatch on `prob.cov`. The isotropic path is untouched.
# This is forward-only (no gradients) — the measured BOSS kernel is fixed.

# `AnisoCov` is defined in types.jl (it is a field type of GraphGPProblem).

"""
    build_anisotropic_covariance(spatial_bins, z_bins, grid, alpha; jitter = 0) -> AnisoCov

Construct a 2-D tabulated covariance K(Δspatial, Δz). `grid` is `(n_s, n_z)`; `spatial_bins`
`(n_s,)` and `z_bins` `(n_z,)` are increasing (first entry 0). `grid[1,1]` (zero separation) is
inflated by `1 + jitter` for positive-definiteness, mirroring the isotropic nugget. Element type
follows `grid` (use Float32 for the production path). Mirrors `build_anisotropic_covariance`.
"""
function build_anisotropic_covariance(spatial_bins::AbstractVector, z_bins::AbstractVector,
        grid::AbstractMatrix, alpha::Real; jitter::Real = 0)
    T = float(eltype(grid))
    size(grid) == (length(spatial_bins), length(z_bins)) ||
        throw(ArgumentError("grid must be (length(spatial_bins), length(z_bins))"))
    sb = collect(T, spatial_bins)
    zb = collect(T, z_bins)
    g = Matrix{T}(grid)
    @inbounds g[1, 1] *= (one(T) + T(jitter))
    return AnisoCov{T, Vector{T}, Matrix{T}}(sb, zb, g, T(alpha))
end

# Fractional grid index for separation `d` on a (possibly non-uniform) increasing axis `bins`,
# matching jnp.interp(d, bins, 0:n-1): 0-based, clamped to [0, n-1]. Device-friendly (scalar).
@inline function _frac_index(d::T, bins, n::Int) where {T}
    @inbounds begin
        d <= T(bins[1]) && return zero(T)
        d >= T(bins[n]) && return T(n - 1)
        lo = 1
        hi = n
        while hi - lo > 1
            mid = (lo + hi) >>> 1
            (T(bins[mid]) <= d) ? (lo = mid) : (hi = mid)
        end
        x0 = T(bins[lo])
        x1 = T(bins[lo + 1])
        return T(lo - 1) + (d - x0) / (x1 - x0)
    end
end

"""
    aniso_lookup(d_spatial, d_z, spatial_bins, z_bins, grid) -> T

Bilinear interpolation of `grid` at the fractional indices of `(d_spatial, d_z)` on the bin
axes — the Julia analog of `aniso_evaluate` / `map_coordinates(grid, [ix, iz], order=1,
mode="nearest")`. Indices are clamped to the grid edges. Device-friendly (only scalar/array
indexing); `grid` lives on the same backend as `coords`.
"""
@inline function aniso_lookup(d_spatial::T, d_z::T, spatial_bins, z_bins, grid) where {T}
    @inbounds begin
        ns = size(grid, 1)
        nz = size(grid, 2)
        fx = _frac_index(d_spatial, spatial_bins, ns)   # 0-based, in [0, ns-1]
        fz = _frac_index(d_z, z_bins, nz)               # 0-based, in [0, nz-1]
        i0 = unsafe_trunc(Int, fx)                      # floor (fx >= 0)
        j0 = unsafe_trunc(Int, fz)
        i1 = i0 + 1 < ns ? i0 + 1 : ns - 1              # clamp to edge ("nearest")
        j1 = j0 + 1 < nz ? j0 + 1 : nz - 1
        tx = fx - T(i0)
        tz = fz - T(j0)
        g00 = T(grid[i0 + 1, j0 + 1]); g10 = T(grid[i1 + 1, j0 + 1])   # +1: 1-based grid
        g01 = T(grid[i0 + 1, j1 + 1]); g11 = T(grid[i1 + 1, j1 + 1])
        return (g00 * (one(T) - tx) + g10 * tx) * (one(T) - tz) +
               (g01 * (one(T) - tx) + g11 * tx) * tz
    end
end

# Anisotropic per-block covariance assembly (analog of the isotropic assemble_cov!). Splits the
# joint coords into spatial (1:D-1) and radial (D); diagonal = K(0,0) = grid[1,1] (incl. jitter).
@inline function assemble_cov!(A, jc, ::Val{KP1}, ::Val{D}, scale::T,
        spatial_bins, z_bins, grid, alpha::T) where {KP1, D, T}
    @inbounds begin
        diagv = aniso_lookup(zero(T), zero(T), spatial_bins, z_bins, grid)
        for a in 1:KP1
            A[a, a] = diagv
            for b in 1:(a - 1)
                sqs = zero(Int64)
                for dd in 1:(D - 1)                      # spatial dims
                    di = Int64(jc[a, dd]) - Int64(jc[b, dd])
                    sqs += di * di
                end
                dzi = Int64(jc[a, D]) - Int64(jc[b, D])  # radial dim
                d_spatial = sqrt(T(sqs)) * scale
                d_z = abs(T(dzi)) * scale / alpha
                A[a, b] = aniso_lookup(d_spatial, d_z, spatial_bins, z_bins, grid)
            end
        end
    end
    return nothing
end

# ── KA forward kernels (anisotropic): same bodies as the isotropic kernels, but the covariance
# is assembled from (spatial_bins, z_bins, grid, alpha) instead of (bins, vals, nbins). ──

@kernel function refine_logdet_kernel_aniso!(logdet_terms, @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(spatial_bins), @Const(z_bins), @Const(grid), alpha::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A = @private T (K + 1, K + 1)
    jc = @private UInt32 (K + 1, D)
    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, spatial_bins, z_bins, grid, alpha)
    chol_lower!(A, Val(K + 1))
    @inbounds logdet_terms[m] = log(A[K + 1, K + 1])
end

@kernel function refine_inv_kernel_aniso!(xi_out, @Const(coords), @Const(neighbors), @Const(values),
        n0, scale::T, @Const(spatial_bins), @Const(z_bins), @Const(grid), alpha::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A = @private T (K + 1, K + 1)
    jc = @private UInt32 (K + 1, D)
    mv = @private T (K,)
    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, spatial_bins, z_bins, grid, alpha)
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

@kernel function refine_meanvec_std_kernel_aniso!(mean_vec, std, @Const(coords), @Const(neighbors),
        n0, scale::T, @Const(spatial_bins), @Const(z_bins), @Const(grid), alpha::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    m = @index(Global)
    A = @private T (K + 1, K + 1)
    jc = @private UInt32 (K + 1, D)
    mv = @private T (K,)
    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, spatial_bins, z_bins, grid, alpha)
    chol_lower!(A, Val(K + 1))
    mean_vec_solve!(mv, A, Val(K))
    @inbounds begin
        for j in 1:K
            mean_vec[j, m] = mv[j]
        end
        std[m] = A[K + 1, K + 1]
    end
end

# ── Native multithreaded CPU variants (the KA CPU backend is far slower; see cpu_native.jl). ──

function _native_refine_logdet_terms_aniso!(terms, prob::GraphGPProblem{T}, ::Val{K}, ::Val{D}) where {T, K, D}
    coords = prob.coords; neighbors = prob.neighbors; n0 = prob.n0; scale = prob.scale
    cov = prob.cov::AnisoCov
    sb = cov.spatial_bins; zb = cov.z_bins; grid = cov.grid; alpha = T(cov.alpha)
    M = nrefined(prob)
    _threaded_points!(M, Val(K), Val(D), T) do m, A, jc, _mv
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, sb, zb, grid, alpha)
        chol_lower!(A, Val(K + 1))
        terms[m] = log(A[K + 1, K + 1])
    end
    return terms
end

function _native_refine_inv_aniso!(xi_out, prob::GraphGPProblem{T}, values, ::Val{K}, ::Val{D}) where {T, K, D}
    coords = prob.coords; neighbors = prob.neighbors; n0 = prob.n0; scale = prob.scale
    cov = prob.cov::AnisoCov
    sb = cov.spatial_bins; zb = cov.z_bins; grid = cov.grid; alpha = T(cov.alpha)
    M = nrefined(prob)
    _threaded_points!(M, Val(K), Val(D), T) do m, A, jc, mv
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, sb, zb, grid, alpha)
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

function _native_refine_meanvec_std_aniso!(mean_vec, std, prob::GraphGPProblem{T}, ::Val{K}, ::Val{D}) where {T, K, D}
    coords = prob.coords; neighbors = prob.neighbors; n0 = prob.n0; scale = prob.scale
    cov = prob.cov::AnisoCov
    sb = cov.spatial_bins; zb = cov.z_bins; grid = cov.grid; alpha = T(cov.alpha)
    M = nrefined(prob)
    _threaded_points!(M, Val(K), Val(D), T) do m, A, jc, mv
        _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
        assemble_cov!(A, jc, Val(K + 1), Val(D), scale, sb, zb, grid, alpha)
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

# ── Dense first-layer (host LAPACK), anisotropic ──

function _assemble_dense_cov_aniso(coords::AbstractMatrix{UInt32}, scale::T, cov::AnisoCov,
        n::Int) where {T}
    coords = Array(coords)
    sb = Array(cov.spatial_bins); zb = Array(cov.z_bins); grid = Array(cov.grid)
    alpha = T(cov.alpha)
    D = size(coords, 1)
    K = Matrix{T}(undef, n, n)
    diagv = aniso_lookup(zero(T), zero(T), sb, zb, grid)
    @inbounds for j in 1:n
        K[j, j] = diagv
        for i in (j + 1):n
            sqs = zero(Int64)
            for d in 1:(D - 1)
                di = Int64(coords[d, i]) - Int64(coords[d, j])
                sqs += di * di
            end
            dzi = Int64(coords[D, i]) - Int64(coords[D, j])
            d_spatial = sqrt(T(sqs)) * scale
            d_z = abs(T(dzi)) * scale / alpha
            v = aniso_lookup(d_spatial, d_z, sb, zb, grid)
            K[i, j] = v
            K[j, i] = v
        end
    end
    return K
end

function generate_dense_aniso(coords::AbstractMatrix{UInt32}, scale::T, cov::AnisoCov,
        xi::AbstractVector{T}) where {T}
    xi = Array(xi)
    n0 = length(xi)
    Kd = _assemble_dense_cov_aniso(coords, scale, cov, n0)
    L = _dense_chol_L(Kd)
    return L * xi
end

function generate_dense_inv_aniso(coords::AbstractMatrix{UInt32}, scale::T, cov::AnisoCov,
        values::AbstractVector{T}) where {T}
    values = Array(values)
    n0 = length(values)
    Kd = _assemble_dense_cov_aniso(coords, scale, cov, n0)
    L = _dense_chol_L(Kd)
    return LinearAlgebra.LowerTriangular(L) \ values
end

function generate_dense_logdet_aniso(coords::AbstractMatrix{UInt32}, scale::T, cov::AnisoCov,
        n0::Int) where {T}
    Kd = _assemble_dense_cov_aniso(coords, scale, cov, n0)
    L = _dense_chol_L(Kd)
    return sum(log, LinearAlgebra.diag(L))
end
