# Kernel hyperparameter layer: discretize continuous covariance functions onto the same
# logspaced bin grid that `cov_lookup` consumes. Matches `graphgp/extras.py` exactly.
# Also provides `hyperparam_grad` — chain rule from the existing cov_vals gradient to
# gradient w.r.t. kernel hyperparameters, via a ForwardDiff Jacobian.

using SpecialFunctions: loggamma
using ForwardDiff: ForwardDiff

"""
    make_cov_bins(r_min, r_max, n_bins) -> Vector{Float64}

Logarithmically-spaced covariance grid with `0.0` prepended. Matches `extras.py:make_cov_bins`.
"""
function make_cov_bins(r_min::Real, r_max::Real, n_bins::Int)
    inner = exp10.(range(log10(Float64(r_min)), log10(Float64(r_max)), n_bins - 1))
    return Float64[0.0; inner]
end

"""
    rbf_kernel(variance, scale, r_min, r_max, n_bins; jitter) -> (bins, vals)

Squared-exponential (RBF) covariance discretized onto `n_bins` logspaced bins.
`vals[i] = variance * exp(-0.5 * (bins[i]/scale)^2)`, with `vals[1]` multiplied by
`(1 + jitter)`. Matches `extras.py:rbf_kernel`. ForwardDiff-compatible.
"""
function rbf_kernel(variance, scale, r_min::Real, r_max::Real, n_bins::Int;
        jitter=zero(variance))
    bins = make_cov_bins(r_min, r_max, n_bins)
    vals = map(eachindex(bins)) do i
        v = variance * exp(-oneunit(variance) / 2 * (bins[i] / scale)^2)
        i == 1 ? v * (1 + jitter) : v
    end
    return bins, vals
end

# Horner evaluation of a polynomial given by descending coefficients at z.
# Matches jnp.polyval semantics: coeffs[1]*z^(n-1) + coeffs[2]*z^(n-2) + ... + coeffs[n].
function _polyval(coeffs::AbstractVector{Float64}, z::T) where {T}
    acc = T(coeffs[1])
    for i in 2:length(coeffs)
        acc = acc * z + T(coeffs[i])
    end
    return acc
end

"""
    matern_kernel(p, variance, cutoff, r_min, r_max, n_bins; jitter) -> (bins, vals)

Matérn covariance with ν = p + 1/2 discretized onto `n_bins` logspaced bins.
Polynomial coefficients via `SpecialFunctions.loggamma`. Matches `extras.py:matern_kernel`.
ForwardDiff-compatible w.r.t. `variance` and `cutoff`; `p` is an integer constant.
"""
function matern_kernel(p::Int, variance, cutoff, r_min::Real, r_max::Real, n_bins::Int;
        jitter=zero(variance))
    bins = make_cov_bins(r_min, r_max, n_bins)
    # Polynomial coefficients (fixed Float64, independent of differentiable hyperparams).
    log_coeffs = [loggamma(p + 1.0) + loggamma(p + i + 1.0) - loggamma(i + 1.0) -
                  loggamma(p - i + 1.0) - loggamma(2p + 1.0) for i in 0:p]
    coeffs = exp.(log_coeffs)
    sqrt2p1 = sqrt(Float64(2p + 1))  # Float64 constant
    vals = map(eachindex(bins)) do idx
        # bins[idx] is Float64; cutoff may be Dual — division promotes to Dual.
        xi = sqrt2p1 * bins[idx] / cutoff
        v = variance * exp(-xi) * _polyval(coeffs, 2 * xi)
        idx == 1 ? v * (1 + jitter) : v
    end
    return bins, vals
end

"""
    hyperparam_grad(grad_cov_vals, make_kernel, hyperparams) -> grad_hyperparams

Gradient of any scalar objective w.r.t. kernel hyperparameters, given:
- `grad_cov_vals`: ∂(objective)/∂(cov_vals) from `refine_logdet_grad_vals` etc.
- `make_kernel(hyperparams...) → (_, vals)`: kernel factory (must be ForwardDiff-compatible).
- `hyperparams`: current hyperparameter vector (e.g. `[variance, scale]`).

Computes the (nbins × nhyperparams) Jacobian J = ∂(cov_vals)/∂(hyperparams) via ForwardDiff,
then returns J' * grad_cov_vals via the chain rule. Cost: O(nbins × nhyperparams) — cheap.
"""
function hyperparam_grad(grad_cov_vals::AbstractVector, make_kernel::Function,
        hyperparams::AbstractVector{<:Real})
    hp64 = Float64.(hyperparams)
    gv64 = Float64.(grad_cov_vals)
    J = ForwardDiff.jacobian(h -> collect(make_kernel(h...)[2]), hp64)
    return J' * gv64
end
