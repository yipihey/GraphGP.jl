# ChainRulesCore integration.
#
# Gradients in GraphGP are hand-written analytic adjoints (see grad.jl / kernels_adjoint.jl).
# Here we expose them as `rrule`s on small functional entry points that take the differentiable
# input *explicitly*, so any reverse-mode AD framework (e.g. Zygote) can compose them into an
# arbitrary scalar loss and chain through a user's kernel `θ -> cov_vals` map.
#
# The two functionals below are exactly the two terms of a GP marginal log-likelihood:
#   logdet_of_vals             →  0.5·log|K|   (via Σ log std)
#   inv_quadratic_loss_of_vals →  0.5·yᵀK⁻¹y   (= 0.5‖xi‖²)
# Differentiating their sum w.r.t. `vals` (and, through the user's kernel, the hyperparameters)
# gives the likelihood gradient used for GP training.

using ChainRulesCore: ChainRulesCore, rrule, NoTangent, @thunk, @not_implemented, unthunk

# Rebuild a problem with `vals` replaced (cheap struct; shares the other arrays). `vals` must
# live on the same backend as `prob.coords` (the inner constructor enforces this).
function _set_vals(prob::GraphGPProblem, vals::AbstractVector)
    return GraphGPProblem(prob.coords, prob.neighbors, prob.offsets, prob.n0, prob.scale,
        prob.bins, vals, prob.indices)
end

"""
    logdet_of_vals(prob, vals) -> T

`generate_logdet` with `prob.vals` replaced by `vals`; differentiable in `vals`. Use with an
AD framework to obtain `d(loss)/d(cov_vals)` (and, via your kernel `θ -> vals`, `d(loss)/dθ`).
"""
logdet_of_vals(prob::GraphGPProblem, vals::AbstractVector) = generate_logdet(_set_vals(prob, vals))

function ChainRulesCore.rrule(::typeof(logdet_of_vals), prob::GraphGPProblem, vals::AbstractVector)
    prob2 = _set_vals(prob, vals)
    y = generate_logdet(prob2)
    function logdet_of_vals_pullback(ȳ)
        dvals = @thunk(generate_logdet_grad_vals(prob2) .* unthunk(ȳ))
        return (NoTangent(), NoTangent(), dvals)
    end
    return y, logdet_of_vals_pullback
end

"""
    inv_quadratic_loss_of_vals(prob, vals, data) -> T

`0.5‖generate_inv(prob_with_vals, data)‖²`, differentiable in `vals`. `data` is in the
original (not tree/depth) ordering; reordering is handled internally.
"""
function inv_quadratic_loss_of_vals(prob::GraphGPProblem, vals::AbstractVector, data::AbstractVector)
    return sum(abs2, generate_inv(_set_vals(prob, vals), data)) / 2
end

function ChainRulesCore.rrule(::typeof(inv_quadratic_loss_of_vals), prob::GraphGPProblem,
        vals::AbstractVector, data::AbstractVector)
    prob2 = _set_vals(prob, vals)
    loss, dvals = generate_inv_loss_grad_vals(prob2, data)
    function inv_quadratic_loss_pullback(l̄)
        dv = @thunk(dvals .* unthunk(l̄))
        # d/ddata is well-defined (the loss is quadratic in data) but not yet provided
        # analytically; flag it rather than silently returning zero.
        return (NoTangent(), NoTangent(), dv,
            @not_implemented("d(inv_quadratic_loss_of_vals)/d(data) is not implemented yet"))
    end
    return loss, inv_quadratic_loss_pullback
end
