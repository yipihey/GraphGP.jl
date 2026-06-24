"""
Basic GP hyperparameter training in Julia using GraphGP.jl.

Generates synthetic data from a known RBF GP, then recovers (variance, scale)
by maximizing the log-marginal-likelihood with gradient ascent.

Run from repo root:
  /tmp/jl/bin/julia --project=. examples/train_compare.jl
"""

using GraphGP
using Random
using LinearAlgebra: norm
using Printf
using Statistics: std

# ── Problem setup ────────────────────────────────────────────────────────────

const rng       = Random.MersenneTwister(42)
const N, D      = 1000, 2
const n0_req, k = 50, 8
const R_MIN, R_MAX, N_BINS = 1e-4, 5.0, 300
const JITTER    = 1e-3
const N_ITER    = 40
const STEP_SIZE = 0.02   # fixed step in the unit-gradient direction (gradient ascent)

const TRUE_VARIANCE = 1.5
const TRUE_SCALE    = 0.4

println("GraphGP Julia training example")
println("  N=$N  D=$D  n0≈$n0_req  k=$k  n_bins=$N_BINS")

# Random points in [0,1]^D
pts = rand(rng, Float64, N, D)

# Build graph once (structure is fixed; only kernel vals change during training)
t_build = @elapsed begin
    bins0, vals0 = rbf_kernel(TRUE_VARIANCE, TRUE_SCALE, R_MIN, R_MAX, N_BINS; jitter=JITTER)
    prob_struct  = build_graph(pts, n0_req, k, bins0, vals0)
end
println("Graph build: $(round(t_build*1000, digits=1)) ms  " *
    "(n0=$(prob_struct.n0), N=$(npoints(prob_struct)), k=$(nneighbors(prob_struct)))")

# Generate synthetic observations from the true GP
xi_true = randn(rng, Float64, N)
data    = generate(prob_struct, xi_true)
println("Data generated: length=$(length(data)), std=$(round(std(data), digits=3))")

# Helper: attach updated kernel vals to the fixed graph structure
function make_prob(variance, scale)
    bins, vals = rbf_kernel(variance, scale, R_MIN, R_MAX, N_BINS; jitter=JITTER)
    return GraphGPProblem(prob_struct.coords, prob_struct.neighbors, prob_struct.offsets,
        prob_struct.n0, prob_struct.scale, bins, vals, prob_struct.indices)
end

make_rbf = (v, s) -> rbf_kernel(v, s, R_MIN, R_MAX, N_BINS; jitter=JITTER)

# ── Training loop: gradient ASCENT on log-likelihood ─────────────────────────
# loss(hp) = -logdet(K) - 0.5*||xi||^2  =  log p(data|hp) + const
# Maximise loss  →  hp += step * ∇loss / ‖∇loss‖
println("\nTraining (unit-gradient ascent on log-likelihood, step=$STEP_SIZE):")
println("  Start: variance=1.0  scale=0.7   True: $TRUE_VARIANCE  $TRUE_SCALE")
println()
println("  iter │    loss    │ variance │  scale  │  ‖g‖    │  ms/iter")
println("  ─────┼────────────┼──────────┼─────────┼─────────┼─────────")

let variance = 1.0, scale = 0.7
    local loss = 0.0
    local gnorm = 0.0
    for iter in 1:N_ITER
        t0 = time_ns()
        prob = make_prob(variance, scale)

        # Fused forward+backward: each pass computes value AND gradient in one traversal.
        ld,  g_ld  = generate_logdet_and_grad_vals(prob)
        inv_loss, g_inv = generate_inv_loss_grad_vals(prob, data)
        loss = -ld - inv_loss

        # ∂loss/∂vals = -(∂logdet/∂vals + ∂inv_loss/∂vals)
        g_vals = -(g_ld .+ g_inv)

        # Chain rule → ∂loss/∂(variance, scale)
        g = hyperparam_grad(g_vals, make_rbf, [variance, scale])
        gnorm = norm(g)

        # Unit-gradient ascent step (maximise log-likelihood)
        if gnorm > 1e-12
            g ./= gnorm
        end
        variance = max(variance + STEP_SIZE * g[1], 1e-3)
        scale    = max(scale    + STEP_SIZE * g[2], 1e-3)
        t = (time_ns() - t0) / 1e6

        if iter == 1 || iter % 5 == 0 || iter == N_ITER
            @printf("  %4d │ %10.3f │ %8.4f │ %7.4f │ %7.3f │ %6.1f\n",
                iter, loss, variance, scale, gnorm, t)
        end
    end

    println()
    @printf("  Final:  variance=%.4f  scale=%.4f\n", variance, scale)
    @printf("  True:   variance=%.4f  scale=%.4f\n", TRUE_VARIANCE, TRUE_SCALE)
end
