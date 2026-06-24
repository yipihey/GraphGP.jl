# GraphGP.jl — parity + differentiability example
# =================================================
#
# This walks through the GraphGP.jl public API and highlights the two things it adds over the
# JAX `graphgp` stack: (1) the same fused per-point kernel runs on CPU *and* GPU from one code
# path, and (2) it differentiates — w.r.t. the covariance table, the kernel hyperparameters,
# AND the point positions — which the GPU CUDA extension cannot do.
#
# Run (CPU):
#   julia --project=. examples/parity_and_autodiff.jl
# For the GPU, add CUDA to the environment and uncomment the `to_backend` block at the end.

using GraphGP
using Random
using Printf

# ----------------------------------------------------------------------------------------------
# 1. Build a graph.  build_graph quantises the points onto an integer lattice, builds the k-d
#    tree, finds each point's k preceding nearest neighbours, and orders by depth — the same
#    pipeline as graphgp.build_graph, all in Julia.
# ----------------------------------------------------------------------------------------------
rng = MersenneTwister(0)
N, D, n0, k = 50_000, 3, 1000, 10
points = randn(rng, N, D)                                    # (N, D) Float64

# A covariance table (RBF), exactly as graphgp.extras.rbf_kernel produces it.
variance, scale = 1.0, 0.3
bins, vals = rbf_kernel(variance, scale, 1e-4, 1e1, 1000; jitter = 1e-3)

prob = build_graph(points, n0, k, bins, vals)
@printf("graph: N=%d  refined M=%d  k=%d  D=%d\n", npoints(prob), nrefined(prob), nneighbors(prob), ndims_space(prob))
check_graph(prob)                                           # throws if topological order / causality is violated

# ----------------------------------------------------------------------------------------------
# 2. Forward evaluations (these match graphgp.refine_* / generate_* to f32 round-off).
# ----------------------------------------------------------------------------------------------
ld = generate_logdet(prob)                                  # dense first layer + Vecchia refine
xi = randn(rng, N)                                          # unit-normal parameters (problem eltype)
v  = generate(prob, xi)                                     # draw a field from unit-normal xi
xi_back = generate_inv(prob, v)                             # exact inverse
@printf("generate_logdet = %.6g    ||generate_inv∘generate − xi|| = %.2e\n",
        ld, sqrt(sum(abs2, xi_back .- xi)))

# ----------------------------------------------------------------------------------------------
# 3. Differentiability — the part the CUDA extension lacks.
#    Each of these is an analytic adjoint (not finite differences, not slow autodiff).
# ----------------------------------------------------------------------------------------------
# 3a. ∂ logdet / ∂ cov_vals  (cotangent on the covariance table)
g_vals = refine_logdet_grad_vals(prob)
@printf("∂logdet/∂cov_vals : length %d, ||·|| = %.4g\n", length(g_vals), sqrt(sum(abs2, g_vals)))

# 3b. ∂ logdet / ∂ hyperparameters  (chain the table gradient through the kernel constructor)
make_kernel(variance, scale) = rbf_kernel(variance, scale, 1e-4, 1e1, 1000; jitter = 1e-3)
g_hyper = hyperparam_grad(g_vals, make_kernel, [variance, scale])
@printf("∂logdet/∂[variance, scale] = [% .6g, % .6g]\n", g_hyper[1], g_hyper[2])

# 3c. ∂ logdet / ∂ points  (positions treated as continuous; straight-through the lattice)
g_points = generate_logdet_grad_points(prob)                # (D, N)
@printf("∂logdet/∂points : size %s, ||·|| = %.4g\n", string(size(g_points)), sqrt(sum(abs2, g_points)))

# 3d. ∂ (½‖generate_inv(v)‖²) / ∂ cov_vals  — a data-dependent loss gradient
loss, g_loss = generate_inv_loss_grad_vals(prob, v)
@printf("inv-quadratic loss = %.6g    ||∂loss/∂cov_vals|| = %.4g\n", loss, sqrt(sum(abs2, g_loss)))

# These also compose with Zygote/ChainRules for arbitrary scalar losses via the exported
# rrule-backed entry points `logdet_of_vals`, `inv_quadratic_loss_of_vals`,
# `logdet_of_points`, ...  (add Zygote to the environment to use `Zygote.gradient`).

# ----------------------------------------------------------------------------------------------
# 4. GPU — the *same* problem and the *same* calls, just moved to the device. (Requires CUDA.)
# ----------------------------------------------------------------------------------------------
#   using CUDA
#   gpu = to_backend(prob, CUDABackend())
#   ld_gpu  = generate_logdet(gpu)             # matches `ld` to ~1e-5 (f32 / fast-math)
#   g_gpu   = Array(refine_logdet_grad_vals(gpu))
#   @assert isapprox(ld_gpu, ld; rtol = 1e-3)

println("\nAll forward + gradient entry points ran on the CPU backend.")
println("The identical calls run on CUDA via `to_backend(prob, CUDABackend())`.")
