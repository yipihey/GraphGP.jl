# Marginal-likelihood hyperparameter MLE for the GraphGP (stretched-exponential) kernel, run in ONE
# Julia process (the optimization loop never crosses the subprocess boundary, so no per-eval cold
# start). This is the showcase of GraphGP.jl's analytic gradients — the capability the CUDA ext lacks.
#
#   julia -t N --project=bench run_kernel_mle.jl <in.npz> <out.npz> <fit|gradcheck>
#
# Objective: negative log marginal likelihood (drop the ½N·log2π constant)
#   f(θ) = ½ logdet K(θ) + ½ yᵀ K(θ)⁻¹ y
# with exact gradient assembled from
#   ∂(½logdet)/∂vals = ½ generate_logdet_grad_vals,   ∂(½yᵀK⁻¹y)/∂vals = generate_inv_loss_grad_vals,
# chained to θ by hyperparam_grad (ForwardDiff through make_kernel). Kernel is the ECHOES
# stretched-exp A·exp(-(r/r0)^α) on the FIXED bin grid; θ = [logA, log r0, α].
#
# in.npz : coords, neighbors, offsets, n0, scale, cov_bins32 (the fixed grid), indices, y (data
#          field, ORIGINAL order), theta0 (3,). out.npz: fit -> theta_hat, nlml, nlml0, gnorm, niter;
#          gradcheck -> f, g_analytic, g_fd (central differences), rel.

using GraphGP, KernelAbstractions, NPZ, Printf

in_npz, out_npz, mode = ARGS[1], ARGS[2], ARGS[3]
@assert mode in ("fit", "gradcheck")

d = npzread(in_npz)
coords = permutedims(UInt32.(d["coords"]))
neighbors = permutedims(Int.(d["neighbors"])) .+ 1
offsets = Int.(d["offsets"])
n0 = Int(d["n0"]); scale = Float64(d["scale"])
bins = Float64.(d["cov_bins32"])
indices = haskey(d, "indices") ? (Int.(vec(d["indices"])) .+ 1) : nothing
y = Float64.(vec(d["y"]))
θ0 = Float64.(vec(d["theta0"]))
JIT = 1e-3

# make_kernel(logA, logr0, α) -> (bins, vals); ForwardDiff-compatible (no in-place, no control flow
# on the dual). vals[1] inflated by (1+JIT) so the dense first block stays positive-definite.
function make_kernel(logA, logr0, α)
    A = exp(logA); r0 = exp(logr0)
    vals = map(eachindex(bins)) do i
        v = A * exp(-((bins[i] / r0)^α))
        i == 1 ? v * (1 + JIT) : v
    end
    return bins, vals
end

probwith(vals) = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals, indices)

# f(θ), ∇f(θ): exact NLML and its gradient w.r.t. θ.
function fg(θ)
    _, vals = make_kernel(θ[1], θ[2], θ[3])
    prob = probwith(vals)
    ld = generate_logdet(prob)
    loss, dvals_loss = generate_inv_loss_grad_vals(prob, y)      # (½yᵀK⁻¹y, ∂/∂vals)
    f = 0.5 * ld + loss
    dvals = 0.5 .* generate_logdet_grad_vals(prob) .+ dvals_loss
    g = hyperparam_grad(dvals, make_kernel, θ)       # hyperparam_grad splats: make_kernel(θ...)
    return f, g
end

# Non-PD kernels (e.g. r0 → huge makes the dense first block singular) throw; return Inf so the
# line search rejects the step and backtracks, keeping the optimizer in the feasible region.
function fg_safe(θ)
    try
        return fg(θ)
    catch
        return Inf, fill(NaN, length(θ))
    end
end

function gradcheck(θ0, fg)
    f0, g = fg(θ0)
    h = 1e-5
    gfd = similar(θ0)
    for i in eachindex(θ0)
        θp = copy(θ0); θp[i] += h
        θm = copy(θ0); θm[i] -= h
        gfd[i] = (fg(θp)[1] - fg(θm)[1]) / (2h)
    end
    rel = maximum(abs.(g .- gfd)) / (maximum(abs.(gfd)) + 1e-30)
    return f0, g, gfd, rel
end

# Hand-rolled L-BFGS (m=7) with Armijo backtracking — Optim.jl is not a bench dependency.
function lbfgs(θ0, fg; m = 7, maxit = 200)
    θ = copy(θ0)
    f, g = fg(θ)
    f0 = f
    sList = Vector{Vector{Float64}}(); yList = Vector{Vector{Float64}}(); ρList = Float64[]
    niter = 0
    for it in 1:maxit
        sqrt(sum(abs2, g)) < 1e-6 && break
        # two-loop recursion -> search direction p = -H·g
        q = copy(g); αs = Float64[]
        for j in length(sList):-1:1
            a = ρList[j] * (sList[j]' * q); push!(αs, a); q -= a .* yList[j]
        end
        γ = isempty(sList) ? 1.0 : (sList[end]' * yList[end]) / (yList[end]' * yList[end])
        q *= γ
        for j in 1:length(sList)
            b = ρList[j] * (yList[j]' * q); q += (αs[length(sList) - j + 1] - b) .* sList[j]
        end
        p = -q
        (p' * g) ≥ 0 && (p = -g)                # not a descent dir -> steepest descent
        # Armijo backtracking
        t = 1.0; fnew, gnew, θnew, ok = f, g, θ, false
        for _ in 1:40
            θnew = θ + t .* p
            fnew, gnew = fg(θnew)
            if fnew ≤ f + 1e-4 * t * (p' * g)
                ok = true; break
            end
            t *= 0.5
        end
        ok || break
        s = θnew - θ; yv = gnew - g; sy = s' * yv
        if sy > 1e-12
            push!(sList, s); push!(yList, yv); push!(ρList, 1.0 / sy)
            if length(sList) > m
                popfirst!(sList); popfirst!(yList); popfirst!(ρList)
            end
        end
        θ, f, g = θnew, fnew, gnew
        niter = it
    end
    return θ, f, f0, sqrt(sum(abs2, g)), niter
end

if mode == "gradcheck"
    f0, g, gfd, rel = gradcheck(θ0, fg)
    npzwrite(out_npz, Dict("f" => f0, "g_analytic" => g, "g_fd" => gfd, "rel" => rel))
    @printf("gradcheck: f=%.6g  max|g-gfd|/|gfd|=%.3e\n", f0, rel)
else
    θ, f, f0, gnorm, niter = lbfgs(θ0, fg_safe)
    npzwrite(out_npz, Dict("theta_hat" => θ, "nlml" => f, "nlml0" => f0,
        "gnorm" => gnorm, "niter" => Float64(niter)))
    @printf("fit: nlml %.6g -> %.6g in %d iters; θ=[%.4f, %.4f, %.4f]\n",
        f0, f, niter, θ[1], θ[2], θ[3])
end
