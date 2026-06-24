# Exact Vecchia GP posterior by Matheron's rule, run in ONE Julia process (the matrix-free CG never
# crosses the subprocess boundary). The joint Vecchia graph is built over Xall = [x_pred; x_data];
# everything is matrix-free via the two operators GraphGP.jl exposes:
#   generate(prob, ξ)          = G ξ           (G = scatter∘L; ξ iid N(0,I) → prior draw, cov = K)
#   generate_grad_xi(prob, u)  = Gᵀ u          (the adjoint / VJP of generate)
# so the full-space covariance apply in ORIGINAL order is
#   K u = generate(prob, generate_grad_xi(prob, u))          (= G Gᵀ u = L Lᵀ permuted).
#
# Matheron posterior at the prediction points:
#   mean_*    = K_{*D} (K_DD + N)⁻¹ y_D
#   f^post_*  = f^prior_* + K_{*D} (K_DD + N)⁻¹ (y_D + ε_D − f^prior_D),  (f^prior,ε) drawn fresh.
# K_DD·v and K_{*D}·w are sub-blocks of the full K-apply via an embed/extract by the data mask.
# (K_DD + N) is solved by conjugate gradient; one CG operator is reused across mean + all samples.
#
#   julia -t N --project=bench run_posterior.jl <in.npz> <out.npz>
#
# in.npz : the joint graph (coords/neighbors/offsets/n0/scale/cov_bins32/cov_vals32/indices over
#          Xall, original = input order [pred; data]), data_mask (N,) {0,1}, y_data (nd,),
#          noise_var (nd,), n_samples, seed, jitter, cg_tol, cg_maxiter.
# out.npz: post_mean (ns,), post_samples (n_samples, ns)  — over the prediction points, input order.

using GraphGP, KernelAbstractions, NPZ, Printf, Random

in_npz, out_npz = ARGS[1], ARGS[2]

d = npzread(in_npz)
coords = permutedims(UInt32.(d["coords"]))
neighbors = permutedims(Int.(d["neighbors"])) .+ 1
offsets = Int.(d["offsets"])
n0 = Int(d["n0"]); scale = Float64(d["scale"])
bins = Float64.(d["cov_bins32"]); vals = Float64.(d["cov_vals32"])
indices = haskey(d, "indices") ? (Int.(vec(d["indices"])) .+ 1) : nothing
mask = Int.(vec(d["data_mask"]))                       # 1 = data, 0 = prediction (input order)
y_data = Float64.(vec(d["y_data"]))
noise = Float64.(vec(d["noise_var"]))
n_samples = Int(d["n_samples"]); seed = Int(d["seed"])
jitter = Float64(d["jitter"]); cg_tol = Float64(d["cg_tol"]); cg_maxiter = Int(d["cg_maxiter"])

prob = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals, indices)
N = npoints(prob)
data_idx = findall(==(1), mask)
pred_idx = findall(==(0), mask)
nd = length(data_idx); ns = length(pred_idx)
@assert nd == length(y_data) == length(noise)

# Full-space covariance apply in original order: K u = generate(prob, generate_grad_xi(prob, u)).
Kapply(u) = generate(prob, generate_grad_xi(prob, u))

# (K_DD + N) · v on the data subspace (embed → full K-apply → extract data rows + noise·v).
function KDD_plus_N(v)
    u = zeros(Float64, N)
    @inbounds u[data_idx] .= v
    Ku = Kapply(u)
    return Ku[data_idx] .+ (noise .+ jitter) .* v
end

# K_{*D} · w : cross block (embed data vector w → full K-apply → extract prediction rows).
function KsD(w)
    u = zeros(Float64, N)
    @inbounds u[data_idx] .= w
    return Kapply(u)[pred_idx]
end

# Conjugate gradient for the SPD system (K_DD + N) x = b.
function cg(b; tol, maxiter)
    x = zeros(Float64, length(b))
    r = b - KDD_plus_N(x)
    p = copy(r); rs = r' * r
    bnorm = sqrt(b' * b) + 1e-300
    for _ in 1:maxiter
        Ap = KDD_plus_N(p)
        α = rs / (p' * Ap)
        x .+= α .* p
        r .-= α .* Ap
        rs_new = r' * r
        sqrt(rs_new) / bnorm < tol && break
        p = r .+ (rs_new / rs) .* p
        rs = rs_new
    end
    return x
end

# Posterior mean (one solve).
w_mean = cg(y_data; tol = cg_tol, maxiter = cg_maxiter)
post_mean = KsD(w_mean)

# Posterior samples (Matheron): fresh prior draw + noise per sample, shared CG operator.
rng = MersenneTwister(seed)
post_samples = Array{Float64}(undef, n_samples, ns)
for s in 1:n_samples
    f_prior = generate(prob, randn(rng, N))                  # G ξ, original order, cov = K
    ε = sqrt.(noise) .* randn(rng, nd)
    r = y_data .+ ε .- f_prior[data_idx]
    w = cg(r; tol = cg_tol, maxiter = cg_maxiter)
    post_samples[s, :] = f_prior[pred_idx] .+ KsD(w)
end

npzwrite(out_npz, Dict("post_mean" => post_mean, "post_samples" => post_samples))
@printf("posterior: ns=%d nd=%d n_samples=%d (CG tol=%.1e)\n", ns, nd, n_samples, cg_tol)
