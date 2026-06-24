# Exact Vecchia GP posterior by Matheron's rule, in ONE Julia process — CPU or GPU. The joint Vecchia
# graph is built over Xall = [x_pred; x_data]; everything is matrix-free via the two operators GraphGP
# exposes:
#   generate(prob, ξ)          = G ξ           (G = scatter∘L; ξ iid N(0,I) → prior draw, cov = K)
#   generate_grad_xi(prob, u)  = Gᵀ u          (the adjoint / VJP of generate)
# so the full-space covariance apply in ORIGINAL order is  K u = generate(prob, generate_grad_xi(u)).
#
# Matheron posterior:  mean_* = K_{*D}(K_DD+N)⁻¹ y_D;  f^post_* = f^prior_* + K_{*D}(K_DD+N)⁻¹(y+ε−f_D).
# K_DD·v / K_{*D}·w are sub-blocks via embed/extract by the data mask; (K_DD+N) is solved by CG. On
# GPU the graph is built with build_graph_ka and the CG vectors / embed-extract / dot products all
# live on the device (CuArray), so only the host control flow crosses back.
#
#   julia -t N --project=... run_posterior.jl <in.npz> <out.npz> [cpu|gpu]
#
# in.npz: either a prebuilt graph (coords/neighbors/offsets/scale/indices) OR build_points (N,D)+k for
#         build-in-Julia; plus cov_bins32/cov_vals32, data_mask (N,) {0,1}, y_data (nd,), noise_var
#         (nd,), n0, n_samples, seed, jitter, cg_tol, cg_maxiter. out.npz: post_mean (ns,),
#         post_samples (n_samples, ns) over the prediction points, input order.

using GraphGP, KernelAbstractions, NPZ, Printf, Random, LinearAlgebra

in_npz, out_npz = ARGS[1], ARGS[2]
backend_arg = length(ARGS) >= 3 ? ARGS[3] : "cpu"
@assert backend_arg in ("cpu", "gpu")
usegpu = backend_arg == "gpu"
usegpu && (using CUDA)
backend = usegpu ? CUDABackend() : CPU()
dev(x) = usegpu ? CuArray(x) : x

d = npzread(in_npz)
n0 = Int(d["n0"])
bins = Float64.(d["cov_bins32"]); vals = Float64.(d["cov_vals32"])
mask = Int.(vec(d["data_mask"]))                       # 1 = data, 0 = prediction (input order)
y_data_h = Float64.(vec(d["y_data"]))
noise_h = Float64.(vec(d["noise_var"]))
n_samples = Int(d["n_samples"]); seed = Int(d["seed"])
jitter = Float64(d["jitter"]); cg_tol = Float64(d["cg_tol"]); cg_maxiter = Int(d["cg_maxiter"])

prob = if haskey(d, "build_points")
    # Build the joint graph on the backend (GPU): tree → neighbors → depth → quantize.
    pts = Float64.(d["build_points"])
    kbuild = Int(d["k"][])
    build_graph_ka(dev(pts), n0, kbuild, dev(bins), dev(vals); backend = backend)
else
    coords = permutedims(UInt32.(d["coords"]))
    neighbors = permutedims(Int.(d["neighbors"])) .+ 1
    offsets = Int.(d["offsets"]); scale = Float64(d["scale"])
    indices = haskey(d, "indices") ? (Int.(vec(d["indices"])) .+ 1) : nothing
    pr = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals, indices)
    usegpu ? to_backend(pr, CUDABackend()) : pr
end
N = npoints(prob)
data_idx = findall(==(1), mask); pred_idx = findall(==(0), mask)
nd = length(data_idx); ns = length(pred_idx)
@assert nd == length(y_data_h) == length(noise_h)

# device-resident gather/scatter indices + data
data_idxb = dev(data_idx); pred_idxb = dev(pred_idx)
y_data = dev(y_data_h); noise = dev(noise_h)
zeros_N() = KernelAbstractions.zeros(backend, Float64, N)

# Full-space covariance apply in original order: K u = generate(prob, generate_grad_xi(prob, u)).
Kapply(u) = generate(prob, generate_grad_xi(prob, u))

# (K_DD + N) · v on the data subspace (embed → full K-apply → extract data rows + noise·v).
function KDD_plus_N(v)
    u = zeros_N(); u[data_idxb] = v
    return Kapply(u)[data_idxb] .+ (noise .+ jitter) .* v
end

# K_{*D} · w : cross block (embed data vector w → full K-apply → extract prediction rows).
function KsD(w)
    u = zeros_N(); u[data_idxb] = w
    return Kapply(u)[pred_idxb]
end

# Conjugate gradient for the SPD system (K_DD + N) x = b (vectors on the device).
function cg(b; tol, maxiter)
    x = KernelAbstractions.zeros(backend, Float64, length(b))
    r = b .- KDD_plus_N(x)
    p = copy(r); rs = dot(r, r)
    bnorm = sqrt(dot(b, b)) + 1e-300
    for _ in 1:maxiter
        Ap = KDD_plus_N(p)
        α = rs / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        rs_new = dot(r, r)
        sqrt(rs_new) / bnorm < tol && break
        p = r .+ (rs_new / rs) .* p
        rs = rs_new
    end
    return x
end

# Posterior mean (one solve).
post_mean = Array(KsD(cg(y_data; tol = cg_tol, maxiter = cg_maxiter)))

# Posterior samples (Matheron): fresh prior draw + noise per sample, shared CG operator.
rng = MersenneTwister(seed)
post_samples = Array{Float64}(undef, n_samples, ns)
for s in 1:n_samples
    f_prior = generate(prob, dev(randn(rng, N)))             # G ξ, original order, cov = K
    ε = dev(sqrt.(noise_h) .* randn(rng, nd))
    r = y_data .+ ε .- f_prior[data_idxb]
    w = cg(r; tol = cg_tol, maxiter = cg_maxiter)
    post_samples[s, :] = Array(f_prior[pred_idxb] .+ KsD(w))
end

npzwrite(out_npz, Dict("post_mean" => post_mean, "post_samples" => post_samples))
@printf("posterior: ns=%d nd=%d n_samples=%d dev=%s (CG tol=%.1e)\n", ns, nd, n_samples, backend_arg, cg_tol)
