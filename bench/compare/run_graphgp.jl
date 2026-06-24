# ECHOES bridge driver: run one graph (dumped by graphgp_julia.py / dump_graph_echoes.py) through
# GraphGP.jl and write the requested ops, in ORIGINAL point order, matching the Python graphgp
# `generate`/`generate_inv` semantics so the field is NOT scrambled.
#
#   julia -t <N> --project=julia/GraphGP/bench run_graphgp.jl <in.npz> <out.npz> <cpu|gpu> <ops> <f32|f64>
#
# <ops> is a comma list of: generate,generate_inv,logdet,grad  (grad = generate_logdet_grad_vals)
#
# INDEX CONVENTION: as of the generate-ordering-dropin fix, GraphGP.jl `generate`/`generate_inv`
# are ORIGINAL-order in AND out (they gather xi->tree on input and scatter the result->ORIGINAL
# internally via `prob.indices`), exactly matching Python `graphgp`. So this driver does NO manual
# reordering — it passes the original-order xi/values straight through. `in.npz` always provides
# xi/values in ORIGINAL order; `out.npz` is always ORIGINAL order.

using GraphGP, KernelAbstractions, NPZ, Printf

in_npz, out_npz, backend_arg, ops_arg, dtype_arg = ARGS[1], ARGS[2], ARGS[3], ARGS[4], ARGS[5]
@assert backend_arg in ("cpu", "gpu")
usegpu = backend_arg == "gpu"
usegpu && (using CUDA)
T = dtype_arg == "f64" ? Float64 : Float32
ops = split(ops_arg, ",")

d = npzread(in_npz)
coords = permutedims(UInt32.(d["coords"]))            # (N,D) -> (D,N)
neighbors = permutedims(Int.(d["neighbors"])) .+ 1    # (M,K) 0-based -> (K,M) 1-based
offsets = Int.(d["offsets"])
n0 = Int(d["n0"]); scale = T(d["scale"])
bins = T.(d["cov_bins32"]); vals = T.(d["cov_vals32"])
indices = haskey(d, "indices") ? (Int.(vec(d["indices"])) .+ 1) : nothing   # 0-based -> 1-based

# Anisotropic kernel K(Δspatial, Δz): the dumped grid already carries the jitter (applied in
# Python build_anisotropic_covariance), so construct with jitter=0. NPZ preserves the (n_s,n_z)
# logical shape; parity is checked against the fork's aniso.py.
prob = if haskey(d, "aniso_grid")
    sb = T.(vec(d["aniso_spatial_bins"]))
    zb = T.(vec(d["aniso_z_bins"]))
    grid = T.(d["aniso_grid"])                         # (n_s, n_z)
    alpha = T(d["aniso_alpha"][])
    cov = build_anisotropic_covariance(sb, zb, grid, alpha; jitter = 0)
    GraphGPProblem(coords, neighbors, offsets, n0, scale, cov, indices)
else
    GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals, indices)
end
if usegpu
    prob = to_backend(prob, CUDABackend())
end

results = Dict{String,Any}()

if "generate" in ops
    xi = T.(d["xi"])                                  # (N,) or (N, n_samples), ORIGINAL order
    xi = ndims(xi) == 1 ? reshape(xi, :, 1) : xi
    N, S = size(xi)
    outv = Array{Float64}(undef, S, N)                # (n_samples, N) ORIGINAL order
    for s in 1:S
        xis = xi[:, s]                                # original order; generate gathers internally
        usegpu && (xis = CuArray(xis))
        outv[s, :] = Float64.(Array(generate(prob, xis)))
    end
    results["generate"] = outv
end

if "generate_inv" in ops
    values = T.(vec(d["values"]))                     # ORIGINAL order
    usegpu && (values = CuArray(values))
    results["generate_inv"] = Float64.(Array(generate_inv(prob, values)))   # ORIGINAL order out
end

if "logdet" in ops
    results["logdet"] = Float64(generate_logdet(prob))
end

if "grad" in ops
    results["grad_logdet_vals"] = Float64.(Array(generate_logdet_grad_vals(prob)))
end

npzwrite(out_npz, results)
@printf("run_graphgp: wrote %s [%s] dev=%s\n", join(keys(results), ","), dtype_arg, backend_arg)
