# Run one graph (from dump_graph.py) through GraphGP.jl on CPU or GPU and emit its outputs +
# timings, mirroring run_jax.py.
#
#   julia -t <N> --project=bench run_julia.jl <graph.npz> <correctness|timing> <cpu|gpu> <outdir>
#
# correctness mode runs Float64 and writes <outdir>/julia-<dev>.npz (logdet/xi/grad) for the
# element-wise cross-check; timing mode runs Float32 and prints one JSON line of milliseconds.

using GraphGP, KernelAbstractions, NPZ, Printf

npz_path, mode, backend_arg, outdir = ARGS[1], ARGS[2], ARGS[3], ARGS[4]
@assert mode in ("correctness", "timing")
@assert backend_arg in ("cpu", "gpu")

usegpu = backend_arg == "gpu"
if usegpu
    using CUDA
end
T = mode == "correctness" ? Float64 : Float32

d = npzread(npz_path)
coords = permutedims(UInt32.(d["coords"]))          # (N,d) -> (d,N)
neighbors = permutedims(Int.(d["neighbors"])) .+ 1  # (M,k) 0-based -> (k,M) 1-based
offsets = Int.(d["offsets"])
n0 = Int(d["n0"])
scale = T(d["scale"])
bins = T.(d["cov_bins32"])
vals = T.(d["cov_vals32"])
values = T.(d["values32"])

prob = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals)
dev = "cpu"
if usegpu
    prob = GraphGPProblem(CuArray(coords), CuArray(neighbors), offsets, n0, scale,
        CuArray(bins), CuArray(vals))
    values = CuArray(values)
    dev = "gpu"
end
label = "julia-$(dev)"
sync() = usegpu ? CUDA.synchronize() : nothing

if mode == "correctness"
    logdet = Float64(refine_logdet(prob))
    xi = Float64.(Array(refine_inv(prob, values)))
    grad = Float64.(Array(refine_logdet_grad_vals(prob)))
    npzwrite(joinpath(outdir, "$(label).npz"), Dict("logdet" => logdet, "xi" => xi,
        "grad_logdet_vals" => grad))
    @printf("%s: correctness outputs written (logdet=%.10g)\n", label, logdet)
else
    function timeit(f; reps = 5)
        f(); sync()
        best = Inf
        for _ in 1:reps
            t0 = time_ns(); f(); sync()
            best = min(best, (time_ns() - t0) / 1e9)
        end
        best
    end
    M = GraphGP.nrefined(prob)
    ld_ms = 1e3 * timeit(() -> refine_logdet(prob))
    iv_ms = 1e3 * timeit(() -> refine_inv(prob, values))
    g_ms = 1e3 * timeit(() -> refine_logdet_grad_vals(prob); reps = 3)
    @printf("TIMING {\"label\":\"%s\",\"M\":%d,\"logdet_ms\":%.4f,\"inv_ms\":%.4f,\"grad_ms\":%.4f}\n",
        label, M, ld_ms, iv_ms, g_ms)
end
