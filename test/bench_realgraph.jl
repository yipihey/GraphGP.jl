# Benchmark GraphGP.jl on the GPU using a REAL gp.build_graph-derived graph dumped by
# bench_realgraph.py (so all implementations share identical spatially-coherent neighbors).
#
#   julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_realgraph.jl <dump_path.npz>

using GraphGP
using KernelAbstractions
using CUDA
using NPZ
using Printf

dump_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: bench_realgraph.jl <dump.npz>")
d = npzread(dump_path)

# Match the conversion in loadref.jl: (N,d)->(d,N), neighbors 0-based ->1-based, to GPU.
coords    = CuArray(permutedims(UInt32.(d["coords"])))
neighbors = CuArray(permutedims(Int.(d["neighbors"])) .+ 1)
offsets   = Int.(d["offsets"])
n0        = Int(d["n0"])
scale     = Float32(d["scale"])
bins      = CuArray(Float32.(d["cov_bins32"]))
vals      = CuArray(Float32.(d["cov_vals32"]))
values    = CuArray(Float32.(d["values32"]))

prob = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals)
M = GraphGP.nrefined(prob)
N = GraphGP.npoints(prob)
K = GraphGP.nneighbors(prob)
D = GraphGP.ndims_space(prob)

function timeit_gpu(f; reps = 5)
    CUDA.@sync f()
    best = Inf
    for _ in 1:reps
        t = CUDA.@elapsed f()
        best = min(best, t)
    end
    return best
end

@printf("REAL-graph bench (GraphGP.jl): N=%d M=%d K=%d D=%d device=%s\n",
    N, M, K, D, CUDA.name(CUDA.device()))

t_ld = timeit_gpu(() -> refine_logdet(prob))
@printf("  [GraphGP.jl] refine_logdet          : %8.2f ms   %7.1f M pts/s\n", 1e3 * t_ld, M / t_ld / 1e6)

t_inv = timeit_gpu(() -> refine_inv(prob, values))
@printf("  [GraphGP.jl] refine_inv             : %8.2f ms   %7.1f M pts/s\n", 1e3 * t_inv, M / t_inv / 1e6)

t_g = timeit_gpu(() -> refine_logdet_grad_vals(prob); reps = 3)
@printf("  [GraphGP.jl] refine_logdet_grad_vals: %8.2f ms   %7.1f M pts/s\n", 1e3 * t_g, M / t_g / 1e6)
