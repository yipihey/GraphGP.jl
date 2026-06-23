# Benchmark graph-construction stages — in particular the KernelAbstractions k-NN query
# (query_preceding_neighbors_ka) on CPU vs GPU, the dominant cost of building the graph.
#
#   julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_build.jl [N] [K] [D]
#
# The CPU k-d tree skeleton is built once (build_tree); the query is then timed on the CPU
# and GPU backends over the same tree. Reports wall time and throughput (queried points/s).

using GraphGP
using CUDA
using Random
using Printf

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
D = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
n0 = 1000

rng = MersenneTwister(1)
pts = randn(rng, N, D)

t_tree = @elapsed (sorted_pts, seg_lo, seg_hi, split_dim, perm) = build_tree(pts)
spts = permutedims(sorted_pts)              # (D, N) Float64
M = N - n0

function timeit(f; reps = 3)
    f()
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed f())
    end
    best
end

@printf("graph-build bench: N=%d M=%d K=%d D=%d\n", N, M, K, D)
@printf("  build_tree (CPU, 1 thread)     : %8.1f ms\n", 1e3 * t_tree)

t_cpu = timeit(() -> query_preceding_neighbors_ka(spts, seg_lo, seg_hi, split_dim, n0, K))
@printf("  query_ka  (CPU/KA)             : %8.1f ms   %7.1f M pts/s\n",
    1e3 * t_cpu, M / t_cpu / 1e6)

if CUDA.functional()
    spts_g = CuArray(spts)
    lo_g = CuArray(seg_lo); hi_g = CuArray(seg_hi); sd_g = CuArray(split_dim)
    qgpu() = query_preceding_neighbors_ka(spts_g, lo_g, hi_g, sd_g, n0, K)
    CUDA.@sync qgpu()
    best = Inf
    for _ in 1:3
        best = min(best, CUDA.@elapsed qgpu())
    end
    @printf("  query_ka  (GPU, %-18s): %8.1f ms   %7.1f M pts/s   (%.1fx vs CPU/KA)\n",
        CUDA.name(CUDA.device()), 1e3 * best, M / best / 1e6, t_cpu / best)
else
    println("  (no functional CUDA device; GPU query skipped)")
end
