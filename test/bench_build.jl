# Benchmark graph construction: the CPU k-d tree skeleton (build_tree) and the GPU
# KernelAbstractions k-NN query (query_preceding_neighbors_ka) — the dominant compute of the
# build. The tree is built once on the CPU, moved to the device, and the query timed on the GPU.
#
#   julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_build.jl [N] [K] [D]
#
# NOTE: the KA query is GPU-oriented. On the KA *CPU* backend it is slower than the scalar
# `query_preceding_neighbors` used by `build_graph`, so it is not timed here. The GPU query
# pruning (static full-segment AABB) is correct but not yet optimal; a tighter split-plane
# bound is a planned optimization.

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

@printf("graph-build bench: N=%d M=%d K=%d D=%d\n", N, M, K, D)
@printf("  build_tree (CPU, 1 thread) : %8.1f ms\n", 1e3 * t_tree)

function bench_gpu_query(spts, seg_lo, seg_hi, split_dim, n0, K, M)
    spts_g = CuArray(spts)
    lo_g = CuArray(seg_lo); hi_g = CuArray(seg_hi); sd_g = CuArray(split_dim)
    qgpu() = query_preceding_neighbors_ka(spts_g, lo_g, hi_g, sd_g, n0, K)
    CUDA.@sync qgpu()                       # warmup / compile
    best = Inf
    for _ in 1:3
        best = min(best, CUDA.@elapsed qgpu())
    end
    @printf("  query_ka   (GPU, %-16s) : %8.1f ms   %7.2f M pts/s\n",
        CUDA.name(CUDA.device()), 1e3 * best, M / best / 1e6)
end

if CUDA.functional()
    bench_gpu_query(spts, seg_lo, seg_hi, split_dim, n0, K, M)
else
    println("  (no functional CUDA device; GPU query skipped)")
end
