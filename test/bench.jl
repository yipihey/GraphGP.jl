# Benchmark the Julia GraphGP kernels (CPU backend; multithread with `julia -t auto`).
#
#   julia -t auto --project=julia/GraphGP julia/GraphGP/test/bench.jl [N] [K] [D]
#
# Reports median wall time and throughput (refined points / second) for refine_logdet,
# refine_inv, and the logdet gradient w.r.t. cov_vals.
#
# NUMA note: on multi-socket / many-NUMA-node hosts, pin the worker threads round-robin
# across NUMA nodes before benchmarking — `using ThreadPinning; pinthreads(:numa)`. On a
# 2-socket EPYC 7763 (16 NUMA nodes) this gave +35% (64 thr) to +59% (128 thr) on the
# gradient path; the bandwidth-bound forward path gains less (~+25%). See bench_cpu_pin.jl.

using GraphGP
using KernelAbstractions
using Random
using Printf

function make_synthetic(; N, K, D, n0, T = Float32, seed = 1)
    rng = MersenneTwister(seed)
    coords = rand(rng, UInt32(0):UInt32(2^21 - 1), D, N)
    M = N - n0
    neighbors = Matrix{Int}(undef, K, M)
    @inbounds for m in 1:M
        # K distinct preceding indices (cheap topological-order proxy for benchmarking).
        hi = n0 + m - 1
        for j in 1:K
            neighbors[j, m] = rand(rng, 1:hi)
        end
    end
    scale = T(3e-6)
    bins = collect(T, range(0, 10; length = 1000)); bins[1] = 0
    vals = T.(exp.(-(Float64.(bins) ./ 0.3) .^ 2)); vals[1] *= (1 + T(1e-3))
    return GraphGPProblem(coords, neighbors, [n0], n0, scale, bins, vals)
end

function timeit(f; reps = 5)
    f()  # warmup / compile
    best = Inf
    for _ in 1:reps
        t = @elapsed f()
        best = min(best, t)
    end
    return best
end

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
D = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
n0 = 1000

prob = make_synthetic(; N = N, K = K, D = D, n0 = n0)
M = GraphGP.nrefined(prob)
values = rand(Float32, N)

@printf("GraphGP.jl CPU bench: N=%d M=%d K=%d D=%d threads=%d\n", N, M, K, D, Threads.nthreads())

t_ld = timeit(() -> refine_logdet(prob))
@printf("  refine_logdet          : %8.2f ms   %6.1f M pts/s\n", 1e3 * t_ld, M / t_ld / 1e6)

t_inv = timeit(() -> refine_inv(prob, values))
@printf("  refine_inv             : %8.2f ms   %6.1f M pts/s\n", 1e3 * t_inv, M / t_inv / 1e6)

t_g = timeit(() -> refine_logdet_grad_vals(prob); reps = 3)
@printf("  refine_logdet_grad_vals: %8.2f ms   %6.1f M pts/s\n", 1e3 * t_g, M / t_g / 1e6)
