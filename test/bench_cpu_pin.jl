# Compare CPU throughput under different ThreadPinning strategies.
#
#   julia -t <N> --project=bench test/bench_cpu_pin.jl <strategy> [Npts]
#
# <strategy> in: none | cores | compact | spread | numa | sockets
# This 2-socket EPYC 7763 has 128 cores / 16 NUMA nodes, so pinning (vs the OS scheduler
# migrating threads across NUMA domains) is expected to matter for the memory-bound path.

using GraphGP
using KernelAbstractions
using Random
using Printf
using ThreadPinning

strategy = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :none
N        = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4_000_000
K, D, n0 = 10, 3, 1000

if strategy != :none
    pinthreads(strategy)
end

function make_synthetic(; N, K, D, n0, T = Float32, seed = 1)
    rng = MersenneTwister(seed)
    coords = rand(rng, UInt32(0):UInt32(2^21 - 1), D, N)
    M = N - n0
    neighbors = Matrix{Int}(undef, K, M)
    @inbounds for m in 1:M
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

timeit(f; reps = 7) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

prob   = make_synthetic(; N, K, D, n0)
M      = GraphGP.nrefined(prob)
values = rand(Float32, N)

@printf("strategy=%-8s threads=%-3d N=%d M=%d\n", strategy, Threads.nthreads(), N, M)
t_ld  = timeit(() -> refine_logdet(prob))
t_inv = timeit(() -> refine_inv(prob, values))
t_g   = timeit(() -> refine_logdet_grad_vals(prob); reps = 4)
@printf("  refine_logdet           %8.2f ms  %7.1f M/s\n", 1e3t_ld,  M/t_ld/1e6)
@printf("  refine_inv              %8.2f ms  %7.1f M/s\n", 1e3t_inv, M/t_inv/1e6)
@printf("  refine_logdet_grad_vals %8.2f ms  %7.1f M/s\n", 1e3t_g,   M/t_g/1e6)
