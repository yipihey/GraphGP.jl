# Benchmark the Julia GraphGP kernels on the CUDA GPU backend (KernelAbstractions + CUDA.jl).
#
#   julia --project=bench test/bench_gpu.jl [N] [K] [D]
#
# Mirrors test/bench.jl but moves the per-kernel array fields onto the GPU so the
# KernelAbstractions backend dispatches to CUDA. Reports median wall time and throughput
# (refined points / second) for refine_logdet, refine_inv, and the logdet gradient.

using GraphGP
using KernelAbstractions
using CUDA
using Random
using Printf

function make_synthetic_gpu(; N, K, D, n0, T = Float32, seed = 1)
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
    # Move all kernel-launched arrays to the GPU; offsets stays on the host (dispatch only).
    return GraphGPProblem(CuArray(coords), CuArray(neighbors), [n0], n0,
        scale, CuArray(bins), CuArray(vals))
end

# Time a GPU op: synchronize the device around each call so we measure kernel time, not
# just launch latency.
function timeit_gpu(f; reps = 5)
    CUDA.@sync f()  # warmup / compile
    best = Inf
    for _ in 1:reps
        t = CUDA.@elapsed f()
        best = min(best, t)
    end
    return best
end

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
D = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
n0 = 1000

CUDA.versioninfo()
prob = make_synthetic_gpu(; N = N, K = K, D = D, n0 = n0)
M = GraphGP.nrefined(prob)
values = CUDA.rand(Float32, N)

@printf("\nGraphGP.jl GPU bench: N=%d M=%d K=%d D=%d device=%s\n",
    N, M, K, D, CUDA.name(CUDA.device()))

t_ld = timeit_gpu(() -> refine_logdet(prob))
@printf("  refine_logdet          : %8.2f ms   %7.1f M pts/s\n", 1e3 * t_ld, M / t_ld / 1e6)

t_inv = timeit_gpu(() -> refine_inv(prob, values))
@printf("  refine_inv             : %8.2f ms   %7.1f M pts/s\n", 1e3 * t_inv, M / t_inv / 1e6)

t_g = timeit_gpu(() -> refine_logdet_grad_vals(prob); reps = 3)
@printf("  refine_logdet_grad_vals: %8.2f ms   %7.1f M pts/s\n", 1e3 * t_g, M / t_g / 1e6)
