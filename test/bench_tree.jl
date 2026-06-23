# Benchmark the GPU sort-based k-d tree build (build_tree_ka) and break down where its time
# goes. The build does, per level (≈ log2(N) levels): segment split-dim selection, a composite
# key build, a global sortperm, gathers (points/perm/node), and child assignment. The per-level
# global sort is expected to dominate (O(N log²N) total).
#
#   julia --project=julia/GraphGP/bench julia/GraphGP/test/bench_tree.jl [Nmax]

using GraphGP
using GraphGP: build_tree_ka
using CUDA
using Random
using Printf

CUDA.functional() || error("no functional CUDA device")

Nmax = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000_000
D = 3

function best_time(f; reps = 3)
    CUDA.@sync f()
    b = Inf
    for _ in 1:reps
        b = min(b, CUDA.@elapsed f())
    end
    b
end

println("=== build_tree_ka end-to-end (RTX A6000, D=$D) ===")
sizes = filter(N -> N <= Nmax, (200_000, 1_000_000, 5_000_000, 20_000_000))
for N in sizes
    pts = CuArray(randn(D, N))
    t = best_time(() -> build_tree_ka(pts))
    @printf("  N=%9d : %8.1f ms   %6.1f M pts/s\n", N, 1e3 * t, N / t / 1e6)
    pts = nothing
    CUDA.reclaim()
end

# Stage breakdown via isolated proxies for the two heavy per-level primitives (global sort and
# gather), summed over the ≈ log2(N) levels, compared with the end-to-end time.
N = min(1_000_000, Nmax)
Lmax = max(1, ceil(Int, log2(max(N, 2))))
pts = CuArray(randn(D, N))
key = CuArray(rand(Float64, N))
sp = CuArray(sortperm(key))
spts = CuArray(randn(D, N))

t_total = best_time(() -> build_tree_ka(pts))
t_sort = best_time(() -> (for _ in 1:Lmax; sortperm(key); end); reps = 2)
t_gather = best_time(() -> (for _ in 1:Lmax; spts[:, sp]; end); reps = 2)

@printf("\n=== stage breakdown at N=%d (%d levels) ===\n", N, Lmax)
@printf("  end-to-end build_tree_ka      : %8.1f ms\n", 1e3 * t_total)
@printf("  Σ sortperm over levels (proxy): %8.1f ms  (%.0f%%)\n", 1e3 * t_sort, 100 * t_sort / t_total)
@printf("  Σ point gather over levels    : %8.1f ms  (%.0f%%)\n", 1e3 * t_gather, 100 * t_gather / t_total)
@printf("  remainder (kernels/overhead)  : %8.1f ms  (%.0f%%)\n",
    1e3 * (t_total - t_sort - t_gather), 100 * (t_total - t_sort - t_gather) / t_total)
