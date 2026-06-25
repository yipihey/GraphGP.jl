# Time GraphGP.jl `build_graph` (parallel CPU). Mirrors build_bench.py.
#
#   julia -t <N> --project=bench bench/compare/build_bench.jl [N] [K] [D]
#
# Prints one `BUILD {...}` JSON line. run_build_compare.sh runs this and build_bench.py under
# matched cores for the head-to-head.

using GraphGP
using Random
using Printf

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
D = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3

bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 200; jitter = 1e-2)
build_graph(randn(500, D), 50, 8, bins, vals)            # warm (compile)
P = randn(MersenneTwister(1), N, D)
t = @elapsed build_graph(P, 256, K, bins, vals)

@printf("BUILD {\"impl\":\"graphgp.jl-cpu\",\"N\":%d,\"K\":%d,\"D\":%d,\"threads\":%d,\"seconds\":%.3f}\n",
    N, K, D, Threads.nthreads(), t)
