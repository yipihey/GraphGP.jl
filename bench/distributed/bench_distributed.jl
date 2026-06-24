# Phase 1 scaling benchmark: time the distributed fused logdet + vals gradient (the fitting
# inner loop) across ranks.
#
#   mpiexec -n <R> julia --project=julia/GraphGP/bench/distributed bench_distributed.jl [N] [K] [cpu|gpu]
#
# Each rank holds the full graph (replicated) and computes its column slice, so fixing N and
# growing R is a STRONG-scaling test: per-iteration time should fall ~linearly until the
# (tiny: scalar + nbins) Allreduce latency dominates.
using MPI
using GraphGP
using Random
using Printf

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
usegpu = length(ARGS) >= 3 && ARGS[3] == "gpu"
D, n0 = 3, 1000

if usegpu
    using CUDA
    lc = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    CUDA.device!(MPI.Comm_rank(lc) % length(CUDA.devices()))
end

rng = MersenneTwister(12345)
points = randn(rng, N, D)
bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 1000; jitter = 1e-3)
prob = build_graph(points, n0, K, bins, vals)
usegpu && (prob = to_backend(prob, CUDABackend()))
dprob = distribute(prob, comm)
M = nrefined(prob)

sync() = usegpu ? CUDA.synchronize() : nothing
function timeit(f; reps = 7)
    f(); sync(); MPI.Barrier(comm)
    best = Inf
    for _ in 1:reps
        MPI.Barrier(comm)
        t0 = time_ns()
        f(); sync()
        MPI.Barrier(comm)
        best = min(best, (time_ns() - t0) / 1e9)
    end
    best
end

t = timeit(() -> generate_logdet_and_grad_vals(dprob))
if rank == 0
    @printf("[R=%-3d N=%d M=%d K=%d %s]  logdet+grad: %8.2f ms   %7.1f M pts/s\n",
        nranks, N, M, K, usegpu ? "gpu" : "cpu", 1e3t, M / t / 1e6)
end
MPI.Finalize()
