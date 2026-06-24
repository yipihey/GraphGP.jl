# Phase 1 correctness: distributed log-likelihood + vals gradient must reproduce the serial
# result, and be invariant to the rank count.
#
#   mpiexec -n <R> julia --project=bench/distributed test_distributed.jl [N] [K]
#
# Every rank builds the SAME graph (fixed seed) so each holds the full problem; `distribute`
# then gives each rank a column slice. We compare the distributed generate_logdet /
# generate_logdet_grad_vals against the serial values computed from the full problem.
using MPI
using GraphGP
using Random
using Printf

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
usegpu = length(ARGS) >= 3 && ARGS[3] == "gpu"
D, n0 = 3, 1000

if usegpu
    using CUDA
    lc = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    CUDA.device!(MPI.Comm_rank(lc) % length(CUDA.devices()))   # one rank per GPU
end

# Identical graph on every rank (deterministic).
rng = MersenneTwister(12345)
points = randn(rng, N, D)
variance, scale = 1.0, 0.3
bins, vals = rbf_kernel(variance, scale, 1e-4, 1e1, 1000; jitter = 1e-3)
prob = build_graph(points, n0, K, bins, vals)            # Float64 problem (CPU backend)
usegpu && (prob = to_backend(prob, CUDABackend()))       # same problem, on the device

# Serial references (computed identically on every rank from the full problem).
ld_serial_f32 = generate_logdet(prob)                                    # f32-summation path
terms = refine_logdet_terms(prob)
ld_serial_f64 = sum(Float64, terms) +
    Float64(generate_dense_logdet(view(prob.coords, :, 1:n0), prob.scale, prob.bins, prob.vals, n0))
dv_serial = generate_logdet_grad_vals(prob)                              # Vector (length nbins)

# Distributed evaluation.
dprob = distribute(prob, comm)
ld_dist = generate_logdet(dprob)                                         # Float64, allreduced
dv_dist = generate_logdet_grad_vals(dprob)                               # Vector{Float64}, allreduced

# Compare on rank 0.
relerr(a, b) = abs(a - b) / max(abs(b), 1e-30)
normrel(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), 1e-30)

if rank == 0
    e_f64 = relerr(ld_dist, ld_serial_f64)
    e_f32 = relerr(ld_dist, Float64(ld_serial_f32))
    e_dv = normrel(dv_dist, Float64.(Array(dv_serial)))
    @printf("[R=%d N=%d K=%d M=%d]\n", nranks, N, K, nrefined(prob))
    @printf("  logdet: dist=%.10g  serial_f64=%.10g  serial_f32=%.10g\n",
        ld_dist, ld_serial_f64, Float64(ld_serial_f32))
    @printf("  logdet relerr vs f64=%.2e  vs f32-path=%.2e\n", e_f64, e_f32)
    @printf("  grad_vals norm-relerr vs serial=%.2e\n", e_dv)
    pass = e_f64 < 1e-9 && e_f32 < 1e-4 && e_dv < 1e-4
    println(pass ? "  PASS" : "  FAIL")
    # Emit the f64 logdet for cross-rank-count invariance checking by the runner.
    @printf("LOGDET64 %.15g\n", ld_dist)
    exit(pass ? 0 : 1)
end
MPI.Finalize()
