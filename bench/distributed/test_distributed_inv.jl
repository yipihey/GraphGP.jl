# Phase 2 correctness: distributed inverse-quadratic loss gradient + refine_inv must reproduce
# the serial result and be rank-invariant.
#
#   mpiexec -n <R> julia --project=julia/GraphGP/bench/distributed test_distributed_inv.jl [N] [K] [cpu|gpu]
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
    CUDA.device!(MPI.Comm_rank(lc) % length(CUDA.devices()))
end

rng = MersenneTwister(2024)
points = randn(rng, N, D)
bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 1000; jitter = 1e-3)
prob = build_graph(points, n0, K, bins, vals)
data = randn(rng, N)
usegpu && (prob = to_backend(prob, CUDABackend()))

# Serial references (prob eltype is Float64 even on GPU; to_backend does not downcast).
loss_s, dv_s = generate_inv_loss_grad_vals(prob, usegpu ? CuArray(data) : data)
data_ord = prob.indices === nothing ? data : data[prob.indices]
xi_s = Array(refine_inv(prob, usegpu ? CuArray(data_ord) : data_ord))

# Distributed (pass raw data; the wrapper reorders + replicates).
dprob = distribute(prob, comm)
loss_d, dv_d = generate_inv_loss_grad_vals(dprob, data)
xi_d = refine_inv(dprob, data)

relerr(a, b) = abs(a - b) / max(abs(b), 1e-30)
normrel(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b))) / max(sqrt(sum(abs2, Float64.(b))), 1e-30)

if rank == 0
    e_loss = relerr(loss_d, Float64(loss_s))
    e_dv = normrel(dv_d, Array(dv_s))
    e_xi = normrel(xi_d, xi_s)
    @printf("[R=%d N=%d K=%d M=%d %s]\n", nranks, N, K, nrefined(prob), usegpu ? "gpu" : "cpu")
    @printf("  inv-loss: dist=%.10g serial=%.10g  relerr=%.2e\n", loss_d, Float64(loss_s), e_loss)
    @printf("  grad_vals norm-relerr=%.2e   xi norm-relerr=%.2e\n", e_dv, e_xi)
    tol = usegpu ? 1e-4 : 1e-9
    pass = e_loss < tol && e_dv < 1e-4 && e_xi < 1e-4
    println(pass ? "  PASS" : "  FAIL")
    @printf("LOSS64 %.15g\n", loss_d)
    exit(pass ? 0 : 1)
end
MPI.Finalize()
