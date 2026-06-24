# Phase 3 correctness:
#   3b — distributed single-field `generate` must reproduce the serial field, rank-invariant.
#   3a — independent realizations: each rank draws its own field; they must be distinct and each
#        must equal the serial draw with the same seed.
#
#   mpiexec -n <R> julia --project=julia/GraphGP/bench/distributed test_distributed_gen.jl [N] [K]
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
D, n0 = 3, 1000

rng = MersenneTwister(99)
points = randn(rng, N, D)
bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 1000; jitter = 1e-3)
prob = build_graph(points, n0, K, bins, vals)
dprob = distribute(prob, comm)

normrel(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b))) / max(sqrt(sum(abs2, Float64.(b))), 1e-30)

# --- 3b: distributed single field vs serial ---
xi = randn(MersenneTwister(7), N)                 # tree-order unit normals (replicated)
field_serial = generate(prob, xi)
field_dist = generate(dprob, xi)
e_gen = normrel(field_dist, field_serial)

# --- 3a: independent realizations (each rank draws its own; uses the full replicated graph) ---
field_r = generate(prob, randn(MersenneTwister(1000 + rank), N))
chk_r = sum(field_r)
# Gather each rank's checksum + recompute rank r's draw on rank 0 to confirm correctness.
chks = MPI.Gather(chk_r, 0, comm)

if rank == 0
    @printf("[R=%d N=%d K=%d M=%d]\n", nranks, N, K, nrefined(prob))
    @printf("  3b distributed generate norm-relerr vs serial = %.2e\n", e_gen)
    # realizations: distinct + reproducible
    distinct = length(unique(round.(chks; digits = 6))) == nranks
    repro = all(0:(nranks - 1)) do r
        isapprox(sum(generate(prob, randn(MersenneTwister(1000 + r), N))), chks[r + 1])
    end
    @printf("  3a realizations: %d fields, distinct=%s reproducible=%s\n", nranks, distinct, repro)
    pass = e_gen < 1e-9 && distinct && repro
    println(pass ? "  PASS" : "  FAIL")
    @printf("FIELDSUM %.12g\n", sum(field_dist))
    exit(pass ? 0 : 1)
end
MPI.Finalize()
