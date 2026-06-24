# Phase 4 correctness:
#   (A) distributed_build_graph (replicated tree + partitioned query) then distributed
#       generate_logdet must equal the serial build_graph_ka logdet — the Vecchia logdet is
#       invariant to the depth reorder, so a tree-order distributed build fits identically.
#   (B) distributed_quantize over spatially-partitioned points must match the serial global
#       lattice (origin/scale).
#
#   mpiexec -n <R> julia --project=bench/distributed test_distributed_build.jl [N] [K]
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

rng = MersenneTwister(31415)
points = randn(rng, N, D)
bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 1000; jitter = 1e-3)

# (A) Distributed build (tree order, partitioned query) → distributed logdet.
dprob = distributed_build_graph(points, comm, n0, K, bins, vals)
ld_dist = generate_logdet(dprob)

# Serial reference: same tree algorithm (build_graph_ka), depth-ordered.
prob_serial = build_graph_ka(points, n0, K, bins, vals)
ld_serial = sum(Float64, refine_logdet_terms(prob_serial)) +
    Float64(generate_dense_logdet(view(prob_serial.coords, :, 1:prob_serial.n0),
        prob_serial.scale, prob_serial.bins, prob_serial.vals, prob_serial.n0))

# (B) Distributed quantize over a spatial row-partition vs serial global quantize.
counts = [div(N, nranks) + (r < mod(N, nranks) ? 1 : 0) for r in 0:(nranks - 1)]
starts = cumsum([1; counts])[1:nranks]
my_rows = starts[rank + 1]:(starts[rank + 1] + counts[rank + 1] - 1)
_, origin_d, scale_d = distributed_quantize(points[my_rows, :], comm)
_, origin_s, scale_s = quantize_to_lattice(points)

relerr(a, b) = abs(a - b) / max(abs(b), 1e-30)

if rank == 0
    e_ld = relerr(ld_dist, ld_serial)
    e_scale = relerr(scale_d, scale_s)
    e_origin = maximum(abs.(origin_d .- origin_s))
    @printf("[R=%d N=%d K=%d M=%d]\n", nranks, N, K, nrefined(prob_serial))
    @printf("  (A) build+fit logdet: dist=%.10g serial=%.10g  relerr=%.2e\n",
        ld_dist, ld_serial, e_ld)
    @printf("  (B) quantize: scale relerr=%.2e  origin max-abs-diff=%.2e\n", e_scale, e_origin)
    pass = e_ld < 1e-9 && e_scale < 1e-12 && e_origin < 1e-9
    println(pass ? "  PASS" : "  FAIL")
    @printf("LOGDET64 %.15g\n", ld_dist)
    exit(pass ? 0 : 1)
end
MPI.Finalize()
