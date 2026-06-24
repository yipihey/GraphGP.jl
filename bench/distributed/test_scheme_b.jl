# Scheme B (partitioned coords + halo) correctness: the distributed log-likelihood, its vals
# gradient, and the inverse-quadratic-loss gradient must reproduce the serial result and be
# invariant to the rank count — while each rank holds only its OWNED + GHOST coords (not the full
# coords). Also reports the coords-compaction ratio to show the memory model.
#
#   mpiexec -n <R> julia --project=bench/distributed test_scheme_b.jl [N] [K]
#   (locally:  julia --project=bench/distributed -e 'using MPI;
#               MPI.mpiexec(e->run(`$e -n 4 julia --project=bench/distributed bench/distributed/test_scheme_b.jl`))')

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

# Identical graph on every rank (deterministic seed) — the replicated INPUT, exactly like the
# scheme-A test. `distribute(..., :partition_coords)` then gives each rank a compacted local
# problem (owned + ghost coords only); we compare its reductions to the serial full-graph values.
rng = MersenneTwister(123)
pts = randn(rng, N, D)
bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 200; jitter = 1e-3)
prob = build_graph(pts, n0, K, bins, vals)

# Serial reference (every rank can compute it from the replicated graph).
ld_serial = generate_logdet(prob)
g_serial = generate_logdet_grad_vals(prob)
data = randn(rng, N)                                  # observations in ORIGINAL order
data_tree = prob.indices === nothing ? data : data[prob.indices]
loss_serial, ginv_serial = generate_inv_loss_grad_vals(prob, data)

# Distribute with scheme B and evaluate.
dprob = distribute(prob, comm; scheme = :partition_coords)
ld = generate_logdet(dprob)
ld2, g = generate_logdet_and_grad_vals(dprob)
loss, ginv = generate_inv_loss_grad_vals(dprob, data_tree)

# Checkpoint roundtrip: save each rank's slab, reload (each rank reads ONLY its own), refit.
ckpt = joinpath(tempdir(), "graphgp_ckpt_$(N)_$(nranks)")
save_graph(ckpt, dprob)
dprob_re = load_graph(ckpt, comm)
ld_reload = generate_logdet(dprob_re)
ld_reload_err = abs(ld_reload - ld_serial) / abs(ld_serial)

# Memory model: max over ranks of (local coord columns) / N — should be « 1 for nranks > 1.
local_cols = length(dprob.gids)
max_cols = MPI.Allreduce(local_cols, MPI.MAX, comm)
sum_owned = MPI.Allreduce(GraphGP.nrefined_local(dprob), MPI.SUM, comm)

reln(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

if rank == 0
    ld_err = abs(ld - ld_serial) / abs(ld_serial)
    g_err = reln(g, g_serial)
    loss_err = abs(loss - loss_serial) / abs(loss_serial)
    ginv_err = reln(ginv, ginv_serial)
    @printf("ranks=%d  N=%d K=%d\n", nranks, N, K)
    @printf("  logdet:        dist=%.6f  serial=%.6f  relerr=%.2e\n", ld, ld_serial, ld_err)
    @printf("  logdet(fused): %.6f  (==above: %s)\n", ld2, ld2 == ld ? "yes" : "no")
    @printf("  grad_vals      norm-relerr=%.2e\n", g_err)
    @printf("  inv-loss:      dist=%.6f  serial=%.6f  relerr=%.2e\n", loss, loss_serial, loss_err)
    @printf("  inv-grad_vals  norm-relerr=%.2e\n", ginv_err)
    @printf("  coords held:   max local cols=%d / N=%d  (%.1f%% of full)  Σowned=%d (==M=%d: %s)\n",
        max_cols, N, 100 * max_cols / N, sum_owned, N - n0, sum_owned == N - n0 ? "yes" : "no")
    @printf("  checkpoint:    save→load logdet relerr=%.2e\n", ld_reload_err)
    ok = ld_err < 1e-9 && g_err < 1e-8 && loss_err < 1e-6 && ginv_err < 1e-6 &&
         sum_owned == N - n0 && ld2 == ld && ld_reload_err < 1e-9
    println(ok ? "  PASS" : "  FAIL")
end

MPI.Finalize()
