# Distributed GraphGP.jl (MPI) — Phases 1–4

Distribute the GraphGP pipeline across many ranks/GPUs. The layer is thin: it dispatches on a
`DistributedGraphGPProblem` wrapper and reuses the existing per-point kernels, adding only a
partition + `MPI` reduction layer (MPI is a weakdep/extension; `using GraphGP` pulls no MPI).
All reductions accumulate in Float64, so results are **bit-identical across rank counts**.

| phase | capability | API |
| --- | --- | --- |
| 1 | log-likelihood + cov-table gradient (fitting) | `generate_logdet`, `generate_logdet_grad_vals`, `generate_logdet_and_grad_vals` on a `DistributedGraphGPProblem` |
| 2 | inverse: loss gradient + `refine_inv` | `generate_inv_loss_grad_vals(dprob, data)`, `refine_inv(dprob, data)` |
| 3 | field generation | 3a realizations (replicate graph, per-rank `generate`); 3b single field `generate(dprob, xi)` |
| 4 | graph construction | `distributed_build_graph(points, comm, n0, k, bins, vals)` (replicated tree + partitioned query, tree order → fitting); `distributed_quantize(points_local, comm)` (global lattice via Allreduce, scheme-B foundation) |

`distribute(prob, comm)` wraps an existing problem (partition `neighbors`, replicate `coords`);
`distributed_build_graph` builds the partitioned graph directly without ever materialising the
full `neighbors` on one rank. The build is **tree order**, which fits identically (the Vecchia
logdet is invariant to the depth reorder) but is not depth-batched for forward `generate` — use
`build_graph_ka` for generation, or a distributed depth-sort (the remaining scheme-B piece).

## What's here

- `spike_allreduce.jl` — Phase 0 plumbing gate: MPI init, one-rank-per-GPU binding, host +
  device `Allreduce` (CUDA-aware if available, else host-staged).
- `test_distributed_inv.jl` (Phase 2), `test_distributed_gen.jl` (Phase 3),
  `test_distributed_build.jl` (Phase 4) — correctness vs serial, rank-invariant.
- `test_distributed.jl` — correctness: distributed `generate_logdet` / `generate_logdet_grad_vals`
  must reproduce the serial result and be invariant to the rank count. `[N] [K] [cpu|gpu]`.
- `bench_distributed.jl` — strong-scaling timing of the fused `generate_logdet_and_grad_vals`.
- `Project.toml` — env with MPI + MPIPreferences + CUDA + GraphGP (path dep).

## Run locally (JLL MPI, no cluster)

`mpiexec` ships with MPI.jl. The simplest launcher:

```bash
cd julia/GraphGP/bench/distributed
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# one-liner that runs R ranks of a script:
julia --project=. -e '
  using MPI; jl = joinpath(Sys.BINDIR, Base.julia_exename())
  MPI.mpiexec() do exe; run(`$exe -n 4 $jl --project=. test_distributed.jl 20000 10`); end'
```

Locally the JLL MPI is **not** CUDA-aware (`MPI.has_cuda() == false`); the device reduction
falls back to host-staging, which is free for the tiny payload. Multiple ranks share the single
local GPU.

## Run on the cluster (system OpenMPI 4.1.4, multi-node, Slurm)

Bind MPI.jl to the **system** MPI once (the default JLL MPI does not interoperate with Slurm
PMI), then launch with `srun`:

```bash
# 1. point MPI.jl at the system OpenMPI (module must be loaded), then re-precompile:
julia --project=julia/GraphGP/bench/distributed -e \
  'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=julia/GraphGP/bench/distributed -e 'using Pkg; Pkg.precompile()'  # ONCE, single rank
                                                                                  # (shared depot: avoid concurrent first-run races)
# 2. in the batch script, one rank per GPU:
srun -n $NTASKS --gpus-per-task=1 julia --project=julia/GraphGP/bench/distributed \
  bench_distributed.jl 1000000000 20 gpu
```

If the system OpenMPI is built with UCX+CUDA, `MPI.has_cuda()` is `true` and the device
`Allreduce` goes GPUDirect; otherwise it host-stages (still correct, negligible cost here). GPU
binding uses the node-local communicator (`MPI.Comm_split_type(..., COMM_TYPE_SHARED)` →
`CUDA.device!(local_rank % ndev)`), done before any CUDA allocation.

## Status / validated

- Correctness (local, CPU and GPU, R=1,2,4): distributed logdet equals the serial Float64
  reference exactly and is **bit-identical across rank counts** (Float64 reduction); gradient
  agrees to ~1e-15.
- Strong scaling (CPU smoke test, N=200K, contended host): per-iteration time falls with R
  (2.6 s → 1.1 s for R=1→4); at small N a fixed rank-0 dense block (`n0×n0` Cholesky) caps the
  speedup (Amdahl) — it shrinks as N grows.

## Notes / next phases

- **Memory:** each rank holds the full (replicated) `coords` (~12 B/pt) + its `neighbors` slice
  (the dominant ~80 B/pt, partitioned). This scales to ~1–3 B points before coords replication
  is the ceiling; beyond that needs the spatial coords partition (Phase 4).
- **Graph for huge N:** build once on a fat-RAM node and scatter `neighbors` slices (the eval is
  decoupled from distributed graph construction). The local test/bench rebuild the graph per
  rank for simplicity — for large N, dump once and load the slice.
- Excludes (by design, this phase): point-position gradients (need a per-point reduce-scatter),
  forward sampling, distributed graph build.
