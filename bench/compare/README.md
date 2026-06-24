# GraphGP.jl ↔ graphgp comparison harness

A self-contained, reproducible comparison of the GraphGP implementations on **one shared
graph**. Every path consumes the *identical* dumped graph — same lattice points, same neighbour
lists, same covariance table — so differences are due only to the implementation, not the
input. The harness reports two things, in this order:

1. **Correctness cross-check** (Float64, element-wise) — do the implementations compute the
   *same answer*?
2. **Throughput** (Float32, the production precision) — how fast?

## Paths compared

| label | what it is | device | autodiff |
| --- | --- | --- | --- |
| `jax-cpu`  | pure-JAX `graphgp` (`cuda=False`) | CPU | yes |
| `jax-gpu`  | pure-JAX `graphgp` (`cuda=False`) | GPU | yes |
| `cuda-gpu` | `graphgp` CUDA extension (`cuda=True`) | GPU | **no** |
| `julia-cpu`| GraphGP.jl, CPU backend | CPU | yes |
| `julia-gpu`| GraphGP.jl, CUDA backend | GPU | yes |

The pure-JAX path materialises the full `(M, k+1, k+1)` covariance tensor and calls a batched
`cholesky`; GraphGP.jl and the CUDA extension use a fused per-point kernel that never
materialises it. The CUDA extension has no differentiation rule (`jax.grad` is unavailable).

## Run

```bash
# from this directory; args: N [K D] [NTHREADS]
./run_all.sh 2000000 10 3 64
cat results/report.md
```

Environment knobs (see top of `run_all.sh`): `PY` (python with jax+graphgp), `JL` (julia),
`CORES` (taskset list for CPU runs), `GPU=off` (skip GPU paths). CPU runs for **both** JAX and
Julia are confined to the same cores via `taskset`, and JAX's OpenBLAS/OMP thread count is
capped to the core budget, so the CPU comparison is fair under machine load.

## Pieces

- `dump_graph.py` — build one graph, dump `results/graph.npz`.
- `run_jax.py <graph> <correctness|timing> <jax|cuda> <outdir>` — one JAX path; device via
  `JAX_PLATFORMS`.
- `run_julia.jl <graph> <correctness|timing> <cpu|gpu> <outdir>` — one GraphGP.jl path.
- `report.py <outdir>` — aggregate the `*.npz` outputs + `timings.jsonl` into Markdown.

## Real ECHOES graphs

`dump_graph.py` builds a *synthetic* point cloud. To benchmark the graphs the ECHOES field
pipeline actually builds, use `dump_graph_echoes.py` (same output schema **plus** the
`indices` permutation, so it also feeds `run_graphgp.jl`):

```bash
# survey ∈ {boss, local}; reads ~/Projects/ECHOES/data (override with ECHOES_ROOT)
python dump_graph_echoes.py boss 2400000 results/boss_2400000/graph.npz 30
```

- `boss`  — CMASS-South *randoms* (the candidate set the field is generated on), SGC footprint
  + CMASS z-cut, embedded as `(n̂, α·z)` (4-D), exactly as
  `twopt_density/observed_ls.generate_catalogs_from_kernel`.
- `local` — 2M++ comoving xyz (3-D); real galaxies at catalog size, uniform-in-volume
  candidates (ZoA gap) above it.

`run_echoes_bench.sh "boss:120000 boss:560000 boss:2400000 local:70000 local:1400000"` sweeps N,
runs every path per point (each in `results/<survey>_<N>/`), and `report_sweep.py` collates the
three real-graph tables: correctness vs N, throughput vs N, and the pure-JAX-GPU **OOM** matrix
(the no-materialization headline). The build uses the ECHOES venv (`~/.venv/k3d`, has `echoes`
+ `graphgp`); GPU paths use the `.venv-gpu` (has `graphgp_cuda`).

## Notes on fairness

- **Same graph, same precision tier.** Correctness is checked in Float64 (independent of f32
  summation order); throughput in Float32.
- **Same cores for CPU.** `taskset` + capped BLAS threads → JAX-CPU and Julia-CPU get identical
  resources.
- **Small vs large N.** At small N the CUDA extension is launch-overhead-bound and looks slow;
  run a large N (≥5 M) for the GPU steady-state, where GraphGP.jl and the CUDA extension are at
  parity. Pure-JAX materialises `(M,k+1,k+1)` and OOMs past ~20 M.
