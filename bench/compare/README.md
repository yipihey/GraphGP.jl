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

## Prerequisites

The `jax-*` and `cuda-*` paths compare against the **Python** reference, which is a separate
package — install it into a Python env and point `PY` at that interpreter:

```bash
pip install jax graphgp            # + graphgp_cuda for the cuda-gpu path (optional)
export PY=/path/to/that/python
```

The `julia-*` paths need only this repository (`julia --project=bench`). If you only want the
GraphGP.jl numbers, skip the Python paths.

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

## Notes on fairness

- **Same graph, same precision tier.** Correctness is checked in Float64 (independent of f32
  summation order); throughput in Float32.
- **Same cores for CPU.** `taskset` + capped BLAS threads → JAX-CPU and Julia-CPU get identical
  resources.
- **Small vs large N.** At small N the CUDA extension is launch-overhead-bound and looks slow;
  run a large N (≥5 M) for the GPU steady-state, where GraphGP.jl and the CUDA extension are at
  parity. Pure-JAX materialises `(M,k+1,k+1)` and OOMs past ~20 M.
