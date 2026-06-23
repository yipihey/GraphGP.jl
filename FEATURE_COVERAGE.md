# GraphGP.jl feature-coverage audit vs Python `graphgp`

Audit of whether the Julia rewrite (`julia/GraphGP/`) covers the capabilities of the
reference Python/JAX package (`graphgp/`). Date: 2026-06. Based on a function-by-function
inventory of both source trees.

> **Update (2026-06, post-implementation).** Most gaps from the original audit are now closed:
> `check_graph` and `compute_cov_matrix` are implemented; differentiability is exposed via
> `ChainRulesCore.rrule`s (`logdet_of_vals`, `inv_quadratic_loss_of_vals`) and now also covers
> `d/dxi` (`generate_grad_xi`, exact) and `d/dpoints` (`generate_logdet_grad_points`, validated
> vs finite differences); and the GPU compute path now runs end-to-end (`generate`/inverse/
> logdet/gradients on GPU, `to_backend`, GPU `compute_depths`/`quantize`, and a
> backend-agnostic k-NN query `query_preceding_neighbors_ka`). A latent 2× error in the dense
> first-layer logdet gradient was found and fixed in the process. **Remaining:** GPU
> `build_tree`/`order_by_depth` (the k-d tree skeleton is still built on the CPU), and the
> `d/dpoints` path for the inverse-quadratic loss. See "Status after implementation" below.

**Headline (original):** the numeric core (forward generation, inverse, logdet — both the dense
first layer and the Vecchia refinement, plus the `generate*` orchestration) is at **full
parity**, and the graph-construction pipeline (`build_tree`/`query_preceding_neighbors`/
`compute_depths`/`order_by_depth`/`build_graph`) is **also implemented in Julia** (CPU).
GraphGP.jl actually *exceeds* Python on differentiation (hand-written CPU+GPU analytic adjoints
where Python only has slow JAX autodiff, and none in the CUDA extension). The real gaps are:
general-purpose differentiability, GPU graph construction, and a few utility/validation
functions.

## Coverage matrix (Python public API → Julia)

| Python (`__all__`) | Julia | Status | Notes |
| --- | --- | --- | --- |
| `build_tree` | `build_tree` | ✅ | Julia CPU-only (`Matrix{Float64}`); Python has `cuda=` GPU path |
| `query_preceding_neighbors` | `query_preceding_neighbors` | ✅ | Julia CPU-only; max-heap + bbox pruning |
| `compute_depths` | `compute_depths` | ✅ | iterative DAG depth, parity |
| `order_by_depth` | `order_by_depth` | ✅ | argsort + neighbor reindex, parity |
| `build_graph` | `build_graph` | ✅ | Julia also quantizes to 21-bit lattice and returns a `GraphGPProblem` (bundles the kernel); CPU-only |
| `Graph` (pytree dataclass) | `GraphGPProblem` | ◑ | different container; Julia bundles coords+graph+kernel, carries optional `indices` permutation (parity), but is not a JAX/AD pytree |
| `check_graph` | — | ❌ | **missing** graph validator |
| `generate` | `generate` | ✅ | both handle `indices` reordering |
| `generate_inv` | `generate_inv` | ✅ | |
| `generate_logdet` | `generate_logdet` | ✅ | |
| `generate_dense` | `generate_dense` | ✅ | dense Cholesky reference path |
| `generate_dense_inv` | `generate_dense_inv` | ✅ | |
| `generate_dense_logdet` | `generate_dense_logdet` | ✅ | |
| `refine` | `refine` / `refine!` | ✅ | Julia has both the depth-batch sequential apply and an in-place form |
| `refine_inv` | `refine_inv` / `refine_inv!` | ✅ | |
| `refine_logdet` | `refine_logdet` / `refine_logdet_terms` | ✅ | |
| `compute_cov_matrix` | — (internal `assemble_cov!`) | ❌ | no **public** dense full-matrix builder; Julia only assembles per-point `(k+1)×(k+1)` blocks inside kernels |
| `extras.rbf_kernel` | `rbf_kernel` | ✅ | |
| `extras.matern_kernel` | `matern_kernel` | ✅ | |
| `extras.make_cov_bins` | `make_cov_bins` | ✅ | |
| `cov_lookup` (internal) | `cov_lookup` (exported) | ✅ | |

## Differentiability (the most important semantic difference)

| | Python `graphgp` | GraphGP.jl |
| --- | --- | --- |
| Mechanism | generic JAX autodiff (no custom primitives) | hand-written analytic adjoints |
| Backends for grad | JAX CPU/GPU; **none in the `cuda=True` extension** (`NotImplementedError`) | CPU (threaded, privatized) **and** GPU (atomic scatter); plus a fused logdet+grad GPU kernel |
| Differentiate w.r.t. `cov_vals` | ✅ | ✅ (`refine_logdet_grad_vals`, `generate_logdet_grad_vals`, `refine_inv_loss_grad_vals`, `generate_inv_loss_grad_vals`) |
| → kernel hyperparameters | ✅ (autodiff through `rbf`/`matern`) | ✅ via `hyperparam_grad` (ForwardDiff chain rule) |
| Differentiate w.r.t. `xi` / `points` | ✅ (autodiff) | ❌ |
| Arbitrary downstream loss | ✅ (autodiff composes) | ❌ only two fixed functionals: `logdet` and `0.5‖xi‖²` |
| Validated | — | hand-written adjoints cross-checked vs Enzyme-through-KA and the JAX f64 oracle |

Net: for **GP training/inference** (where the gradients that matter are of the log-marginal-
likelihood pieces — `logdet` and the quadratic form — w.r.t. kernel hyperparameters),
GraphGP.jl is fully covered *and faster/more scalable* than Python (Python's only working
gradient path is pure-JAX autodiff, which is ~1000× slower and OOMs at scale; the CUDA
extension has no gradient at all). For **general AD** (e.g. d/dxi, d/dpoints, or gradients of
an arbitrary user loss), Python is more flexible.

## Gaps, by severity

**Functional gaps**
1. **GPU graph construction** — Julia's `build_tree`/`query_preceding_neighbors`/`build_graph`
   are CPU-only; Python has a `cuda=True` path. For billion-point problems the build itself
   becomes a bottleneck. (The benchmarks sidestep this by building the graph on the Python/CUDA
   side and feeding Julia a prebuilt `GraphGPProblem`.)
2. **General differentiability** — only `cov_vals` for two fixed loss functionals (above).
3. **`check_graph`** — no validator for topological order / batch consistency / neighbor
   validity. Easy to port; useful for catching malformed inputs.
4. **`compute_cov_matrix`** — no public dense full-covariance builder (only the internal
   per-point `assemble_cov!`). Minor; mostly a convenience/debugging utility.

**Non-gaps / Julia-only extras (beyond Python)**
- Hand-written analytic Cholesky pullback (`chol_pullback!` / `chol_logdet_pullback!`),
  CPU-threaded and GPU-atomic gradient kernels, and a **fused** logdet+grad GPU kernel.
- `hyperparam_grad` convenience (cov_vals-grad → hyperparam-grad via ForwardDiff).
- Integer-lattice (21-bit/axis) exact-distance representation; f32 default with f64 oracle.
- Backend-agnostic single kernel source (CPU + CUDA) via KernelAbstractions.

## Stale documentation found
- `julia/GraphGP/README.md` states the k-d tree build / neighbor query / depth ordering "stay
  in the Python/JAX side" and that the package "owns only the hot per-point inner loop." That
  is no longer accurate: `tree.jl` + `graph_build.jl` implement the full pipeline in Julia
  (CPU). The README should be updated when GraphGP.jl is split into a standalone library.

## Suggested priorities for standalone-library readiness
1. Update the README to reflect that graph construction is implemented in Julia.
2. Port `check_graph` (cheap, improves robustness).
3. Decide the differentiability contract: document that AD is analytic and restricted to
   `cov_vals`/hyperparameters for `logdet` and the inverse quadratic form — or add a general
   AD path (e.g. via ChainRules/Enzyme rules on the public entry points) if broader AD is a goal.
4. (Larger) GPU graph construction, if Julia is to own the end-to-end pipeline at scale.
5. (Minor) public `compute_cov_matrix` for parity/debugging.

## Status after implementation

| Item | Status |
| --- | --- |
| `check_graph` | ✅ implemented + tested (`graph_build.jl`) |
| `compute_cov_matrix` | ✅ implemented + tested (`dense.jl`) |
| README stale claim | ✅ corrected; Differentiability section added |
| AD: ChainRules over `cov_vals`/hyperparameters | ✅ `logdet_of_vals`, `inv_quadratic_loss_of_vals` rrules (Zygote-composable) |
| AD: `d/dxi` | ✅ `generate_grad_xi` + `generate` rrule (exact; adjoint identity to machine ε) |
| AD: `d/dpoints` (logdet) | ✅ `generate_logdet_grad_points` (validated vs continuous FD) |
| GPU compute path (generate/inverse/logdet/grad) | ✅ works end-to-end; validated in CI GPU testset |
| `to_backend` (build on CPU, run on GPU) | ✅ |
| GPU `compute_depths` / `quantize_to_lattice` | ✅ backend-dispatched KA |
| GPU k-NN query (`query_preceding_neighbors_ka`) | ✅ CPU+GPU; AABB + index-range skip + Float32 packed node records; ~12× faster than the first version at 200 K (0.4 → 4.7 M pts/s) |
| GPU `build_tree` (`build_tree_ka`) | ✅ sort-based level build (one global `sortperm`/level), CPU+GPU; valid k-d tree (brute-force-validated); ~29× faster than the CPU BFS build at 200 K |
| GPU `order_by_depth` (`order_by_depth_ka`) | ✅ GPU `sortperm` of depths + KA scatter for the inverse permutation and neighbor remap |
| Fully-fused on-device build (`build_graph_ka`) | ✅ tree → query → depths → reorder → quantize entirely on `backend`, returns a device-resident `GraphGPProblem` (no host round-trip). 200 K in ~340 ms, 1 M in ~2.1 s; validated by `check_graph` + generate/inverse roundtrip |
| Latent dense-logdet-gradient 2× bug | ✅ found and fixed (was untested against truth) |

**Still open (clearly scoped follow-ups):**
- **`d/dpoints` for the inverse-quadratic loss**, and GPU (atomic-scatter) variants of the
  point gradients (current point-gradient path is CPU/host accumulation).
- **Query throughput**: now ~4.7 M pts/s (Float32 packed records + index-range skip). Still
  below the refine kernels (~150 M pts/s) — irregular per-query tree traversal is latency-bound
  on dependent node reads. Diagnostics ruled out occupancy (stack-size) and coalescing
  (record-packing); f32 precision was the effective lever. Further gains (a buffer-kd-tree /
  query-reordering scheme) are possible but the query is not the build bottleneck (the tree
  build dominates), so this is low priority.
- **Sort primitive:** `build_tree_ka`/`order_by_depth_ka` use generic `sortperm`, which
  dispatches to the CUDA method at run time — so no CUDA package extension was needed to keep
  the core CUDA-free.
