# Custom-CUDA accelerator (optional)

A thin C-ABI bridge that lets GraphGP.jl call **hand-written CUDA** kernels — the per-point
ops (`refine_logdet`, `refine_inv`) **and the full shallow graph build** (`build_graph_cuda`) —
alongside the portable KernelAbstractions path. The device kernels are vendored from the original
`graphgp_cuda` extension (`vendor/`, MIT, © Benjamin Dodge & Philipp Frank); `graphgp_capi.cu`
wraps them in `extern "C"` launchers that take raw device pointers, and the `GraphGPCUDAExt`
extension `ccall`s them on `CuArray`s.

**Where it pays off: the graph build.** `build_graph_cuda` (the hand-written build_tree → k-NN →
depths → order pipeline) is **at parity with `gp.build_graph(cuda=True)`** (~4.6 s vs 4.1 s at 10 M,
~55 s vs 49 s at 80 M on an A6000) and produces the same *shallow* graph as the CPU `build_graph` —
closing the one place where the portable path lost at scale (CPU build is gather-bound; `build_graph_ka`
is ~3× slower and deep). The per-point refine kernels, by contrast, do **not** beat KA (below).

```julia
using GraphGP, CUDA
prob = build_graph_cuda(points, n0, k, bins, vals)   # fast shallow GPU build → GPU GraphGPProblem
field = generate(prob, xi)                            # then the winning on-device forward/derivatives
```

## Build (out-of-band; not part of the package build or CI)

```bash
julia csrc/build.jl                       # auto-detects nvcc (the cu13 toolkit) + sm_86
# or: NVCC=/path/to/nvcc GPU_ARCH=sm_90 julia csrc/build.jl
```

This produces `csrc/libgraphgpcapi.so` (gitignored). Then:

```julia
using GraphGP, CUDA          # extension activates; finds the .so next to csrc/ (or $GRAPHGP_CUDA_LIB)
prob = to_backend(build_graph(points, n0, k, bins, vals), CUDABackend())
refine_logdet_custom(prob)            # hand-CUDA path
refine_inv_custom(prob, values)
```

The package runs fully without any of this — `refine_logdet` / `refine_inv` use the portable KA
kernels and need neither CUDA-the-extension nor the `.so`.

## Honest finding: it is not (currently) faster than the KA path

We wired this up expecting the ~2× that hand-written CUDA bought us on the hydro solvers. It does
**not** transfer here. On an A6000 (Float32, D=3), custom-kernel-only vs KA throughput:

| op | k=8 (MAX_K=8) | k=10 (MAX_K=16) | k=16 (MAX_K=16) |
| --- | --- | --- | --- |
| `refine_logdet` KA | **415 M/s** | **257 M/s** | 74 M/s |
| `refine_logdet` custom | 365 M/s | 219 M/s | 74 M/s |
| `refine_inv` KA | **352 M/s** | **210 M/s** | 65 M/s |
| `refine_inv` custom | 291 M/s | 177 M/s | 62 M/s |

Both compute the same answer (validated to ~1–7e-6 f32). Why KA wins / ties:

1. These per-point kernels are **register/occupancy-bound** (each thread assembles and Cholesky-
   factorizes a `(k+1)²` block in registers) — there is little headroom for a hand-written kernel
   to exploit, unlike the memory-/branch-heavy hydro kernels.
2. The GraphGP.jl KA kernels **specialize on the exact `k` via `Val(K)`** (tight register arrays),
   whereas the vendored kernel pads to `MAX_K ∈ {4,8,16,32,64}` — so it is notably slower at
   non-power-of-two `k` (e.g. k=10 → MAX_K=16) and only ties at large `k`.
3. The custom path also needs float positions (`Float32(coords)·scale`); the KA path works on the
   integer lattice directly.

So GraphGP.jl's portable path is already at or beyond the hand-written reference for this workload.
A real custom-CUDA win would require *new* kernels (warp-cooperative Cholesky, shared-memory
tiling, exact-`k` instantiation) — a research effort with uncertain payoff given how strong the KA
path already is.

## Why keep the bridge

- **Cross-check.** An independent (different-codebase) validation of the KA kernels at f32.
- **Drop-in point.** If someone writes better hand-tuned kernels, they slot in behind the same
  `refine_*_custom` API with no further Julia plumbing.
- **Reuse.** Demonstrates calling the original project's CUDA kernels from GraphGP.jl directly.
