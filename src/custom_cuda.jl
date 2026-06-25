# Optional hand-written CUDA accelerator — a thin wrapper around the vendored graphgp_cuda device
# kernels (csrc/), callable from Julia for the per-point, order-independent ops where custom CUDA
# can beat the portable KernelAbstractions path. Implemented by the `GraphGPCUDAExt` extension,
# which activates with `using CUDA` once the accelerator library is built (`julia csrc/build.jl`).
# The portable `refine_logdet` / `refine_inv` need neither CUDA-the-extension nor the .so.

@noinline _custom_cuda_required() = error(
    "refine_*_custom requires the CUDA extension + the built accelerator library. Build it with " *
    "`julia csrc/build.jl`, then `using CUDA` alongside `using GraphGP`. The portable " *
    "KernelAbstractions path (`refine_logdet` / `refine_inv`) needs neither.")

"""
    refine_logdet_custom(prob) -> Float32

Hand-written-CUDA `refine_logdet` (Σ log std) via the optional accelerator library. Requires a GPU
(`CuArray`) problem, `using CUDA`, and the built `csrc/libgraphgpcapi.so`. Matches the portable
[`refine_logdet`](@ref) to Float32 round-off; see `docs/benchmarks.md` for the speedup.
"""
refine_logdet_custom(args...; kwargs...) = _custom_cuda_required()

"""
    refine_inv_custom(prob, values) -> CuVector{Float32}

Hand-written-CUDA `refine_inv` (recover the unit-normal `xi` from `values`) via the accelerator
library. Same requirements/contract as [`refine_logdet_custom`](@ref).
"""
refine_inv_custom(args...; kwargs...) = _custom_cuda_required()

"""
    build_graph_cuda(points, n0, k, bins, vals; lattice_bits=21) -> GraphGPProblem

Fast **shallow on-GPU** graph build via the optional accelerator library — the same hand-written
pipeline as `gp.build_graph(cuda=True)` (build_tree special heap order → preceding k-NN → depths →
order-by-depth), all on the device. Returns a GPU `GraphGPProblem` (shallow, ≈tens of depth
batches) ready for `generate` / the derivatives. `points` is `(N, D)` (CPU or `CuArray`); needs
`using CUDA` + the built `csrc/libgraphgpcapi.so`. This closes the build gap at scale where the
portable CPU `build_graph` is gather-bound and `build_graph_ka` is deep (see docs/benchmarks.md §5).
"""
build_graph_cuda(args...; kwargs...) = _custom_cuda_required()
