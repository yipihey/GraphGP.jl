module GraphGPCUDAExt

# Custom-CUDA accelerator binding: ccall the vendored graphgp_cuda per-point kernels
# (csrc/libgraphgpcapi.so) on CuArray device pointers, ordered on CUDA.jl's stream. Activates with
# `using GraphGP, CUDA`; the library is built out-of-band by `julia csrc/build.jl`. Falls back with
# a clear message if the .so is missing.

using GraphGP
using CUDA
using Libdl
import GraphGP: refine_logdet_custom, refine_inv_custom, GraphGPProblem,
    npoints, nrefined, nneighbors, ndims_space, nbins

const LIBREF = Ref{Ptr{Cvoid}}(C_NULL)
function _lib()
    LIBREF[] != C_NULL && return LIBREF[]
    path = get(ENV, "GRAPHGP_CUDA_LIB",
        joinpath(pkgdir(GraphGP), "csrc", "libgraphgpcapi.so"))
    isfile(path) || error("custom CUDA accelerator not found at $path — build it with " *
                          "`julia csrc/build.jl` (or set ENV[\"GRAPHGP_CUDA_LIB\"]).")
    LIBREF[] = Libdl.dlopen(path)
    return LIBREF[]
end

# Physical float positions (D,N) + 0-based Int64 neighbors + f32 cov table, all on the device.
function _prep(prob::GraphGPProblem)
    prob.coords isa CuArray || error("refine_*_custom requires a GPU (CuArray) problem; " *
                                     "move it with `to_backend(prob, CUDABackend())`.")
    pts = Float32.(prob.coords) .* Float32(prob.scale)          # (D,N) physical positions
    nbr0 = Int64.(prob.neighbors) .- Int64(1)                   # (K,M) 0-based, point-major
    bins = prob.bins isa CuArray{Float32} ? prob.bins : CuArray{Float32}(prob.bins)
    vals = prob.vals isa CuArray{Float32} ? prob.vals : CuArray{Float32}(prob.vals)
    return pts, nbr0, bins, vals
end

_stream() = reinterpret(Ptr{Cvoid}, CUDA.stream().handle)

function refine_logdet_custom(prob::GraphGPProblem)
    pts, nbr0, bins, vals = _prep(prob)
    out = CUDA.zeros(Float32, 1)
    rc = ccall(Libdl.dlsym(_lib(), :gpcuda_refine_logdet_f32), Cint,
        (CuPtr{Float32}, CuPtr{Int64}, CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32},
         Int64, Int64, Int64, Int64, Int64, Ptr{Cvoid}),
        pts, nbr0, bins, vals, out,
        prob.n0, nneighbors(prob), npoints(prob), nbins(prob), ndims_space(prob), _stream())
    rc == 0 || error("gpcuda_refine_logdet_f32 returned $rc")
    CUDA.synchronize()
    return CUDA.@allowscalar out[1]
end

function refine_inv_custom(prob::GraphGPProblem, values::AbstractVector)
    pts, nbr0, bins, vals = _prep(prob)
    vv = values isa CuArray{Float32} ? values : CuArray{Float32}(values)
    xi = CUDA.zeros(Float32, nrefined(prob))
    rc = ccall(Libdl.dlsym(_lib(), :gpcuda_refine_inv_f32), Cint,
        (CuPtr{Float32}, CuPtr{Int64}, CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32},
         Int64, Int64, Int64, Int64, Int64, Ptr{Cvoid}),
        pts, nbr0, bins, vals, vv, xi,
        prob.n0, nneighbors(prob), npoints(prob), nbins(prob), ndims_space(prob), _stream())
    rc == 0 || error("gpcuda_refine_inv_f32 returned $rc")
    CUDA.synchronize()
    return xi
end

end # module
