module GraphGPCUDAExt

# Custom-CUDA accelerator binding: ccall the vendored graphgp_cuda per-point kernels
# (csrc/libgraphgpcapi.so) on CuArray device pointers, ordered on CUDA.jl's stream. Activates with
# `using GraphGP, CUDA`; the library is built out-of-band by `julia csrc/build.jl`. Falls back with
# a clear message if the .so is missing.

using GraphGP
using CUDA
using Libdl
import GraphGP: refine_logdet_custom, refine_inv_custom, build_graph_cuda, GraphGPProblem,
    npoints, nrefined, nneighbors, ndims_space, nbins,
    quantize_to_lattice, _compute_offsets, _move_to_backend

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

function build_graph_cuda(points::AbstractMatrix, n0::Integer, k::Integer,
        bins::AbstractVector, vals::AbstractVector; lattice_bits::Int = 21)
    N, D = size(points)
    M = N - n0
    backend = CUDABackend()
    # point-major (D, N) Float32 device input (D contiguous per point — the .cu layout).
    pts_dn = CuArray{Float32}(undef, D, N)
    pts_dn .= permutedims(points isa CuArray ? Float32.(points) : CuArray{Float32}(points))
    pout = CuArray{Float32}(undef, D, N)
    indices = CuArray{Int32}(undef, N)
    neighbors = CuArray{Int32}(undef, k, M)          # (k,M) col-major = point-major, 0-based
    depths = CuArray{Int32}(undef, N)
    temp = CuArray{Int32}(undef, 2N)
    rc = ccall(Libdl.dlsym(_lib(), :gpcuda_build_graph_f32), Cint,
        (CuPtr{Float32}, CuPtr{Float32}, CuPtr{Int32}, CuPtr{Int32}, CuPtr{Int32}, CuPtr{Int32},
         Int64, Int64, Int64, Int64, Ptr{Cvoid}),
        pts_dn, pout, indices, neighbors, depths, temp, n0, k, N, D, _stream())
    rc == 0 || error("gpcuda_build_graph_f32 returned $rc")
    CUDA.synchronize()

    # Quantise the reordered points to the integer lattice (as build_graph does on the host).
    coords_nd, _, scale = quantize_to_lattice(permutedims(pout), lattice_bits)   # (N,D) UInt32
    coords = permutedims(coords_nd)                                              # (D,N)
    dh = Int.(Array(depths))
    offsets = _compute_offsets(dh)                   # depths are ascending after order_by_depth
    n0_final = count(==(0), dh)
    nb1 = neighbors .+ Int32(1)                       # 0-based → 1-based
    bins_b = bins isa CuArray{Float32} ? bins : CuArray{Float32}(bins)
    vals_b = vals isa CuArray{Float32} ? vals : CuArray{Float32}(vals)
    return GraphGPProblem(coords, nb1, offsets, n0_final, Float32(scale), bins_b, vals_b,
        Array(indices) .+ 1)
end

end # module
