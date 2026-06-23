# Container for a prebuilt GraphGP problem (graph + discretized kernel). The k-d tree
# build / neighbor query / depth ordering is done once on the Python/JAX side; Julia
# consumes the result and owns only the hot per-point numeric core.
#
# Layout is chosen for coalesced per-workitem reads on GPU:
#   coords    :: (D, N)  UInt32   integer lattice coordinates (21-bit/axis Morton-friendly)
#   neighbors :: (K, M)  integer  1-based indices into the point set, M = N - n0
#   offsets   :: (B,)    integer  1-based *exclusive* end index of each depth batch
#   bins,vals :: (nbins,) T       discretized covariance (T defaults to Float32)
#   scale     :: T                physical length per lattice step (isotropic)
struct GraphGPProblem{T, MC <: AbstractMatrix, MN <: AbstractMatrix, V <: AbstractVector}
    coords::MC
    neighbors::MN
    offsets::Vector{Int}
    n0::Int
    scale::T
    bins::V
    vals::V
    # Optional reordering permutation: prob.indices[tree_pos] = original_pos.
    # When built from Python/JAX, this is the `graph.indices` field.
    # Nothing means the identity permutation (points already in the desired order).
    indices::Union{Nothing, Vector{Int}}

    function GraphGPProblem(coords::MC, neighbors::MN, offsets::Vector{Int}, n0::Int,
            scale::T, bins::V, vals::V,
            indices::Union{Nothing, Vector{Int}} = nothing) where {T, MC, MN, V}
        # All array fields that appear in @kernel launches must reside on the same device.
        # offsets and indices are always CPU (used in host dispatch loops only).
        b = KernelAbstractions.get_backend(coords)
        KernelAbstractions.get_backend(neighbors) == b ||
            throw(ArgumentError(
                "neighbors backend ($(KernelAbstractions.get_backend(neighbors))) ≠ " *
                "coords backend ($b): all kernel arrays must be on the same device"))
        KernelAbstractions.get_backend(bins) == b ||
            throw(ArgumentError(
                "bins backend ($(KernelAbstractions.get_backend(bins))) ≠ " *
                "coords backend ($b): all kernel arrays must be on the same device"))
        KernelAbstractions.get_backend(vals) == b ||
            throw(ArgumentError(
                "vals backend ($(KernelAbstractions.get_backend(vals))) ≠ " *
                "coords backend ($b): all kernel arrays must be on the same device"))
        new{T, MC, MN, V}(coords, neighbors, offsets, n0, scale, bins, vals, indices)
    end
end


npoints(p::GraphGPProblem) = size(p.coords, 2)
ndims_space(p::GraphGPProblem) = size(p.coords, 1)
nneighbors(p::GraphGPProblem) = size(p.neighbors, 1)
nrefined(p::GraphGPProblem) = size(p.neighbors, 2)
nbins(p::GraphGPProblem) = length(p.bins)

KernelAbstractions.get_backend(p::GraphGPProblem) = KernelAbstractions.get_backend(p.coords)

# Move an array to `backend` (backend-generic: works for CPU and any KA GPU backend without a
# hard CUDA dependency).
function _move_to_backend(x::AbstractArray, backend)
    dst = KernelAbstractions.allocate(backend, eltype(x), size(x)...)
    copyto!(dst, x)
    return dst
end

"""
    to_backend(prob::GraphGPProblem, backend) -> GraphGPProblem

Return a copy of `prob` with its device arrays (`coords`, `neighbors`, `bins`, `vals`) moved to
`backend` — e.g. `to_backend(prob, CUDABackend())` to run generation/gradients on the GPU, or
`to_backend(prob, CPU())` to bring a device problem back to the host. `offsets`/`indices` are
host-side dispatch data and stay on the CPU. This is the one-call "build on CPU, run on GPU"
bridge; all kernels then dispatch to `backend` automatically.
"""
function to_backend(prob::GraphGPProblem, backend)
    return GraphGPProblem(
        _move_to_backend(prob.coords, backend),
        _move_to_backend(prob.neighbors, backend),
        prob.offsets, prob.n0, prob.scale,
        _move_to_backend(prob.bins, backend),
        _move_to_backend(prob.vals, backend),
        prob.indices)
end
