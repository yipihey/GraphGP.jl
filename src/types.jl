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
end

npoints(p::GraphGPProblem) = size(p.coords, 2)
ndims_space(p::GraphGPProblem) = size(p.coords, 1)
nneighbors(p::GraphGPProblem) = size(p.neighbors, 1)
nrefined(p::GraphGPProblem) = size(p.neighbors, 2)
nbins(p::GraphGPProblem) = length(p.bins)

KernelAbstractions.get_backend(p::GraphGPProblem) = KernelAbstractions.get_backend(p.coords)
