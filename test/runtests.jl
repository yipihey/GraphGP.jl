using GraphGP
using KernelAbstractions
using Random
using Test
using ForwardDiff

include("loadref.jl")

@testset "GraphGP" begin
    include("test_interp.jl")
    include("test_linalg.jl")
    include("test_refine.jl")
    include("test_gradients.jl")
    include("test_extras.jl")
    include("test_dense.jl")
    include("test_orchestrate.jl")
    include("test_graph_build.jl")
end
