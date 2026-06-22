using GraphGP
using KernelAbstractions
using Random
using Test

include("loadref.jl")

@testset "GraphGP" begin
    include("test_interp.jl")
    include("test_linalg.jl")
    include("test_refine.jl")
end
