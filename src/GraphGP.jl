module GraphGP

using KernelAbstractions
using LinearAlgebra: LinearAlgebra

include("interp.jl")
include("linalg.jl")
include("types.jl")
include("kernels.jl")
include("api.jl")

export GraphGPProblem
export cov_lookup
export refine_logdet, refine_logdet_terms, refine_inv, refine_inv!

end # module
