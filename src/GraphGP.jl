module GraphGP

using KernelAbstractions
using LinearAlgebra: LinearAlgebra

include("interp.jl")
include("linalg.jl")
include("types.jl")
include("kernels.jl")
include("kernels_adjoint.jl")
include("api.jl")
include("grad.jl")

export GraphGPProblem
export cov_lookup
export refine_logdet, refine_logdet_terms, refine_inv, refine_inv!
export refine, refine!
export refine_logdet_grad_vals, refine_inv_loss_grad_vals

end # module
