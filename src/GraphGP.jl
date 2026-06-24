module GraphGP

using KernelAbstractions
using LinearAlgebra: LinearAlgebra

include("interp.jl")
include("linalg.jl")
include("types.jl")
include("kernels.jl")
include("kernels_adjoint.jl")
include("cpu_native.jl")
include("api.jl")
include("grad.jl")
include("extras.jl")
include("dense.jl")
include("aniso.jl")
include("orchestrate.jl")
include("tree.jl")
include("tree_gpu.jl")
include("graph_build.jl")
include("chainrules.jl")
include("grad_generate_vals.jl")

export GraphGPProblem, npoints, nneighbors, nrefined, ndims_space, nbins, to_backend
export cov_lookup
export refine_logdet, refine_logdet_terms, refine_inv, refine_inv!
export refine, refine!
export refine_logdet_grad_vals, refine_inv_loss_grad_vals
export generate_logdet_grad_vals, generate_inv_loss_grad_vals
export generate_logdet_and_grad_vals
# Phase 7: kernel hyperparameter layer
export make_cov_bins, rbf_kernel, matern_kernel, hyperparam_grad
# Anisotropic covariance K(Δspatial, Δz) — forward-only drop-in (see src/aniso.jl)
export AnisoCov, build_anisotropic_covariance, aniso_lookup
# Phase 8: dense first layer + orchestration
export generate_dense, generate_dense_inv, generate_dense_logdet, compute_cov_matrix
export generate, generate_inv, generate_logdet
# Phase 9: graph build pipeline
export build_graph, compute_depths, order_by_depth, build_tree
export query_preceding_neighbors, quantize_to_lattice, check_graph
export query_preceding_neighbors_ka, build_tree_ka, order_by_depth_ka, build_graph_ka
# Phase 10: differentiability (ChainRules entry points)
export logdet_of_vals, inv_quadratic_loss_of_vals, generate_grad_xi
export generate_grad_vals, generate_of_vals
export logdet_of_points, inv_quadratic_loss_of_points
export refine_logdet_grad_points, generate_logdet_grad_points
export refine_inv_loss_grad_points, generate_inv_loss_grad_points

end # module
