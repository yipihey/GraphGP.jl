# Top-level orchestration: glue the dense first layer (n0 points) to the refined layer
# (M = N - n0 points) to produce the full GP operations. Matches `refine.py:generate`,
# `generate_inv`, and `generate_logdet`.
#
# If prob.indices is set, the output is reordered back to the original point order (the
# inverse permutation), matching graph.py:Graph.indices semantics.

"""
    generate(prob, xi; backend) -> Vector{T}

Full GP forward generation: sample all N points given unit-normal parameters `xi`.
Both `xi` and the returned field are in the ORIGINAL point order (drop-in for Python
`graphgp.generate`): when `prob.indices` is set, `xi` is gathered original→tree on input and
the field is scattered tree→original on output, so no caller-side reordering is needed. (With
`prob.indices === nothing` the graph is already in tree order and both steps are the identity.)
Internally `xi[1:n0]` (tree order) feeds the dense first layer and `xi[n0+1:N]` the refinement.
Matches `refine.py:generate`.
"""
function generate(prob::GraphGPProblem{T}, xi::AbstractVector{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    # Gather xi original→tree (graph.indices[tree_pos] = original_pos), matching Python; identity
    # when indices is nothing (no allocation).
    xi_ord = prob.indices !== nothing ? xi[_move_to_backend(prob.indices, backend)] : xi
    # Dense layer (anisotropic or isotropic).
    v_dense = prob.cov !== nothing ?
        generate_dense_aniso(view(prob.coords, :, 1:n0), prob.scale, prob.cov, xi_ord[1:n0]) :
        generate_dense(view(prob.coords, :, 1:n0), prob.scale, prob.bins, prob.vals, xi_ord[1:n0])
    # Refinement layer (sequential scan over depth batches).
    N = npoints(prob)
    values = KernelAbstractions.zeros(backend, T, N)
    copyto!(view(values, 1:n0), v_dense)
    refine!(values, prob, xi_ord[n0+1:end]; backend = backend)
    # Scatter the field tree→original if the problem was built from a permuted graph.
    if prob.indices !== nothing
        out = similar(values)
        idx = _move_to_backend(prob.indices, backend)   # index vector must be on-device for GPU
        out[idx] = values
        return out
    end
    return values
end

"""
    generate_inv(prob, values; backend) -> Vector{T}

Inverse of `generate`: recover unit-normal parameters `xi` from observed GP values. Both
`values` and the returned `xi` are in the ORIGINAL point order (drop-in for Python
`graphgp.generate_inv`): when `prob.indices` is set, `values` is gathered original→tree on input
and `xi` is scattered tree→original on output. Exact inverse of [`generate`](@ref) in original
order. (Identity reordering when `prob.indices === nothing`.) Matches `refine.py:generate_inv`.
"""
function generate_inv(prob::GraphGPProblem{T}, values::AbstractVector{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    # Gather values original→tree if needed.
    values_ord = if prob.indices !== nothing
        values[_move_to_backend(prob.indices, backend)]
    else
        values
    end
    # Refinement inverse (on the problem's backend).
    xi_ref = refine_inv(prob, values_ord; backend = backend)
    # Dense inverse (host LAPACK; small block; anisotropic or isotropic).
    xi_dense = prob.cov !== nothing ?
        generate_dense_inv_aniso(view(prob.coords, :, 1:n0), prob.scale, prob.cov, values_ord[1:n0]) :
        generate_dense_inv(view(prob.coords, :, 1:n0), prob.scale, prob.bins, prob.vals,
            values_ord[1:n0])
    # Assemble the full xi (tree order: dense block first, then refined).
    N = npoints(prob)
    xi = KernelAbstractions.zeros(backend, T, N)
    copyto!(view(xi, 1:n0), xi_dense)        # host → device if on GPU
    copyto!(view(xi, n0 + 1:N), xi_ref)
    # Scatter xi tree→original so input and output share the original ordering.
    if prob.indices !== nothing
        out = similar(xi)
        out[_move_to_backend(prob.indices, backend)] = xi
        return out
    end
    return xi
end

"""
    generate_logdet(prob; backend) -> T

Log-determinant of the full GP covariance (dense + refinement contributions).
Matches `refine.py:generate_logdet`.
"""
function generate_logdet(prob::GraphGPProblem{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    ld_dense = prob.cov !== nothing ?
        generate_dense_logdet_aniso(view(prob.coords, :, 1:n0), prob.scale, prob.cov, n0) :
        generate_dense_logdet(view(prob.coords, :, 1:n0), prob.scale, prob.bins, prob.vals, n0)
    ld_ref = refine_logdet(prob; backend = backend)
    return T(ld_dense) + ld_ref
end
