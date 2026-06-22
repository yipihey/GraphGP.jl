# Top-level orchestration: glue the dense first layer (n0 points) to the refined layer
# (M = N - n0 points) to produce the full GP operations. Matches `refine.py:generate`,
# `generate_inv`, and `generate_logdet`.
#
# If prob.indices is set, the output is reordered back to the original point order (the
# inverse permutation), matching graph.py:Graph.indices semantics.

"""
    generate(prob, xi; backend) -> Vector{T}

Full GP forward generation: sample all N points given unit-normal parameters `xi`.
`xi[1:n0]` feeds the dense first layer; `xi[n0+1:N]` feeds the refinement.
If `prob.indices` is set, output is reordered to the original point ordering.
Matches `refine.py:generate`.
"""
function generate(prob::GraphGPProblem{T}, xi::AbstractVector{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    # Dense layer.
    v_dense = generate_dense(view(prob.coords, :, 1:n0), prob.scale, prob.bins, prob.vals,
        xi[1:n0])
    # Refinement layer (sequential scan over depth batches).
    N = npoints(prob)
    values = KernelAbstractions.zeros(backend, T, N)
    copyto!(view(values, 1:n0), v_dense)
    refine!(values, prob, xi[n0+1:end]; backend = backend)
    # Reorder to original ordering if the problem was built from a permuted graph.
    if prob.indices !== nothing
        out = similar(values)
        out[prob.indices] = values
        return out
    end
    return values
end

"""
    generate_inv(prob, values; backend) -> Vector{T}

Inverse of `generate`: recover unit-normal parameters `xi` from observed GP values.
If `prob.indices` is set, values are permuted from original to tree/depth order first.
Matches `refine.py:generate_inv`.
"""
function generate_inv(prob::GraphGPProblem{T}, values::AbstractVector{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    # Permute to tree/depth order if needed.
    values_ord = if prob.indices !== nothing
        values[prob.indices]
    else
        values
    end
    # Refinement inverse.
    xi_ref = refine_inv(prob, values_ord; backend = backend)
    # Dense inverse.
    xi_dense = generate_dense_inv(view(prob.coords, :, 1:n0), prob.scale, prob.bins, prob.vals,
        values_ord[1:n0])
    return vcat(xi_dense, xi_ref)
end

"""
    generate_logdet(prob; backend) -> T

Log-determinant of the full GP covariance (dense + refinement contributions).
Matches `refine.py:generate_logdet`.
"""
function generate_logdet(prob::GraphGPProblem{T};
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    ld_dense = generate_dense_logdet(view(prob.coords, :, 1:n0), prob.scale, prob.bins,
        prob.vals, n0)
    ld_ref = refine_logdet(prob; backend = backend)
    return T(ld_dense) + ld_ref
end
