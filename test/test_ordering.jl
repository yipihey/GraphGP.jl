# generate / generate_inv original-order convention (drop-in for Python `graphgp`).
#
# Python `graphgp` keeps every public array in ORIGINAL point order: `generate` gathers xi
# original→tree and scatters the field tree→original; `generate_inv` does the reverse. With
# `graph.indices[tree_pos] = original_pos`, the forward map is exactly  Scatter ∘ L ∘ Gather.
#
# These tests pin that convention WITHOUT a live Python dependency by comparing against the
# reference algorithm directly: a second problem with `indices === nothing` runs the pure
# tree-order map `L`, and we apply the gather/scatter by hand. A non-identity `indices` is used
# throughout (the bug these guard against is silent under the identity permutation).

@testset "generate / generate_inv original-order convention" begin
    rng = Random.MersenneTwister(20240607)
    N, D, n0, k = 2_500, 3, 200, 8
    points = randn(rng, N, D)
    bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 400; jitter = 1e-3)

    prob = build_graph(points, n0, k, bins, vals)          # tree/depth order + non-identity indices
    idx = prob.indices
    @test idx !== nothing
    @test idx != collect(1:N)                              # genuinely permuted (else test is vacuous)

    # Same graph, tree order (indices === nothing → pure L, no gather/scatter).
    prob_tree = GraphGPProblem(prob.coords, prob.neighbors, prob.offsets, prob.n0, prob.scale,
        prob.bins, prob.vals)

    gather(v) = v[idx]                                     # original → tree  (matches xi[indices])
    function scatter(v)                                    # tree → original  (matches .at[indices].set)
        out = similar(v); out[idx] = v; out
    end

    # (a) generate parity: generate(prob, xi) == Scatter(L(Gather(xi))), original order in/out.
    xi = randn(rng, N)
    field_ref = scatter(generate(prob_tree, gather(xi)))
    field = generate(prob, xi)
    @test field ≈ field_ref rtol = 1e-12
    # Sanity: it is NOT the no-gather result (that is the bug we are guarding against).
    @test !isapprox(field, scatter(generate(prob_tree, xi)); rtol = 1e-6)

    # (b) inverse parity + roundtrip, all in original order.
    values = randn(rng, N)
    xi_ref = scatter(generate_inv(prob_tree, gather(values)))
    @test generate_inv(prob, values) ≈ xi_ref rtol = 1e-12
    @test generate_inv(prob, generate(prob, xi)) ≈ xi rtol = 1e-10     # exact inverse, original order

    # (c) adjoint correctness: generate_grad_xi is the VJP of generate; both original order.
    # Check against finite differences of  f(xi) = sum(w .* generate(prob, xi))  (w original order).
    w = randn(rng, N)
    xg = generate_grad_xi(prob, w)
    @test length(xg) == N
    f(z) = sum(w .* generate(prob, z))
    ε = 1e-6
    for t in (1, n0, n0 + 1, N ÷ 2, N)                    # probe dense + refined + permuted spots
        e = zeros(N); e[t] = ε
        fd = (f(xi .+ e) - f(xi .- e)) / (2ε)
        @test xg[t] ≈ fd rtol = 1e-4 atol = 1e-6
    end

    # The indices === nothing fast path is unchanged (identity in/out).
    @test generate(prob_tree, xi) ≈ generate(prob_tree, xi)            # stable
    @test generate_inv(prob_tree, generate(prob_tree, xi)) ≈ xi rtol = 1e-10
end
