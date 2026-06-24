# Kernel derivative of the forward pass: d/d(cov_vals) of generate(prob, xi).
# generate_grad_vals is the analytic VJP; check it against finite differences and confirm it
# composes through Zygote via the generate_of_vals rrule (which also returns the d/dxi tangent).

@testset "generate_grad_vals (kernel derivative of the forward pass)" begin
    using Zygote
    rng = Random.MersenneTwister(1)
    N, D, n0, k = 2_000, 3, 100, 8
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 50; jitter = 1e-2)   # healthy jitter → PD dense block
    prob = build_graph(pts, n0, k, bins, vals)
    xi = randn(rng, N)
    w = randn(rng, N)

    dvals = generate_grad_vals(prob, xi, w)
    @test length(dvals) == length(vals)
    @test all(isfinite, dvals)

    # (a) finite differences of  L(vals) = sum(w .* generate_of_vals(prob, vals, xi)).
    f(v) = sum(w .* generate_of_vals(prob, v, xi))
    ε = 1e-6
    for i in (5, 12, 25, 30, 40, 45)
        e = zeros(length(vals)); e[i] = ε
        fd = (f(vals .+ e) - f(vals .- e)) / (2ε)
        @test dvals[i] ≈ fd rtol = 1e-4 atol = 1e-6
    end

    # (b) Zygote composition: d/dvals AND d/dxi through generate_of_vals match the analytic VJPs.
    gz_vals = Zygote.gradient(v -> sum(w .* generate_of_vals(prob, v, xi)), vals)[1]
    @test gz_vals ≈ dvals rtol = 1e-5
    gz_xi = Zygote.gradient(z -> sum(w .* generate_of_vals(prob, vals, z)), xi)[1]
    @test gz_xi ≈ generate_grad_xi(prob, w) rtol = 1e-5
end
