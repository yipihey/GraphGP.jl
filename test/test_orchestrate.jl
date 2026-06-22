@testset "generate / generate_inv roundtrip (f64)" begin
    prob, ref = load_problem("small"; T = Float64)
    N = npoints(prob)
    n0 = prob.n0

    rng = Random.MersenneTwister(7)
    xi_orig = randn(rng, Float64, N)

    # Forward then inverse.
    values = generate(prob, xi_orig)
    xi_back = generate_inv(prob, values)
    @test isapprox(xi_back, xi_orig; rtol = 1e-8, atol = 1e-10)
end

@testset "generate / generate_inv roundtrip (f32)" begin
    prob, ref = load_problem("small"; T = Float32)
    N = npoints(prob)

    rng = Random.MersenneTwister(7)
    xi_orig = Float32.(randn(rng, N))

    values = generate(prob, xi_orig)
    @test !any(isnan, values)
    xi_back = generate_inv(prob, values)
    @test !any(isnan, xi_back)
    # f32 roundtrip: looser tolerance due to f32 Cholesky accumulation.
    @test isapprox(Float64.(xi_back), Float64.(xi_orig); rtol = 1e-3, atol = 1e-4)
end

@testset "generate_logdet = dense_logdet + refine_logdet" begin
    prob, _ = load_problem("small"; T = Float64)
    n0 = prob.n0

    ld_dense = generate_dense_logdet(view(prob.coords, :, 1:n0), prob.scale, prob.bins,
        prob.vals, n0)
    ld_ref = refine_logdet(prob)
    ld_total = generate_logdet(prob)

    @test isapprox(ld_total, Float64(ld_dense) + ld_ref; rtol = 1e-12)
end
