@testset "generate_dense roundtrip (f64)" begin
    prob, ref = load_problem("small"; T = Float64)
    n0 = prob.n0
    coords_n0 = prob.coords[:, 1:n0]

    # Random xi, generate then invert.
    rng = Random.MersenneTwister(42)
    xi = randn(rng, Float64, n0)
    vals_dense = generate_dense(coords_n0, prob.scale, prob.bins, prob.vals, xi)
    xi_back = generate_dense_inv(coords_n0, prob.scale, prob.bins, prob.vals, vals_dense)
    @test isapprox(xi_back, xi; rtol = 1e-10, atol = 1e-12)
end

@testset "generate_dense_logdet (f64)" begin
    prob, _ = load_problem("small"; T = Float64)
    n0 = prob.n0
    coords_n0 = prob.coords[:, 1:n0]

    ld = generate_dense_logdet(coords_n0, prob.scale, prob.bins, prob.vals, n0)
    @test isfinite(ld)
    @test ld < 0  # log-std of a variance<1 covariance is negative

    # Cross-check against LinearAlgebra.logdet of the assembled matrix.
    D = size(coords_n0, 1)
    T = prob.scale |> typeof
    nb = length(prob.bins)
    K = zeros(Float64, n0, n0)
    for j in 1:n0
        K[j, j] = cov_lookup(zero(Float64), prob.bins, prob.vals, nb)
        for i in j+1:n0
            sq = zero(Int64)
            for d in 1:D
                di = Int64(coords_n0[d, i]) - Int64(coords_n0[d, j])
                sq += di * di
            end
            r = sqrt(Float64(sq)) * prob.scale
            v = cov_lookup(r, prob.bins, prob.vals, nb)
            K[i, j] = v
            K[j, i] = v
        end
    end
    ld_ref = 0.5 * LinearAlgebra.logdet(LinearAlgebra.Symmetric(K))
    @test isapprox(ld, ld_ref; rtol = 1e-8)
end

@testset "generate_dense f32 smoke" begin
    prob, _ = load_problem("small"; T = Float32)
    n0 = prob.n0
    coords_n0 = prob.coords[:, 1:n0]
    rng = Random.MersenneTwister(17)
    xi = Float32.(randn(rng, n0))
    vals_dense = generate_dense(coords_n0, prob.scale, prob.bins, prob.vals, xi)
    @test !any(isnan, vals_dense)
    xi_back = generate_dense_inv(coords_n0, prob.scale, prob.bins, prob.vals, vals_dense)
    @test isapprox(Float64.(xi_back), Float64.(xi); rtol = 1e-4, atol = 1e-5)
end
