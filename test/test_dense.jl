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

@testset "_dense_chol_L: exact when PD, jitter fallback when non-PD" begin
    rng = Random.MersenneTwister(5)
    # Positive-definite input: must match LAPACK exactly (no jitter added).
    A = randn(rng, 30, 30)
    Kpd = A * A' + 30 * LinearAlgebra.I
    Lpd = GraphGP._dense_chol_L(Kpd)
    @test isapprox(Lpd * Lpd', Kpd; rtol = 1e-10)

    # Rank-deficient Gram (rank 3, size 40) → raw Cholesky is non-PD; the jittered fallback
    # must return a finite factor instead of throwing.
    B = randn(rng, 40, 3)
    Knpd = B * B'
    @test !LinearAlgebra.issuccess(
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Knpd, :L); check = false))
    Lnpd = GraphGP._dense_chol_L(Knpd)
    @test size(Lnpd) == (40, 40)
    @test all(isfinite, Lnpd)

    # End-to-end: a moderately ill-conditioned dense block (n0=1000) that previously threw
    # PosDefException now yields finite logdet/gradients via the fallback.
    N, D, n0, k = 4000, 3, 1000, 10
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(Float64(1.0), Float64(0.4), 1e-4, 1e1, 300; jitter = Float64(1e-3))
    prob = build_graph(pts, n0, k, bins, vals)
    @test isfinite(generate_logdet(prob))
    @test all(isfinite, generate_logdet_grad_points(prob))
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
