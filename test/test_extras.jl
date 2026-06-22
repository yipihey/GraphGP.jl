@testset "make_cov_bins" begin
    bins = GraphGP.make_cov_bins(1e-4, 1e1, 100)
    @test length(bins) == 100
    @test bins[1] == 0.0
    @test isapprox(bins[end], 1e1; rtol = 1e-10)
    @test issorted(bins)
end

@testset "rbf_kernel" begin
    bins, vals = rbf_kernel(Float64(1.0), Float64(0.3), 1e-4, 1e1, 1000; jitter = Float64(1e-3))
    @test length(vals) == 1000
    @test !any(isnan, vals)
    @test all(vals .>= 0)
    # r=0: variance * (1 + jitter)
    @test isapprox(vals[1], 1.0 * (1 + 1e-3); rtol = 1e-10)
    # Monotonically non-increasing (RBF).
    @test issorted(vals[2:end]; rev = true)
    # At large r: approaches 0.
    @test vals[end] < 1e-4

    # Float32 works.
    _, vals32 = rbf_kernel(Float32(1), Float32(0.3), 1f-4, 1f1, 100)
    @test eltype(vals32) == Float32

    # Load JAX reference and compare.
    d = npzread(joinpath(REFDIR, "small.npz"))
    bins_ref = Float64.(d["cov_bins64"])
    vals_ref = Float64.(d["cov_vals64"])
    bins_jl, vals_jl = rbf_kernel(Float64(1.0), Float64(0.3), 1e-4, 1e1, 1000; jitter = Float64(1e-3))
    @test isapprox(bins_jl, bins_ref; rtol = 1e-8)
    @test isapprox(vals_jl, vals_ref; rtol = 1e-8)
end

@testset "matern_kernel" begin
    for p in (1, 2, 3)
        bins, vals = matern_kernel(p, Float64(1.0), Float64(0.5), 1e-4, 1e1, 200)
        @test length(vals) == 200
        @test !any(isnan, vals)
        @test all(vals .>= 0)
        # At r=0 with no jitter: variance=1, x=0, polynomial(0) = 1, so vals[1]=1.
        @test isapprox(vals[1], 1.0; rtol = 1e-8)
        # Decays toward 0 for large r.
        @test vals[end] < 1e-2
    end
    # Jitter applied correctly.
    _, vals_jit = matern_kernel(1, Float64(2.0), Float64(0.5), 1e-4, 1e1, 50; jitter = Float64(0.1))
    @test isapprox(vals_jit[1], 2.0 * 1.1; rtol = 1e-10)
end

@testset "hyperparam_grad via ForwardDiff roundtrip" begin
    # Build a small problem in f64.
    prob, _ = load_problem("small"; T = Float64)

    # Define RBF kernel factory with the same parameters as the reference fixture.
    r_min, r_max, n_bins = 1e-4, 1e1, 1000
    jitter_val = 1e-3
    make_rbf = (variance, scale) -> rbf_kernel(variance, scale, r_min, r_max, n_bins;
        jitter = jitter_val)

    hyperparams = Float64[1.0, 0.3]
    g_cov = refine_logdet_grad_vals(prob)

    # hyperparam_grad should match ForwardDiff.gradient of the full chain.
    g_hyp = hyperparam_grad(g_cov, make_rbf, hyperparams)
    @test length(g_hyp) == 2
    @test !any(isnan, g_hyp)

    # Cross-check: ForwardDiff through the full refine_logdet ∘ make_rbf chain.
    function full_logdet(hp)
        _, new_vals = make_rbf(hp[1], hp[2])
        new_prob = GraphGPProblem(prob.coords, prob.neighbors, prob.offsets, prob.n0,
            prob.scale, Float64.(prob.bins), Float64.(new_vals))
        return refine_logdet(new_prob)
    end
    g_ref = ForwardDiff.gradient(full_logdet, hyperparams)
    @test isapprox(g_hyp, g_ref; rtol = 1e-8, atol = 1e-12)
end
