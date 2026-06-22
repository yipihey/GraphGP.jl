@testset "refine_logdet / refine_inv vs JAX ($name)" for name in ("small", "medium")
    # f32 problem vs JAX f32 reference (apples-to-apples), and vs the f64 oracle (sanity).
    prob32, ref = load_problem(name; T = Float32)

    ld = refine_logdet(prob32)
    @test isfinite(ld)
    @test isapprox(ld, ref.logdet32; rtol = 2e-3)
    # f32 + quantization must not drift far from the true (f64) answer.
    @test isapprox(ld, ref.logdet64; rtol = 5e-3)

    xi = refine_inv(prob32, ref.values32)
    @test !any(isnan, xi)
    @test isapprox(xi, ref.xi32; rtol = 1e-2, atol = 1e-3)
    @test isapprox(Float64.(xi), ref.xi64; rtol = 2e-2, atol = 2e-3)

    # Forward generation: regenerate values from initial values + refined xi.
    v = refine(prob32, ref.initial_values32, ref.xi_ref32)
    @test !any(isnan, v)
    @test isapprox(v, ref.refine_values32; rtol = 1e-2, atol = 1e-3)
    @test isapprox(Float64.(v), ref.refine_values64; rtol = 2e-2, atol = 2e-3)

    # f64 Julia path (debugging oracle) should match the JAX f64 oracle tightly.
    prob64, _ = load_problem(name; T = Float64)
    ld64 = refine_logdet(prob64)
    @test isapprox(ld64, ref.logdet64; rtol = 1e-6)
    xi64 = refine_inv(prob64, ref.values64)
    @test isapprox(xi64, ref.xi64; rtol = 1e-6, atol = 1e-8)
    v64 = refine(prob64, ref.initial_values64, ref.xi_ref64)
    @test isapprox(v64, ref.refine_values64; rtol = 1e-6, atol = 1e-8)
end
