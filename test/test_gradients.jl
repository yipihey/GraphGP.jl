@testset "cov_vals gradients vs JAX ($name)" for name in ("small",)
    # Differentiate in f64 (the Julia oracle path) and compare to the JAX f64 grad.
    prob, ref = load_problem(name; T = Float64)

    # Gradient of logdet w.r.t. cov_vals (hand-written adjoint kernel).
    g_ld = refine_logdet_grad_vals(prob)
    @test length(g_ld) == length(prob.vals)
    @test !any(isnan, g_ld)
    nz = findall(!=(0), ref.grad_logdet_vals64)
    @test isapprox(g_ld[nz], ref.grad_logdet_vals64[nz]; rtol = 1e-5, atol = 1e-8)

    # Hand-written adjoint must agree with the Enzyme-through-KA gradient.
    g_ld_enz = GraphGP.refine_logdet_grad_vals_enzyme(prob)
    @test isapprox(g_ld, g_ld_enz; rtol = 1e-6, atol = 1e-10)

    # Gradient of the inverse-half loss 0.5*||xi||^2 w.r.t. cov_vals.
    loss, g_inv = refine_inv_loss_grad_vals(prob, ref.values64)
    @test isapprox(loss, ref.inv_loss64; rtol = 1e-6)
    nz2 = findall(!=(0), ref.grad_inv_loss_vals64)
    @test isapprox(g_inv[nz2], ref.grad_inv_loss_vals64[nz2]; rtol = 1e-4, atol = 1e-7)
end
