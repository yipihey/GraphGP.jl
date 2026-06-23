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

@testset "ChainRules / Zygote composition" begin
    using Zygote
    prob, ref = load_problem("small"; T = Float64)
    vals = Array(prob.vals)
    data = ref.values64

    # (1) Arbitrary scalar loss through the rrules, differentiated by Zygote w.r.t. cov_vals,
    #     must equal the validated analytic gradients (sum of the two likelihood terms).
    L(v) = logdet_of_vals(prob, v) + inv_quadratic_loss_of_vals(prob, v, data)
    gz = Zygote.gradient(L, vals)[1]
    g_ref = generate_logdet_grad_vals(prob) .+ generate_inv_loss_grad_vals(prob, data)[2]
    @test isapprox(gz, g_ref; rtol = 1e-8, atol = 1e-10)

    # logdet term alone matches its analytic adjoint exactly.
    gz_ld = Zygote.gradient(v -> logdet_of_vals(prob, v), vals)[1]
    @test isapprox(gz_ld, generate_logdet_grad_vals(prob); rtol = 1e-10)

    # (2) Hyperparameter chaining: a Zygote-safe scalar kernel param α scaling the cov_vals.
    #     d/dα of the loss must equal dot(analytic cov_vals grad at α·base, base).
    base = copy(vals)
    α0 = 1.0
    gα = Zygote.gradient(α -> logdet_of_vals(prob, α .* base), α0)[1]
    probα = GraphGPProblem(prob.coords, prob.neighbors, prob.offsets, prob.n0, prob.scale,
        prob.bins, α0 .* base, prob.indices)
    gα_ref = sum(generate_logdet_grad_vals(probα) .* base)
    @test isapprox(gα, gα_ref; rtol = 1e-6)
end

@testset "d/dxi VJP and generate rrule" begin
    using Zygote
    using LinearAlgebra: dot
    rng = Random.MersenneTwister(9)
    N, D, n0, k = 600, 3, 25, 8
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(Float64(1.0), Float64(0.4), 1e-4, 1e1, 200; jitter = Float64(1e-3))
    prob = build_graph(pts, n0, k, bins, vals)
    xr = randn(rng, N)
    vb = randn(rng, N)

    # generate is linear in xi → exact adjoint identity ⟨v̄, G·x⟩ = ⟨Gᵀv̄, x⟩.
    @test isapprox(dot(vb, generate(prob, xr)), dot(generate_grad_xi(prob, vb), xr); rtol = 1e-10)

    # Zygote backprop through generate matches the analytic VJP.
    gz = Zygote.gradient(x -> dot(vb, generate(prob, x)), xr)[1]
    @test isapprox(gz, generate_grad_xi(prob, vb); rtol = 1e-8)

    # Scalar loss 0.5‖generate(xi)‖²: gradient is Gᵀ(G·xi).
    gz2 = Zygote.gradient(x -> sum(abs2, generate(prob, x)) / 2, xr)[1]
    @test isapprox(gz2, generate_grad_xi(prob, generate(prob, xr)); rtol = 1e-8)
end

@testset "d/dpoints (generate_logdet) vs continuous finite differences" begin
    using GraphGP: cov_lookup
    using LinearAlgebra: norm, diag, cholesky, Symmetric
    rng = Random.MersenneTwister(13)
    N, D, n0, k = 200, 3, 16, 6
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(Float64(1.0), Float64(0.5), 1e-4, 1e1, 400; jitter = Float64(1e-3))
    nbn = length(bins)
    prob = build_graph(pts, n0, k, bins, vals)
    coords = Array(prob.coords)
    nbr = Array(prob.neighbors)
    M = nrefined(prob)
    X = prob.scale .* Float64.(coords)   # dequantized positions (D, N), tree order

    # Continuous-position forward: dense 0.5·log|K| over the first n0, plus the per-point
    # refinement log-std. Matches generate_logdet at X = scale·coords.
    function fwd(X)
        s = 0.0
        Kd = zeros(n0, n0)
        for i in 1:n0, j in 1:n0
            Kd[i, j] = cov_lookup(norm(@view(X[:, i]) .- @view(X[:, j])), bins, vals, nbn)
        end
        s += sum(log, diag(cholesky(Symmetric(Kd, :L)).L))
        A = zeros(k + 1, k + 1)
        for m in 1:M
            idx = ntuple(j -> j <= k ? nbr[j, m] : n0 + m, k + 1)
            for i in 1:(k + 1), j in 1:(k + 1)
                A[i, j] = cov_lookup(norm(@view(X[:, idx[i]]) .- @view(X[:, idx[j]])), bins, vals, nbn)
            end
            s += log(cholesky(Symmetric(A, :L)).L[k + 1, k + 1])
        end
        s
    end

    @test isapprox(fwd(X), generate_logdet(prob); rtol = 1e-10)
    gp = generate_logdet_grad_points(prob)
    @test size(gp) == (D, N)
    for (p, d) in ((3, 1), (12, 2), (n0 + 5, 1), (n0 + 60, 3), (n0 + 150, 2))
        h = 1e-6
        Xp = copy(X); Xp[d, p] += h
        Xm = copy(X); Xm[d, p] -= h
        fd = (fwd(Xp) - fwd(Xm)) / (2h)
        @test isapprox(gp[d, p], fd; rtol = 2e-3, atol = 1e-6)
    end

    # Regression: the dense first-layer logdet grad w.r.t. cov_vals (FD of generate_dense_logdet).
    cn0 = view(prob.coords, :, 1:n0)
    gv = GraphGP._dense_logdet_grad_vals(prob)
    f(v) = generate_dense_logdet(cn0, prob.scale, bins, v, n0)
    for j in findall(!=(0), gv)[1:3]
        h = 1e-6 * max(abs(vals[j]), 1e-3)
        v1 = copy(vals); v1[j] += h
        v2 = copy(vals); v2[j] -= h
        @test isapprox(gv[j], (f(v1) - f(v2)) / (2h); rtol = 2e-3, atol = 1e-6)
    end
end
