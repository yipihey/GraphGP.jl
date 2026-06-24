# Degenerate (coincident / near-coincident point) blocks must stay finite — real catalogues have
# coincident points (groups at identical sky+redshift), and build_graph_ka quantises to a 21-bit
# lattice so dense points can collide into the same cell → rank-deficient (k+1) conditional block.
# chol_lower! floors the pivot AND zeros the degenerate column (inert), and mean_vec_solve! zeros
# its conditional-mean weight, so a coincident point contributes a finite, regularised
# (near-zero-information) conditional instead of NaN. (GPU↔CPU agreement is in the GPU testset.)

@testset "degenerate (coincident-point) blocks stay finite" begin
    rng = Random.MersenneTwister(0xDEAD)
    n, D, n0, k = 5_000, 3, 128, 20
    bins = collect(Float64, range(0, 2, 200)); vals = Float64.(exp.(-(bins) .^ 1.5)); vals[1] *= 1.001

    p = rand(rng, n, D)
    for i in 1:30
        p[200 + i, :] = p[i, :]            # 30 exact duplicates → rank-deficient blocks
    end
    prob = build_graph_ka(p, n0, k, bins, vals; backend = KernelAbstractions.CPU())
    xi = randn(rng, n)
    f = generate(prob, xi)
    @test all(isfinite, f)
    @test all(isfinite, generate_inv(prob, f))
    @test all(isfinite, generate_grad_xi(prob, randn(rng, n)))
    @test isfinite(generate_logdet(prob))

    # The well-conditioned (no-duplicate) path is unchanged: exact inverse roundtrip.
    p2 = rand(rng, n, D)
    prob2 = build_graph_ka(p2, n0, k, bins, vals; backend = KernelAbstractions.CPU())
    xi2 = randn(rng, n)
    @test generate_inv(prob2, generate(prob2, xi2)) ≈ xi2 rtol = 1e-8
end
