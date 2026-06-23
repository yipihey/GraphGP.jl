@testset "compute_depths vs reference" begin
    # Use the JAX-built neighbors from the small fixture and verify compute_depths
    # produces valid (non-decreasing, causal) depths.
    d = npzread(joinpath(REFDIR, "small.npz"))
    neighbors_raw = permutedims(Int.(d["neighbors"])) .+ 1   # (k, M) 1-based
    n0 = Int(d["n0"])
    offsets = Int.(d["offsets"])

    depths = compute_depths(neighbors_raw, n0)
    @test length(depths) == n0 + size(neighbors_raw, 2)
    @test all(depths[1:n0] .== 0)
    # Each refined point's depth > max(depths[neighbors]).
    k, M = size(neighbors_raw)
    for m in 1:M
        max_nb = maximum(depths[neighbors_raw[ki, m]] for ki in 1:k if neighbors_raw[ki, m] > 0)
        @test depths[n0 + m] == max_nb + 1
    end
    # Depths are sorted (non-decreasing) after accounting for offsets.
    # The fixture is already in depth order, so depths should be non-decreasing.
    @test issorted(depths)
end

@testset "query_preceding_neighbors: all neighbors precede their point" begin
    # Build a small random k-d tree and check the causal invariant.
    rng = Random.MersenneTwister(42)
    N, D, n0, k = 200, 3, 20, 5
    pts = randn(rng, N, D)
    sorted_pts, seg_lo, seg_hi, split_dim, perm = build_tree(pts)
    neighbors = query_preceding_neighbors(sorted_pts, seg_lo, seg_hi, split_dim, n0, k)
    M = N - n0
    @test size(neighbors) == (k, M)
    for m in 1:M
        for ki in 1:k
            nb = neighbors[ki, m]
            nb > 0 || continue
            @test nb < n0 + m  # neighbor must precede this point in tree order
        end
    end
end

@testset "query_preceding_neighbors_ka (CPU backend) matches scalar reference" begin
    rng = Random.MersenneTwister(123)
    for (N, D, n0, k) in ((500, 3, 20, 10), (1200, 2, 40, 8))
        pts = randn(rng, N, D)
        sorted_pts, seg_lo, seg_hi, split_dim, _ = build_tree(pts)
        nb_ref = query_preceding_neighbors(sorted_pts, seg_lo, seg_hi, split_dim, n0, k)
        spts = permutedims(sorted_pts)  # (D, N)
        nb_ka = query_preceding_neighbors_ka(spts, seg_lo, seg_hi, split_dim, n0, k)
        @test size(nb_ka) == (k, N - n0)
        # Same k-NN *set* per query point (heap tie-order may differ).
        for m in 1:(N - n0)
            @test Set(nb_ka[:, m]) == Set(nb_ref[:, m])
        end
    end
end

@testset "build_graph smoke: valid GP structure" begin
    # Build a graph from scratch, run refine_logdet, check it's finite.
    rng = Random.MersenneTwister(99)
    N, D, n0, k = 300, 2, 30, 8
    pts = randn(rng, N, D)

    bins, vals = rbf_kernel(Float32(1.0f0), Float32(0.5f0), 1e-4, 1e1, 200;
        jitter = Float32(1e-3))
    vals32 = Float32.(vals)
    bins32 = Float32.(bins)

    prob = build_graph(pts, n0, k, bins32, vals32)

    @test npoints(prob) == N
    @test prob.n0 <= n0 + 1  # depth-0 count may differ slightly from n0
    @test nneighbors(prob) == k
    @test !isnothing(prob.indices)
    @test length(prob.indices) == N

    # All neighbors precede their point.
    K_nb = nneighbors(prob)
    M = nrefined(prob)
    for m in 1:M
        for ki in 1:K_nb
            nb = prob.neighbors[ki, m]
            nb > 0 || continue
            @test nb < prob.n0 + m
        end
    end

    ld = refine_logdet(prob)
    @test isfinite(ld)
end

@testset "check_graph: valid passes, malformed throws" begin
    rng = Random.MersenneTwister(7)
    N, D, n0, k = 250, 3, 25, 6
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(Float64(1.0), Float64(0.4), 1e-4, 1e1, 200; jitter = Float64(1e-3))
    prob = build_graph(pts, n0, k, bins, vals)

    @test check_graph(prob) === nothing  # a freshly built graph is valid

    # Wrong n0 (offsets[1] != n0).
    bad_n0 = GraphGPProblem(prob.coords, prob.neighbors, prob.offsets, prob.n0 + 1,
        prob.scale, prob.bins, prob.vals, prob.indices)
    @test_throws ArgumentError check_graph(bad_n0)

    # Non-monotone offsets.
    bad_off = copy(prob.offsets)
    if length(bad_off) >= 3
        bad_off[2], bad_off[3] = bad_off[3], bad_off[2]
        prob_bo = GraphGPProblem(prob.coords, prob.neighbors, bad_off, prob.n0,
            prob.scale, prob.bins, prob.vals, prob.indices)
        @test_throws ArgumentError check_graph(prob_bo)
    end

    # Break batch causality: point a neighbor of the first refined point at a later position.
    bad_nb = copy(prob.neighbors)
    bad_nb[1, 1] = N  # neighbor at the very end → not preceding / wrong batch
    prob_bn = GraphGPProblem(prob.coords, bad_nb, prob.offsets, prob.n0,
        prob.scale, prob.bins, prob.vals, prob.indices)
    @test_throws ArgumentError check_graph(prob_bn)
end

@testset "compute_cov_matrix: symmetric, correct diagonal" begin
    rng = Random.MersenneTwister(11)
    N, D, n0, k = 80, 3, 10, 5
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(Float64(2.0), Float64(0.5), 1e-4, 1e1, 300; jitter = Float64(1e-3))
    prob = build_graph(pts, n0, k, bins, vals)
    C = compute_cov_matrix(prob)
    @test size(C) == (N, N)
    @test C ≈ transpose(C)
    # Diagonal is the r=0 covariance (variance * (1 + jitter)).
    @test all(isapprox.(LinearAlgebra.diag(C), vals[1]; rtol = 1e-6))
    # Cross form matches the full form.
    coords = Array(prob.coords)
    C2 = compute_cov_matrix(coords, coords, prob.scale, Array(prob.bins), Array(prob.vals))
    @test C == C2
end

@testset "build_graph generate/generate_inv roundtrip" begin
    rng = Random.MersenneTwister(55)
    N, D, n0, k = 150, 2, 15, 6
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(Float64(1.0), Float64(0.4), 1e-4, 1e1, 100; jitter = Float64(1e-3))
    prob = build_graph(pts, n0, k, bins, vals)
    xi = randn(rng, N)
    vals_out = generate(prob, xi)
    xi_back = generate_inv(prob, vals_out)
    @test isapprox(xi_back, xi; rtol = 1e-8, atol = 1e-10)
end
