# Scheme B (partitioned coords + halo) — pure remap correctness, no MPI required.
# `_scheme_b_local` must (1) remap each owned point's neighbors so they resolve to the SAME global
# coordinates in the compacted local problem, and (2) preserve the per-point logdet/grad terms, so
# that summing over any disjoint partition of the refined columns reproduces the serial result.

@testset "scheme B: _scheme_b_local remap preserves coords, logdet, grad" begin
    rng = Random.MersenneTwister(20)
    N, D, n0, k = 2000, 3, 64, 8
    pts = randn(rng, N, D)
    bins, vals = rbf_kernel(1.0, 0.4, 1e-4, 1e1, 200; jitter = 1e-3)
    prob = build_graph(pts, n0, k, bins, vals)
    M = nrefined(prob)

    gc = Array(prob.coords)
    nb = Array(prob.neighbors)

    # Partition the refined columns into 4 disjoint contiguous chunks (correctness is
    # partition-independent; spatial order only affects halo SIZE, not the result).
    P = 4
    csz = cld(M, P)
    chunks = [collect(((p - 1) * csz + 1):min(p * csz, M)) for p in 1:P]
    @test sort(vcat(chunks...)) == collect(1:M)            # a true partition

    ld_terms_sum = 0.0
    dv_sum = zeros(Float64, length(vals))
    total_ctx = 0
    for owned in chunks
        isempty(owned) && continue
        lp, gids = GraphGP._scheme_b_local(prob.coords, prob.neighbors, n0, prob.scale,
            prob.bins, prob.vals, owned)
        lc = Array(lp.coords)
        lnb = Array(lp.neighbors)

        # (1a) every local neighbor resolves to the SAME global coordinate.
        for (j, c) in enumerate(owned), i in 1:k
            @test lc[:, lnb[i, j]] == gc[:, nb[i, c]]
        end
        # (1b) each owned refined self sits at local column lp.n0 + j with its global coordinate.
        for (j, c) in enumerate(owned)
            @test lc[:, lp.n0 + j] == gc[:, n0 + c]
        end
        # local n0 == number of context (halo+dense) points; selves follow.
        @test lp.n0 == length(gids) - length(owned)
        @test nrefined(lp) == length(owned)
        total_ctx += lp.n0

        # (2) accumulate the per-point logdet terms and the vals-gradient over this chunk.
        ld_terms_sum += sum(Float64, refine_logdet_terms(lp))
        dv_sum .+= Float64.(refine_logdet_grad_vals(lp))
    end

    # Refined logdet over the partition + the dense first layer == serial generate_logdet.
    dense_ld = generate_dense_logdet(view(prob.coords, :, 1:n0), prob.scale,
        prob.bins, prob.vals, n0)
    @test isapprox(ld_terms_sum + dense_ld, generate_logdet(prob); rtol = 1e-10)

    # Refined vals-gradient over the partition == serial refined vals-gradient.
    @test isapprox(dv_sum, Float64.(refine_logdet_grad_vals(prob)); rtol = 1e-8)

    # Sanity: the halo is real (context points exist) but bounded.
    @test total_ctx > 0
end
