@testset "cov_lookup semantics" begin
    bins = Float32[0.0, 1.0, 2.0, 4.0]
    vals = Float32[1.0, 0.5, 0.2, 0.05]
    n = length(bins)

    # Boundary clamping (matches jnp.interp).
    @test cov_lookup(0.0f0, bins, vals, n) == 1.0f0
    @test cov_lookup(-3.0f0, bins, vals, n) == 1.0f0
    @test cov_lookup(4.0f0, bins, vals, n) == 0.05f0
    @test cov_lookup(99.0f0, bins, vals, n) == 0.05f0

    # Exact bin hits.
    @test cov_lookup(1.0f0, bins, vals, n) == 0.5f0
    @test cov_lookup(2.0f0, bins, vals, n) == 0.2f0

    # Linear interpolation within an interval.
    @test cov_lookup(1.5f0, bins, vals, n) ≈ 0.35f0
    @test cov_lookup(3.0f0, bins, vals, n) ≈ 0.125f0   # midpoint of [2,4] -> [0.2,0.05]

    # Reference against a brute-force linear scan on a logspaced grid.
    lbins = Float32.(vcat(0.0, exp10.(range(-4, 1; length = 199))))
    lvals = Float32.(exp.(-(Float64.(lbins) ./ 0.3) .^ 2))
    nn = length(lbins)
    function ref_interp(r, xp, fp)
        r <= xp[1] && return fp[1]
        r >= xp[end] && return fp[end]
        i = searchsortedlast(xp, r)
        t = (r - xp[i]) / (xp[i + 1] - xp[i])
        return fp[i] + t * (fp[i + 1] - fp[i])
    end
    for r in Float32.(exp10.(range(-5, 1.3; length = 500)))
        # atol floors the comparison at denormal/near-zero covariances (~1e-43), where the
        # @fastmath path flushes denormals differently; normal-range values match to ~1e-5.
        @test cov_lookup(r, lbins, lvals, nn) ≈ ref_interp(r, lbins, lvals) rtol = 1e-4 atol = 1e-20
    end
end
