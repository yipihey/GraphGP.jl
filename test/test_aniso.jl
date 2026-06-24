# Anisotropic covariance K(Δspatial, Δz) — feature-parity with the Python fork's aniso.py.
# Self-contained checks (no live Python): the 2-D bilinear lookup, the spatial/radial split in
# the assembler (the #1 failure mode is collapsing to an isotropic 4-D distance), and an
# end-to-end aniso generate/generate_inv roundtrip. Live Python parity is python_parity_aniso.*.

@testset "anisotropic covariance" begin
    rng = Random.MersenneTwister(0xA115)

    # Reference bilinear matching map_coordinates(order=1, mode="nearest") with jnp.interp indices.
    function ref_aniso(ds, dz, sb, zb, grid)
        ns, nz = length(sb), length(zb)
        fi(d, b, n) = d <= b[1] ? 0.0 : d >= b[n] ? float(n - 1) :
            (i = searchsortedlast(b, d); (i - 1) + (d - b[i]) / (b[i + 1] - b[i]))
        fx, fz = fi(ds, sb, ns), fi(dz, zb, nz)
        i0, j0 = floor(Int, fx), floor(Int, fz)
        i1, j1 = min(i0 + 1, ns - 1), min(j0 + 1, nz - 1)
        tx, tz = fx - i0, fz - j0
        (grid[i0+1, j0+1]*(1-tx) + grid[i1+1, j0+1]*tx)*(1-tz) +
        (grid[i0+1, j1+1]*(1-tx) + grid[i1+1, j1+1]*tx)*tz
    end

    # (a) aniso_lookup vs the reference on a random non-uniform grid (Float64, ~1e-13).
    sb = Float64[0.0; sort(rand(rng, 6) .* 3)]
    zb = Float64[0.0; sort(rand(rng, 5) .* 2)]
    grid = rand(rng, length(sb), length(zb))
    for _ in 1:500
        ds = rand(rng) * 4 - 0.3        # span below 0 (clamp) through above the top bin (clamp)
        dz = rand(rng) * 2.5 - 0.2
        @test aniso_lookup(ds, dz, sb, zb, grid) ≈ ref_aniso(ds, dz, sb, zb, grid) rtol = 1e-13 atol = 1e-14
    end
    # diagonal = grid[1,1]
    @test aniso_lookup(0.0, 0.0, sb, zb, grid) == grid[1, 1]

    # (b) the assembler splits spatial (1:D-1) vs radial (D) — NOT a 4-D Euclidean collapse.
    # Two pairs with the SAME 4-D lattice distance (3) but different (Δspatial, Δz):
    #   P: spatial (3,0,0), radial 0   Q: spatial (0,0,0), radial 3.
    # A non-symmetric grid must give DIFFERENT covariances (isotropic collapse would not).
    D = 4
    alpha = 2.0
    sbins = Float64[0, 1, 2, 3, 4]
    zbins = Float64[0, 0.5, 1, 1.5, 2]
    g = [exp(-(s) - 3 * zz) for s in sbins, zz in zbins]   # depends very differently on each axis
    cov = build_anisotropic_covariance(sbins, zbins, g, alpha)
    coords = UInt32[ 0 3 0 0;     # dim1
                     0 0 0 0;     # dim2
                     0 0 0 0;     # dim3
                     0 0 0 3]     # dim4 = radial
    # cols: 1=origin, 2=spatial-shifted, 3=origin dup, 4=radial-shifted
    scale = 1.0
    Kd = GraphGP._assemble_dense_cov_aniso(coords, scale, cov, 4)
    # K[1,2]: d_spatial=3, d_z=0 → aniso_lookup(3,0).  K[1,4]: d_spatial=0, d_z=3/alpha=1.5.
    @test Kd[1, 2] ≈ aniso_lookup(3.0, 0.0, sbins, zbins, g)
    @test Kd[1, 4] ≈ aniso_lookup(0.0, 1.5, sbins, zbins, g)
    @test !isapprox(Kd[1, 2], Kd[1, 4]; rtol = 1e-3)        # the anti-isotropic-collapse assertion

    # (c) end-to-end aniso generate / generate_inv roundtrip (Float64). Build the graph on the
    # embedded points (n̂, alpha*z) and attach the AnisoCov.
    N, n0, k = 1500, 150, 8
    nhat = randn(rng, N, 3); nhat ./= sqrt.(sum(abs2, nhat; dims = 2))   # unit sky vectors
    z = rand(rng, N) .* 0.5
    points = hcat(nhat, alpha .* z)                                      # (N, 4) embedded
    sbins2 = collect(range(0.0, 2.0; length = 24))
    zbins2 = collect(range(0.0, 1.0; length = 16))
    grid2 = [exp(-(s / 0.4)^2 - (zz / 0.15)^2) for s in sbins2, zz in zbins2]
    acov = build_anisotropic_covariance(sbins2, zbins2, grid2, alpha; jitter = 1e-3)
    # Reuse build_graph for the tree/neighbors (Euclidean on embedded points); dummy iso table.
    db, dv = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 50; jitter = 1e-3)
    g0 = build_graph(points, n0, k, db, dv)
    aprob = GraphGPProblem(g0.coords, g0.neighbors, g0.offsets, g0.n0, g0.scale, acov, g0.indices)
    @test aprob.cov isa AnisoCov

    xi = randn(rng, N)
    field = generate(aprob, xi)
    @test all(isfinite, field)
    @test generate_inv(aprob, field) ≈ xi rtol = 1e-8     # exact inverse, original order
    @test isfinite(generate_logdet(aprob))               # aniso dense + refine logdet
end
