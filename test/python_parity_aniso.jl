# Live Python-parity check for ANISOTROPIC generate / generate_inv. Opt-in (needs the fixture
# from python_parity_aniso.py); not part of the default suite.
#
#   python test/python_parity_aniso.py test/reference/aniso_parity.npz          # aniso fork env
#   julia --project=bench test/python_parity_aniso.jl \
#       test/reference/aniso_parity.npz                                          # needs NPZ
using GraphGP, NPZ, Printf

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "reference", "aniso_parity.npz")
d = npzread(path)

coords = permutedims(UInt32.(d["coords"]))            # (N,d) -> (d,N), tree order
neighbors = permutedims(Int.(d["neighbors"])) .+ 1    # (M,k) 0-based -> (k,M) 1-based
offsets = Int.(d["offsets"]); n0 = Int(d["n0"]); scale = Float64(d["scale"])
indices = Int.(d["indices"]) .+ 1                     # 0-based tree→orig -> 1-based
cov = build_anisotropic_covariance(Float64.(d["spatial_bins"]), Float64.(d["z_bins"]),
    Float64.(d["grid"]), Float64(d["alpha"]))         # grid already jittered in the dump
prob = GraphGPProblem(coords, neighbors, offsets, n0, scale, cov, indices)

xi = Float64.(d["xi"]); values = Float64.(d["values"])
field_py = Float64.(d["generate_field"]); xi_py = Float64.(d["generate_inv_xi"])

field_jl = generate(prob, xi)                         # ORIGINAL order in/out
xi_jl = generate_inv(prob, values)

relerr(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), 1e-30)
e_gen = relerr(field_jl, field_py)
e_inv = relerr(xi_jl, xi_py)
e_round = relerr(generate_inv(prob, field_jl), xi)

@printf("N=%d  anisotropic  non-identity indices=%s\n", length(xi), indices != collect(1:length(xi)))
@printf("  generate     vs Python: norm-relerr = %.2e\n", e_gen)
@printf("  generate_inv vs Python: norm-relerr = %.2e\n", e_inv)
@printf("  roundtrip generate_inv∘generate (original order): norm-relerr = %.2e\n", e_round)
ok = e_gen < 1e-11 && e_inv < 1e-11 && e_round < 1e-9
println(ok ? "ANISO PARITY PASS" : "ANISO PARITY FAIL")
exit(ok ? 0 : 1)
