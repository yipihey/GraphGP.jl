# Live Python-parity check for generate / generate_inv with a NON-IDENTITY indices.
# Opt-in (needs the fixture from python_parity_generate.py); not part of the default suite.
#
#   python test/python_parity_generate.py test/reference/generate_parity.npz       # graphgp env
#   julia --project=bench test/python_parity_generate.jl \
#       test/reference/generate_parity.npz                                          # needs NPZ
using GraphGP, NPZ, Printf

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "reference", "generate_parity.npz")
d = npzread(path)

coords = permutedims(UInt32.(d["coords"]))            # (N,d) -> (d,N), tree order
neighbors = permutedims(Int.(d["neighbors"])) .+ 1    # (M,k) 0-based -> (k,M) 1-based
offsets = Int.(d["offsets"])
n0 = Int(d["n0"]); scale = Float64(d["scale"])
indices = Int.(d["indices"]) .+ 1                     # 0-based tree→orig -> 1-based
bins = Float64.(d["cov_bins64"]); vals = Float64.(d["cov_vals64"])
prob = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals, indices)

xi = Float64.(d["xi"]); values = Float64.(d["values"])
field_py = Float64.(d["generate_field"]); xi_py = Float64.(d["generate_inv_xi"])

field_jl = generate(prob, xi)                         # ORIGINAL order in/out
xi_jl = generate_inv(prob, values)                   # ORIGINAL order in/out

relerr(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), 1e-30)
e_gen = relerr(field_jl, field_py)
e_inv = relerr(xi_jl, xi_py)
e_round = relerr(generate_inv(prob, field_jl), xi)

@printf("N=%d  non-identity indices=%s\n", length(xi), indices != collect(1:length(xi)))
@printf("  generate     vs Python: norm-relerr = %.2e\n", e_gen)
@printf("  generate_inv vs Python: norm-relerr = %.2e\n", e_inv)
@printf("  roundtrip generate_inv∘generate (original order): norm-relerr = %.2e\n", e_round)
ok = e_gen < 1e-12 && e_inv < 1e-12 && e_round < 1e-10
println(ok ? "PARITY PASS" : "PARITY FAIL")
exit(ok ? 0 : 1)
