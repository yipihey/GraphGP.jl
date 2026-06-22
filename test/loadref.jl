# Helpers to load the JAX reference fixtures (.npz) produced by dump_reference.py and
# build a GraphGPProblem in the layout the Julia kernels expect.

using NPZ

const REFDIR = joinpath(@__DIR__, "reference")

"""
    load_problem(name; T=Float32) -> (prob, ref)

Load fixture `<name>.npz`. Returns the `GraphGPProblem` (precision `T`) and a NamedTuple of
reference quantities (`logdet32/64`, `xi32/64`, `values`, `grad_logdet_vals64`, ...).
"""
function load_problem(name::AbstractString; T = Float32)
    d = npzread(joinpath(REFDIR, "$(name).npz"))

    coords = permutedims(UInt32.(d["coords"]))          # (N,d) -> (d,N)
    neighbors = permutedims(Int.(d["neighbors"])) .+ 1  # (M,k) 0-based -> (k,M) 1-based
    offsets = Int.(d["offsets"])
    n0 = Int(d["n0"])
    scale = T(d["scale"])

    binskey = T === Float64 ? "cov_bins64" : "cov_bins32"
    valskey = T === Float64 ? "cov_vals64" : "cov_vals32"
    bins = T.(d[binskey])
    vals = T.(d[valskey])

    prob = GraphGPProblem(coords, neighbors, offsets, n0, scale, bins, vals)

    ref = (
        logdet32 = Float32(d["logdet32"]),
        logdet64 = Float64(d["logdet64"]),
        xi32 = Float32.(d["xi32"]),
        xi64 = Float64.(d["xi64"]),
        values32 = Float32.(d["values32"]),
        values64 = Float64.(d["values64"]),
        cov_bins64 = Float64.(d["cov_bins64"]),
        cov_vals64 = Float64.(d["cov_vals64"]),
        grad_logdet_vals64 = Float64.(d["grad_logdet_vals64"]),
    )
    return prob, ref
end
