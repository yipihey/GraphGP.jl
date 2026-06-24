# Peak GPU memory + time for the DERIVATIVE OF THE FORWARD PASS (white noise xi -> correlated
# field) via GraphGP.jl's ANALYTIC adjoint generate_grad_xi — no materialised (M,k+1,k+1) tensor,
# no autodiff tape. Compare to run_gradmem.py (JAX autodiff). Reports CUDA.jl per-call device
# allocation and the device high-water (peak) usage.
#
#   julia --project=bench/compare run_gradmem.jl N [K]
using GraphGP, CUDA, KernelAbstractions, Random, Printf

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200_000
K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
D, n0 = 3, 256

rng = MersenneTwister(0)
pts = randn(rng, N, D)
bins0, vals0 = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 200; jitter = 1e-3)
bins = Float32.(bins0); vals = Float32.(vals0)            # f32 production path
prob = build_graph_ka(CuArray(pts), n0, K, CuArray(bins), CuArray(vals); backend = CUDABackend())
w = CuArray(randn(Float32, N))

dev_used() = (CUDA.total_memory() - CUDA.free_memory()) / 1e6   # MB resident on device

xi = CuArray(randn(Float32, N))

# Forward generate (baseline vs the CUDA extension's forward generate).
CUDA.reclaim(); generate(prob, xi); CUDA.reclaim()
falloc = CUDA.@allocated generate(prob, xi)
ft = CUDA.@elapsed generate(prob, xi)

# Derivatives of the forward pass (analytic adjoints; the CUDA extension can't do these at all).
CUDA.reclaim(); generate_grad_xi(prob, w); CUDA.reclaim()
galloc = CUDA.@allocated generate_grad_xi(prob, w)
gt = CUDA.@elapsed generate_grad_xi(prob, w)

CUDA.reclaim(); generate_grad_vals(prob, xi, w); CUDA.reclaim()
valloc = CUDA.@allocated generate_grad_vals(prob, xi, w)
vt = CUDA.@elapsed generate_grad_vals(prob, xi, w)

@printf("GRADMEM julia-gpu N=%d K=%d | forward: alloc=%.1f MB %.1f ms | d/dxi: alloc=%.1f MB %.1f ms | d/dcov_vals: alloc=%.1f MB %.1f ms\n",
    N, K, falloc / 1e6, 1e3ft, galloc / 1e6, 1e3gt, valloc / 1e6, 1e3vt)
