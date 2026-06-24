using GraphGP
using KernelAbstractions
using LinearAlgebra
using Random
using Test
using ForwardDiff

include("loadref.jl")

@testset "GraphGP" begin
    include("test_interp.jl")
    include("test_linalg.jl")
    include("test_refine.jl")
    include("test_gradients.jl")
    include("test_extras.jl")
    include("test_dense.jl")
    include("test_orchestrate.jl")
    include("test_ordering.jl")
    include("test_graph_build.jl")
end

# GPU tests: automatically activated when CUDA is installed and a device is available.
# Mirrors the CPU test suite using CuArray inputs; validates the new hand-written adjoint
# kernels (refine_inv_loss_grad_kernel!, refine_logdet_and_grad_kernel!) against the proven
# CPU path.
let cuda_ok = false
    try
        using CUDA
        cuda_ok = CUDA.functional()
    catch
    end

    if cuda_ok
        @testset "GraphGP GPU (CUDA)" begin
            # Move a CPU GraphGPProblem to GPU; offsets and indices stay on CPU (host loops).
            function to_gpu(prob::GraphGPProblem)
                GraphGPProblem(CuArray(prob.coords), CuArray(prob.neighbors), prob.offsets,
                    prob.n0, prob.scale, CuArray(prob.bins), CuArray(prob.vals), prob.indices)
            end

            # Load in Float64 so the problem eltype matches the Float64 reference data
            # (generate_inv_loss_grad_vals requires data eltype == problem eltype).
            prob_cpu, ref = load_problem("small"; T = Float64)
            prob_gpu = to_gpu(prob_cpu)
            data64 = ref.values64
            data_gpu = CuArray(data64)

            # GPU-vs-CPU parity tolerances are looser than f64 round-off would suggest: with
            # @fastmath enabled in the shared kernels, the GPU and CPU paths fuse FMAs and
            # reassociate differently, and the Cholesky pivot clamp can fire on a marginal
            # (k+1) block on one path but not the other, so the two agree to ~1e-4, not 1e-6.
            # Correctness is anchored by the CPU-vs-JAX oracle tests (which stay tight).
            @testset "forward parity" begin
                ld_cpu = generate_logdet(prob_cpu)
                ld_gpu = generate_logdet(prob_gpu)
                @test isapprox(ld_gpu, ld_cpu, rtol = 1e-4)

                xi_cpu = refine_inv(prob_cpu, data64)
                xi_gpu = Array(refine_inv(prob_gpu, data_gpu))
                @test isapprox(xi_gpu, xi_cpu, rtol = 1e-3)
            end

            @testset "logdet grad parity (refine_logdet_grad_kernel!)" begin
                dv_cpu = generate_logdet_grad_vals(prob_cpu)
                dv_gpu = Array(generate_logdet_grad_vals(prob_gpu))
                @test isapprox(dv_gpu, dv_cpu, rtol = 1e-3)   # @fastmath GPU-vs-CPU; see note above
            end

            @testset "fused logdet+grad (refine_logdet_and_grad_kernel!)" begin
                ld_sep = generate_logdet(prob_gpu)
                dv_sep = Array(generate_logdet_grad_vals(prob_gpu))
                ld_fused, dv_fused = generate_logdet_and_grad_vals(prob_gpu)
                @test isapprox(ld_fused, ld_sep, rtol = 1e-5)
                @test isapprox(Array(dv_fused), dv_sep, rtol = 1e-5)
            end

            @testset "inv loss grad parity (refine_inv_loss_grad_kernel!)" begin
                loss_cpu, dv_cpu = generate_inv_loss_grad_vals(prob_cpu, data64)
                loss_gpu, dv_gpu = generate_inv_loss_grad_vals(prob_gpu, data_gpu)
                @test isapprox(loss_gpu, loss_cpu, rtol = 1e-4)
                @test isapprox(Array(dv_gpu), dv_cpu, rtol = 1e-3)
            end

            @testset "GPU k-NN query parity (query_preceding_neighbors_ka)" begin
                N, D, n0, k = 4000, 3, 80, 10
                pts = randn(MersenneTwister(321), N, D)
                sorted_pts, seg_lo, seg_hi, split_dim, _ = build_tree(pts)
                nb_ref = query_preceding_neighbors(sorted_pts, seg_lo, seg_hi, split_dim, n0, k)
                spts = permutedims(sorted_pts)
                nb_gpu = Array(query_preceding_neighbors_ka(
                    CuArray(spts), CuArray(seg_lo), CuArray(seg_hi), CuArray(split_dim), n0, k))
                mism = 0
                for m in 1:(N - n0)
                    Set(nb_gpu[:, m]) == Set(nb_ref[:, m]) || (mism += 1)
                end
                @test mism == 0
            end

            @testset "GPU build_tree_ka produces a valid k-d tree" begin
                function brute_preceding(spts, n0, k)
                    D, N = size(spts)
                    nb = zeros(Int, k, N - n0)
                    for m in (n0 + 1):N
                        ds = [(sum(abs2, @view(spts[:, j]) .- @view(spts[:, m])), j) for j in 1:(m - 1)]
                        sort!(ds)
                        for i in 1:k
                            nb[i, m - n0] = ds[i][2]
                        end
                    end
                    nb
                end
                N, D, n0, k = 2000, 3, 50, 8
                pts = randn(MersenneTwister(2), D, N)
                spts, lo, hi, sd, perm = build_tree_ka(CuArray(pts))   # GPU sort-based build
                @test sort(Array(perm)) == collect(1:N)
                nbq = Array(query_preceding_neighbors_ka(spts, lo, hi, sd, n0, k))
                nbb = brute_preceding(Array(spts), n0, k)
                mism = 0
                for m in 1:(N - n0)
                    Set(nbq[:, m]) == Set(nbb[:, m]) || (mism += 1)
                end
                @test mism == 0
            end

            @testset "GPU fused build_graph_ka: on-device build + roundtrip" begin
                N, D, n0, k = 5000, 3, 100, 10
                pts = randn(MersenneTwister(4), N, D)
                bins, vals = rbf_kernel(1.0, 0.4, 1e-4, 1e1, 300; jitter = 1e-3)
                prob = build_graph_ka(CuArray(pts), n0, k, CuArray(bins), CuArray(vals);
                    backend = CUDABackend())
                @test prob.coords isa CuArray            # device-resident problem
                @test check_graph(prob) === nothing
                xi = CuArray(randn(N))
                xib = Array(generate_inv(prob, generate(prob, xi)))
                @test isapprox(xib, Array(xi); rtol = 1e-8, atol = 1e-10)
                @test isfinite(generate_logdet(prob))
            end

            @testset "GPU point gradients (atomic scatter) match CPU host" begin
                rng = MersenneTwister(7)
                N, D, n0, k = 4000, 3, 80, 10
                pts = randn(rng, N, D)
                bins, vals = rbf_kernel(1.0, 0.4, 1e-4, 1e1, 300; jitter = 1e-3)
                prob = build_graph(pts, n0, k, bins, vals)
                pg = to_backend(prob, CUDABackend())
                data = randn(rng, N)
                # GPU-vs-CPU point gradients diverge under @fastmath (see forward-parity note);
                # ~1e-4 agreement, anchored by the CPU-vs-JAX point-grad oracle tests.
                @test isapprox(Array(generate_logdet_grad_points(pg)),
                    generate_logdet_grad_points(prob); rtol = 1e-3, atol = 1e-6)
                @test isapprox(Array(generate_inv_loss_grad_points(pg, CuArray(data))),
                    generate_inv_loss_grad_points(prob, data); rtol = 1e-3, atol = 1e-6)
            end

            @testset "GPU end-to-end: build on CPU, run on GPU (to_backend)" begin
                rng = MersenneTwister(77)
                N, D, n0, k = 1500, 3, 40, 8
                pts = randn(rng, N, D)
                bins, vals = rbf_kernel(1.0, 0.4, 1e-4, 1e1, 300; jitter = 1e-3)  # Float64
                prob = build_graph(pts, n0, k, bins, vals)
                pg = to_backend(prob, CUDABackend())

                # compute_depths parity (GPU relaxation kernel).
                @test compute_depths(prob.neighbors, prob.n0) ==
                      Array(compute_depths(CuArray(prob.neighbors), prob.n0))

                # Full generate / generate_inv roundtrip on the GPU (dense + refine + permute).
                xi = randn(rng, N)
                v_gpu = generate(pg, CuArray(xi))
                xi_back = Array(generate_inv(pg, v_gpu))
                @test isapprox(xi_back, xi; rtol = 1e-6, atol = 1e-8)

                # logdet and its gradient match the CPU build (loosened for @fastmath
                # GPU-vs-CPU divergence; see forward-parity note).
                @test isapprox(generate_logdet(pg), generate_logdet(prob); rtol = 1e-4)
                @test isapprox(Array(generate_logdet_grad_vals(pg)),
                    generate_logdet_grad_vals(prob); rtol = 1e-3)
            end

            @testset "backend consistency check" begin
                @test_throws ArgumentError GraphGPProblem(
                    CuArray(prob_cpu.coords), prob_cpu.neighbors,  # mixed CPU/GPU
                    prob_cpu.offsets, prob_cpu.n0, prob_cpu.scale,
                    prob_cpu.bins, prob_cpu.vals)
            end
        end
    end
end
