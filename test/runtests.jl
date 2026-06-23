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

            prob_cpu, ref = load_problem("small")
            prob_gpu = to_gpu(prob_cpu)
            data64 = ref.values64
            data_gpu = CuArray(data64)

            @testset "forward parity" begin
                ld_cpu = generate_logdet(prob_cpu)
                ld_gpu = generate_logdet(prob_gpu)
                @test isapprox(ld_gpu, ld_cpu, rtol = 1e-5)

                xi_cpu = refine_inv(prob_cpu, data64)
                xi_gpu = Array(refine_inv(prob_gpu, data_gpu))
                @test isapprox(xi_gpu, xi_cpu, rtol = 1e-5)
            end

            @testset "logdet grad parity (refine_logdet_grad_kernel!)" begin
                dv_cpu = generate_logdet_grad_vals(prob_cpu)
                dv_gpu = Array(generate_logdet_grad_vals(prob_gpu))
                @test isapprox(dv_gpu, dv_cpu, rtol = 1e-4)
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

            @testset "backend consistency check" begin
                @test_throws ArgumentError GraphGPProblem(
                    CuArray(prob_cpu.coords), prob_cpu.neighbors,  # mixed CPU/GPU
                    prob_cpu.offsets, prob_cpu.n0, prob_cpu.scale,
                    prob_cpu.bins, prob_cpu.vals)
            end
        end
    end
end
