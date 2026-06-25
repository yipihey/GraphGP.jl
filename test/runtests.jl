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
    include("test_aniso.jl")
    include("test_degenerate.jl")
    include("test_grad_generate_vals.jl")
    include("test_graph_build.jl")
    include("test_scheme_b.jl")
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

            @testset "anisotropic generate: GPU matches CPU" begin
                rng = MersenneTwister(0xA115)
                Na, n0a, ka, alpha = 1200, 120, 8, 2.0
                nh = randn(rng, Na, 3); nh ./= sqrt.(sum(abs2, nh; dims = 2))
                za = rand(rng, Na) .* 0.5
                pts = hcat(nh, alpha .* za)
                sb = collect(range(0.0, 2.0; length = 20)); zb = collect(range(0.0, 1.0; length = 14))
                grid = [exp(-(s / 0.4)^2 - (zz / 0.15)^2) for s in sb, zz in zb]
                acov = build_anisotropic_covariance(sb, zb, grid, alpha; jitter = 1e-3)
                db, dv = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 50; jitter = 1e-3)
                g0 = build_graph(pts, n0a, ka, db, dv)
                acpu = GraphGPProblem(g0.coords, g0.neighbors, g0.offsets, g0.n0, g0.scale,
                    acov, g0.indices)
                agpu = to_backend(acpu, CUDABackend())
                @test agpu.cov isa AnisoCov
                xi = randn(rng, Na)
                fc = generate(acpu, xi)
                fg = Array(generate(agpu, CuArray(xi)))
                @test isapprox(fg, fc; rtol = 1e-3)                       # @fastmath GPU-vs-CPU
                @test isapprox(generate_logdet(agpu), generate_logdet(acpu); rtol = 1e-4)
            end

            @testset "degenerate (coincident) blocks: GPU finite + matches CPU" begin
                rng = MersenneTwister(0xDEAD)
                n, D, n0d, kd = 5_000, 3, 128, 20
                bs = collect(Float64, range(0, 2, 200)); vs = Float64.(exp.(-(bs) .^ 1.5)); vs[1] *= 1.001
                p = rand(rng, n, D)
                for i in 1:30; p[200 + i, :] = p[i, :]; end          # exact duplicates
                pc = build_graph_ka(p, n0d, kd, bs, vs; backend = CPU())
                pg = to_backend(pc, CUDABackend())
                xi = randn(rng, n)
                fc = generate(pc, xi)
                fg = Array(generate(pg, CuArray(xi)))
                @test all(isfinite, fg)                              # the reported NaN is gone
                @test isapprox(fg, fc; rtol = 1e-3)                  # finite AND correct (CPU↔GPU)
                v = randn(rng, n)
                gc = generate_grad_xi(pc, v)
                gg = Array(generate_grad_xi(pg, CuArray(v)))
                @test all(isfinite, gg)
                @test isapprox(gg, gc; rtol = 1e-3)
                @test isapprox(sum(v .* fg), sum(gg .* xi); rtol = 1e-4)   # adjoint identity (GPU)
            end

            @testset "generate_grad_vals (kernel derivative): GPU matches CPU" begin
                rng = MersenneTwister(2)
                n, Dd, n0g, kg = 3_000, 3, 128, 10
                pts = randn(rng, n, Dd)
                bs, vs = rbf_kernel(1.0, 0.3, 1e-4, 1e1, 80; jitter = 1e-2)
                pc = build_graph(pts, n0g, kg, bs, vs)
                pc32 = GraphGPProblem(pc.coords, pc.neighbors, pc.offsets, pc.n0,
                    Float32(pc.scale), Float32.(pc.bins), Float32.(pc.vals), pc.indices)
                pg = to_backend(pc32, CUDABackend())
                xi = randn(rng, Float32, n); w = randn(rng, Float32, n)
                gc = generate_grad_vals(pc32, xi, w)
                gg = Array(generate_grad_vals(pg, CuArray(xi), CuArray(w)))
                @test all(isfinite, gg)
                @test isapprox(gg, gc; rtol = 1e-3)
            end

            @testset "backend consistency check" begin
                @test_throws ArgumentError GraphGPProblem(
                    CuArray(prob_cpu.coords), prob_cpu.neighbors,  # mixed CPU/GPU
                    prob_cpu.offsets, prob_cpu.n0, prob_cpu.scale,
                    prob_cpu.bins, prob_cpu.vals)
            end

            # Optional hand-written-CUDA accelerator: only if the .so has been built
            # (csrc/build.jl). Validates the custom path matches the KA path to f32.
            customlib = get(ENV, "GRAPHGP_CUDA_LIB",
                joinpath(pkgdir(GraphGP), "csrc", "libgraphgpcapi.so"))
            if isfile(customlib)
                @testset "custom CUDA accelerator vs KA (f32)" begin
                    rng = Random.MersenneTwister(7)
                    N, D, n0c, kc = 20_000, 3, 256, 10
                    pts = randn(rng, N, D)
                    b32, v32 = rbf_kernel(1.0f0, 0.3f0, 1f-4, 1f1, 200; jitter = 1f-2)
                    pc = build_graph(pts, n0c, kc, Float32.(b32), Float32.(v32))
                    pg = to_backend(pc, CUDABackend())
                    @test isapprox(refine_logdet_custom(pg), refine_logdet(pg); rtol = 1e-4)
                    field = generate(pg, CuArray(randn(Float32, N)))
                    xi_c = Array(refine_inv_custom(pg, field))
                    xi_k = Array(refine_inv(pg, field))
                    @test sqrt(sum(abs2, xi_c .- xi_k)) / sqrt(sum(abs2, xi_k)) < 1e-4
                end

                @testset "build_graph_cuda: shallow GPU build matches CPU build_graph" begin
                    rng = Random.MersenneTwister(7)
                    N2, D2, n02, k2 = 20_000, 3, 256, 10
                    p2 = randn(rng, N2, D2)
                    b2, v2 = rbf_kernel(1.0f0, 0.3f0, 1f-4, 1f1, 200; jitter = 1f-2)
                    gc = build_graph(p2, n02, k2, Float32.(b2), Float32.(v2))      # CPU reference
                    gg = build_graph_cuda(p2, n02, k2, Float32.(b2), Float32.(v2)) # .cu GPU build
                    @test gg.coords isa CuArray
                    @test check_graph(gg) === nothing                              # valid Vecchia graph
                    @test length(gg.offsets) == length(gc.offsets)                 # same shallow depth
                    @test gg.n0 == gc.n0
                    # same graph ⇒ same logdet to f32 (built identically: float tree, then quantize)
                    @test isapprox(refine_logdet_custom(gg), refine_logdet(gc); rtol = 1e-4)
                    # usable end-to-end
                    xi2 = CuArray(randn(Float32, N2))
                    @test isapprox(Array(generate_inv(gg, generate(gg, xi2))), Array(xi2);
                        rtol = 1e-3, atol = 1e-4)
                end
            end
        end
    end
end
