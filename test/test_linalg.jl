using LinearAlgebra

@testset "chol_lower! / mean_vec_solve!" begin
    rng = Random.MersenneTwister(42)
    for KP1 in (2, 5, 11)
        K = KP1 - 1
        # Random SPD matrix.
        Braw = randn(rng, KP1, KP1)
        S = Braw * Braw' + KP1 * I
        S = Float64.(S)

        A = copy(S)              # chol_lower! reads/writes the lower triangle
        GraphGP.chol_lower!(A, Val(KP1))
        Lref = cholesky(Symmetric(S, :L)).L
        for i in 1:KP1, j in 1:i
            @test A[i, j] ≈ Lref[i, j] rtol = 1e-10
        end

        # mean_vec: solve L[1:K,1:K]' x = L[K+1,1:K].
        x = zeros(Float64, K)
        GraphGP.mean_vec_solve!(x, A, Val(K))
        Lblock = LowerTriangular(Lref[1:K, 1:K])
        rhs = Lref[KP1, 1:K]
        xref = Lblock' \ rhs
        @test x ≈ xref rtol = 1e-10
    end

    # Non-PD input propagates NaN (mirrors jnp.linalg.cholesky).
    Abad = Float32[1.0 2.0; 2.0 1.0]   # indefinite
    GraphGP.chol_lower!(Abad, Val(2))
    @test isnan(Abad[2, 2])
end
