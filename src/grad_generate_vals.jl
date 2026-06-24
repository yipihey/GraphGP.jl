# Analytic kernel-derivative of the FORWARD pass: d/d(cov_vals) of generate(prob, xi).
#
# generate maps white noise xi -> correlated field v through (a) the dense first-layer factor
# L_dense = chol(K_dense(vals)) with v[1:n0] = L_dense·xi[1:n0], and (b) the sequential per-point
# refinement v[p] = mean_vec(vals)·v[neighbours] + std(vals)·xi[p]. Both depend on the covariance
# table `vals`. `generate_grad_vals(prob, xi, vbar)` is the vector-Jacobian product: given an
# output cotangent `vbar` (on the field, original order), it returns the cotangent on `vals`.
#
# Structure: the reverse depth-batch sweep of `generate_grad_xi` (propagating vbar to neighbours)
# ALSO yields, per point, the cotangents on mean_vec and std; those go through the same per-point
# Cholesky-vals backward as the inverse-loss path (only the seed differs), scattering into d_vals.
# The dense block uses a runtime-sized Cholesky pullback. Host/sequential like generate_grad_xi.
# Forward-only feature; mirrors no Python entry point (the fork has no field-vals adjoint either).

# Runtime-sized reverse-mode Cholesky (the loop form of `chol_pullback!`, for the n0×n0 dense
# block where a `Val(n0)` unroll is infeasible). `Lbar` holds incoming cotangents on the factor
# (lower triangle); on return `Abar` holds d(loss)/d(A[i,j]) (lower triangle). `Lbar` is consumed.
function chol_pullback_dyn!(Abar::AbstractMatrix{T}, Lbar::AbstractMatrix{T},
        L::AbstractMatrix{T}, n::Int) where {T}
    @inbounds for j in n:-1:1
        ljj = L[j, j]
        for i in n:-1:(j + 1)
            lbij = Lbar[i, j]
            tbar = lbij / ljj
            Lbar[j, j] -= lbij * L[i, j] / ljj
            Abar[i, j] = tbar
            for p in 1:(j - 1)
                Lbar[i, p] -= tbar * L[j, p]
                Lbar[j, p] -= tbar * L[i, p]
            end
        end
        sbar = Lbar[j, j] / (2 * ljj)
        Abar[j, j] = sbar
        for p in 1:(j - 1)
            Lbar[j, p] -= 2 * sbar * L[j, p]
        end
    end
    return nothing
end

# Per-point backward for the generate vals-gradient. Recomputes the (k+1) block, seeds the factor
# cotangents from (mean_vec̄, std̄), runs chol_pullback!, and scatters into `dv` via the kernel
# interpolation weights. `vbp = vbar[p]`; mean_vec̄[j] = vbp·v[neighbour_j]; std̄ = vbp·xi[p].
@inline function _accumulate_generate_point_vals!(dv, A, Abar, Lbar, zbar, mv, jc,
        coords, neighbors, scale::T, bins, vals, nb_count, n0, m, v, vbp::T, stdbar::T,
        ::Val{K}, ::Val{D}) where {T, K, D}
    _gather_joint!(jc, coords, neighbors, m, n0, Val(K), Val(D))
    assemble_cov!(A, jc, Val(K + 1), Val(D), scale, bins, vals, nb_count)
    chol_lower!(A, Val(K + 1))
    mean_vec_solve!(mv, A, Val(K))
    pivfloor = T(1.5) * sqrt(eps(T)) * A[1, 1]

    @inbounds begin
        for j in 1:(K + 1), i in 1:(K + 1)
            Lbar[i, j] = zero(T)
        end
        Lbar[K + 1, K + 1] = stdbar                       # std = L[K+1,K+1]

        # zbar ← mean_vec̄[j] = vbp·v[neighbour_j], forward-substituted through L[1:K,1:K]
        # (adjoint of mean_vec_solve!); guard the degenerate (clamped) pivots like the forward.
        for j in 1:K
            zbar[j] = vbp * v[neighbors[j, m]]
        end
        for i in 1:K
            if A[i, i] > pivfloor
                s = zbar[i]
                for p in 1:(i - 1)
                    s -= A[i, p] * zbar[p]
                end
                zbar[i] = s / A[i, i]
            else
                zbar[i] = zero(T)
            end
        end
        for j in 1:K
            Lbar[K + 1, j] += zbar[j]
            for i in 1:j
                Lbar[j, i] -= zbar[i] * mv[j]
            end
        end
    end

    chol_pullback!(Abar, Lbar, A, Val(K + 1))

    @inbounds for a in 1:(K + 1)
        dv[1] += Abar[a, a]
        for b in 1:(a - 1)
            sq = zero(Int64)
            for dd in 1:D
                di = Int64(jc[a, dd]) - Int64(jc[b, dd])
                sq += di * di
            end
            r = sqrt(T(sq)) * scale
            lo, wlo, whi = cov_lookup_weights(r, bins, nb_count)
            g = Abar[a, b]
            dv[lo] += g * wlo
            dv[lo + 1] += g * whi
        end
    end
    return nothing
end

"""
    generate_grad_vals(prob, xi, vbar; backend) -> d_vals

Vector-Jacobian product of `generate(prob, xi)` w.r.t. the covariance table `prob.vals`, for an
output cotangent `vbar` (same length/order as the field). Returns `d_vals` (length `nbins`) on
`prob`'s backend. The kernel derivative of the white-noise→field forward pass — the piece the
CUDA extension cannot provide (no autodiff). Sequential host reverse sweep (like `generate_grad_xi`).
"""
function generate_grad_vals(prob::GraphGPProblem{T}, xi::AbstractVector, vbar::AbstractVector;
        backend = KernelAbstractions.get_backend(prob)) where {T}
    n0 = prob.n0
    N = npoints(prob)
    M = nrefined(prob)
    K = nneighbors(prob)
    D = ndims_space(prob)
    nb_count = nbins(prob)

    mean_vec = KernelAbstractions.zeros(backend, T, K, M)
    std = KernelAbstractions.zeros(backend, T, M)
    refine_meanvec_std_kernel!(backend)(mean_vec, std, prob.coords, prob.neighbors, n0,
        prob.scale, prob.bins, prob.vals, nb_count, Val(K), Val(D);
        ndrange = M, workgroupsize = _wgsize(backend))
    KernelAbstractions.synchronize(backend)
    mv = Array(mean_vec)
    sd = Array(std)
    nbrs = Array(prob.neighbors)
    coords = Array(prob.coords)
    bins = Array(prob.bins)
    vals = Array(prob.vals)
    scale = prob.scale
    offs = prob.offsets

    # xi gathered original→tree (generate's input gather); v = forward field in tree order.
    # Convert inputs to the problem's eltype so the host reverse sweep is type-consistent.
    xih = T.(Array(xi))
    if prob.indices !== nothing
        xih = xih[prob.indices]
    end
    Kd = _assemble_dense_cov(coords, scale, bins, vals, n0)   # uses coords[:,1:n0]
    Ld = _dense_chol_L(Kd)
    v = zeros(T, N)
    @inbounds v[1:n0] .= Ld * @view(xih[1:n0])
    @inbounds for b in 2:length(offs)
        for p in (offs[b - 1] + 1):offs[b]
            m = p - n0
            acc = zero(T)
            for j in 1:K
                nbj = nbrs[j, m]
                nbj > 0 && (acc += mv[j, m] * v[nbj])
            end
            v[p] = acc + sd[m] * xih[p]
        end
    end

    # vbar un-scattered to tree order (adjoint of generate's output scatter out[indices]=v).
    vb = T.(Array(vbar))
    if prob.indices !== nothing
        vb = vb[prob.indices]
    end

    dv = zeros(T, nb_count)
    A = Matrix{T}(undef, K + 1, K + 1); Abar = similar(A); Lbar = similar(A)
    jc = Matrix{UInt32}(undef, K + 1, D); zbar = Vector{T}(undef, K); mvv = Vector{T}(undef, K)

    # Reverse depth-batch sweep (latest first): vb[p] finalised when reached → per-point backward,
    # then propagate vb to neighbours (earlier points).
    @inbounds for b in length(offs):-1:2
        for p in offs[b]:-1:(offs[b - 1] + 1)
            m = p - n0
            vbp = vb[p]
            _accumulate_generate_point_vals!(dv, A, Abar, Lbar, zbar, mvv, jc, coords, nbrs,
                scale, bins, vals, nb_count, n0, m, v, vbp, vbp * xih[p], Val(K), Val(D))
            for j in 1:K
                nbj = nbrs[j, m]
                nbj > 0 && (vb[nbj] += mv[j, m] * vbp)
            end
        end
    end

    # Dense first-layer backward: v[1:n0]=L_dense·xi → L̄[i,j]=vb[i]·xi[j] (i≥j); pullback → K̄;
    # scatter K̄ into d_vals through the same interpolation weights.
    Lbar_d = zeros(T, n0, n0)
    @inbounds for j in 1:n0, i in j:n0
        Lbar_d[i, j] = vb[i] * xih[j]
    end
    Kbar_d = zeros(T, n0, n0)
    chol_pullback_dyn!(Kbar_d, Lbar_d, Ld, n0)
    @inbounds for j in 1:n0
        dv[1] += Kbar_d[j, j]
        for i in (j + 1):n0
            sq = zero(Int64)
            for dd in 1:D
                di = Int64(coords[dd, i]) - Int64(coords[dd, j])
                sq += di * di
            end
            r = sqrt(T(sq)) * scale
            lo, wlo, whi = cov_lookup_weights(r, bins, nb_count)
            g = Kbar_d[i, j]
            dv[lo] += g * wlo
            dv[lo + 1] += g * whi
        end
    end

    return _move_to_backend(dv, backend)
end

"""
    generate_of_vals(prob, vals, xi) -> field

`generate` with `prob.vals` replaced by `vals`; differentiable in BOTH `vals` (the kernel
derivative of the forward pass) and `xi`. Use with an AD framework (e.g. Zygote) to obtain
`d(loss)/d(cov_vals)` (and, via your kernel `θ→vals`, `d(loss)/dθ`) through generated samples.
"""
generate_of_vals(prob::GraphGPProblem, vals::AbstractVector, xi::AbstractVector) =
    generate(_set_vals(prob, vals), xi)

function ChainRulesCore.rrule(::typeof(generate_of_vals), prob::GraphGPProblem,
        vals::AbstractVector, xi::AbstractVector)
    prob2 = _set_vals(prob, vals)
    y = generate(prob2, xi)
    function generate_of_vals_pullback(ȳ)
        v̄ = unthunk(ȳ)
        dvals = @thunk(generate_grad_vals(prob2, xi, v̄))
        x̄ = @thunk(generate_grad_xi(prob2, v̄))
        return (NoTangent(), NoTangent(), dvals, x̄)
    end
    return y, generate_of_vals_pullback
end
