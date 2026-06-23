# Covariance lookup: linear interpolation on a discretized, monotonically-increasing
# radius grid. Reproduces `jnp.interp(r, cov_bins, cov_vals)` from graphgp/refine.py:
#
#   * r <= bins[1]      -> vals[1]            (bins[1] is always 0.0)
#   * r >= bins[n]      -> vals[n]
#   * otherwise         -> linear interpolation between the bracketing bins.
#
# The bins are logarithmically spaced (see `make_cov_bins` in graphgp/extras.py), so the
# bracketing interval is found with a binary search rather than a constant-stride index.
@inline function cov_lookup(r::T, bins, vals, n::Int) where {T}
    @inbounds begin
        if r <= bins[1]
            return T(vals[1])
        elseif r >= bins[n]
            return T(vals[n])
        end
        # Binary search for lo with bins[lo] <= r < bins[lo+1], lo in 1:n-1.
        lo = 1
        hi = n
        while hi - lo > 1
            mid = (lo + hi) >>> 1
            if T(bins[mid]) <= r
                lo = mid
            else
                hi = mid
            end
        end
        x0 = T(bins[lo]); x1 = T(bins[lo + 1])
        y0 = T(vals[lo]); y1 = T(vals[lo + 1])
        # Match numpy/jax slope form: y0 + (dy/dx) * (r - x0).
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (r - x0)
    end
end

# Adjoint of `cov_lookup` w.r.t. `vals`: the value at radius `r` is a linear combination of
# (at most) two `vals` entries. Returns `(lo, w_lo, w_hi)` so that
# `cov_lookup(r) = w_lo*vals[lo] + w_hi*vals[lo+1]`; a cotangent `g` on the looked-up value
# scatters as `vals[lo] += g*w_lo`, `vals[lo+1] += g*w_hi`.
# Derivative of the interpolated covariance w.r.t. the radius `r`: the slope of the bracketing
# segment (0 outside [bins[1], bins[n]]). Used for gradients w.r.t. point positions, where
# d(cov)/d(x) = cov'(r) · dr/dx.
@inline function cov_lookup_dr(r::T, bins, vals, n::Int) where {T}
    @inbounds begin
        (r <= bins[1] || r >= bins[n]) && return zero(T)
        lo = 1
        hi = n
        while hi - lo > 1
            mid = (lo + hi) >>> 1
            if T(bins[mid]) <= r
                lo = mid
            else
                hi = mid
            end
        end
        x0 = T(bins[lo]); x1 = T(bins[lo + 1])
        y0 = T(vals[lo]); y1 = T(vals[lo + 1])
        return (y1 - y0) / (x1 - x0)
    end
end

@inline function cov_lookup_weights(r::T, bins, n::Int) where {T}
    @inbounds begin
        if r <= bins[1]
            return (1, one(T), zero(T))
        elseif r >= bins[n]
            return (n - 1, zero(T), one(T))
        end
        lo = 1
        hi = n
        while hi - lo > 1
            mid = (lo + hi) >>> 1
            if T(bins[mid]) <= r
                lo = mid
            else
                hi = mid
            end
        end
        x0 = T(bins[lo]); x1 = T(bins[lo + 1])
        t = (r - x0) / (x1 - x0)
        return (lo, one(T) - t, t)
    end
end
