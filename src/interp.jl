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
