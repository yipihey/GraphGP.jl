"""Dump a Python `graphgp` generate / generate_inv reference with a NON-IDENTITY `indices`, so
the Julia drop-in can be checked element-wise in ORIGINAL point order.

    python python_parity_generate.py [out.npz]

`graphgp.generate(graph, cov, xi)` and `generate_inv(graph, cov, values)` are both original-order
in / original-order out. We dump the integer-lattice graph (tree order) + `graph.indices`, a
random original-order `xi` and its field, and a random original-order `values` and its xi.
Run with an env that has `graphgp` (e.g. ~/.venv-gpu). Checked by python_parity_generate.jl.
"""
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import graphgp as gp

out = sys.argv[1] if len(sys.argv) > 1 else "reference/generate_parity.npz"
N, D, n0, k = 2000, 3, 200, 8
BITS = 21
LMAX = (1 << BITS) - 1

key = jr.key(20240607)
kp, kx, kv = jr.split(key, 3)
points = np.asarray(jr.normal(kp, (N, D), dtype=jnp.float64))
origin = points.min(axis=0)
scale = float((points.max(axis=0) - origin).max()) / LMAX
coords0 = np.clip(np.rint((points - origin) / scale), 0, LMAX).astype(np.uint32)
points_q = origin + scale * coords0.astype(np.float64)

graph = gp.build_graph(jnp.asarray(points_q, dtype=jnp.float64), n0=n0, k=k)
cov = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=400, jitter=1e-3)
cov = (jnp.asarray(cov[0], jnp.float64), jnp.asarray(cov[1], jnp.float64))

# Integer coords in tree order, and the (0-based) tree→original permutation.
coords = np.rint((np.asarray(graph.points, dtype=np.float64) - origin) / scale).astype(np.uint32)
indices = np.asarray(graph.indices, dtype=np.int64)            # indices[tree_pos] = original_pos

xi = np.asarray(jr.normal(kx, (N,), dtype=jnp.float64))        # ORIGINAL order
field = np.asarray(gp.generate(graph, cov, jnp.asarray(xi)))   # ORIGINAL order
values = np.asarray(jr.normal(kv, (N,), dtype=jnp.float64))    # ORIGINAL order
xi_from_values = np.asarray(gp.generate_inv(graph, cov, jnp.asarray(values)))  # ORIGINAL order

assert not np.array_equal(indices, np.arange(N)), "indices is identity — test would be vacuous"
np.savez(
    out,
    coords=coords, neighbors=np.asarray(graph.neighbors, dtype=np.int64),
    offsets=np.asarray(graph.offsets, dtype=np.int64), n0=np.int64(n0),
    scale=np.float64(scale), indices=indices,
    cov_bins64=np.asarray(cov[0]), cov_vals64=np.asarray(cov[1]),
    xi=xi, generate_field=field, values=values, generate_inv_xi=xi_from_values,
)
print(f"dumped generate parity (N={N}, non-identity indices) -> {out}")
