"""Dump a Python `graphgp` ANISOTROPIC generate / generate_inv reference (aniso.py), so the
Julia drop-in can be checked element-wise in ORIGINAL point order.

    python python_parity_aniso.py [out.npz]

Same embedded points (n̂, alpha*z), same AnisotropicCovariance (a NON-trivial grid with distinct
Δθ vs Δz structure, so an accidental isotropic collapse is caught), same build_graph indices.
Run with an env that has the aniso fork (~/Projects/graphgp on branch hierarchical-chunked-generation).
"""
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import graphgp as gp

out = sys.argv[1] if len(sys.argv) > 1 else "reference/aniso_parity.npz"
N, n0, k = 2000, 200, 8
alpha = 2.0
BITS = 21
LMAX = (1 << BITS) - 1

key = jr.key(0xA115)
kn, kz, kx, kv = jr.split(key, 4)
nhat = jr.normal(kn, (N, 3), dtype=jnp.float64)
nhat = nhat / jnp.linalg.norm(nhat, axis=1, keepdims=True)          # unit sky vectors
z = jr.uniform(kz, (N,), dtype=jnp.float64) * 0.5
points = np.asarray(gp.embed_points(nhat, z, alpha))                # (N, 4): [n̂, alpha*z]

# Quantize to the integer lattice (same as the isotropic bridge), then dequantize for the build.
origin = points.min(axis=0)
scale = float((points.max(axis=0) - origin).max()) / LMAX
coords0 = np.clip(np.rint((points - origin) / scale), 0, LMAX).astype(np.uint32)
points_q = origin + scale * coords0.astype(np.float64)

graph = gp.build_graph(jnp.asarray(points_q, dtype=jnp.float64), n0=n0, k=k)

# A NON-trivial 2-D kernel: very different dependence on Δθ vs Δz (anti-collapse).
spatial_bins = jnp.asarray(np.linspace(0.0, 2.0, 24))
z_bins = jnp.asarray(np.linspace(0.0, 1.0, 16))
S, Z = jnp.meshgrid(spatial_bins, z_bins, indexing="ij")
grid = jnp.exp(-(S / 0.4) ** 2 - (Z / 0.15) ** 2)
cov = gp.build_anisotropic_covariance(spatial_bins, z_bins, grid, alpha, jitter=1e-3)

coords = np.rint((np.asarray(graph.points, dtype=np.float64) - origin) / scale).astype(np.uint32)
indices = np.asarray(graph.indices, dtype=np.int64)

xi = np.asarray(jr.normal(kx, (N,), dtype=jnp.float64))
field = np.asarray(gp.generate(graph, cov, jnp.asarray(xi)))
values = np.asarray(jr.normal(kv, (N,), dtype=jnp.float64))
xi_from_values = np.asarray(gp.generate_inv(graph, cov, jnp.asarray(values)))

assert not np.array_equal(indices, np.arange(N)), "indices is identity — test would be vacuous"
np.savez(
    out,
    coords=coords, neighbors=np.asarray(graph.neighbors, dtype=np.int64),
    offsets=np.asarray(graph.offsets, dtype=np.int64), n0=np.int64(n0),
    scale=np.float64(scale), indices=indices, alpha=np.float64(alpha),
    spatial_bins=np.asarray(cov.spatial_bins), z_bins=np.asarray(cov.z_bins),
    grid=np.asarray(cov.grid),
    xi=xi, generate_field=field, values=values, generate_inv_xi=xi_from_values,
)
print(f"dumped aniso parity (N={N}, non-identity indices, non-trivial grid) -> {out}")
