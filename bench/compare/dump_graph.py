"""Build ONE graphgp graph and dump it so every implementation (pure-JAX CPU/GPU, the
graphgp CUDA extension, and GraphGP.jl CPU/GPU) evaluates the *identical* spatially-coherent
neighbour graph on the *identical* integer lattice. This is the shared input for the
correctness cross-check and the timing comparison.

    python dump_graph.py N [K D] [out.npz]

Points are quantised onto the 21-bit/axis integer lattice (the representation GraphGP.jl
consumes), then dequantised, so all paths see the same points. Output keys match
test/loadref.jl / bench_realgraph.jl conventions.
"""

import sys

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import graphgp as gp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
D = int(sys.argv[3]) if len(sys.argv) > 3 else 3
out = sys.argv[4] if len(sys.argv) > 4 else "graph.npz"
n0 = 1000

BITS = 21
LMAX = (1 << BITS) - 1

key = jr.key(1)
kp, kx = jr.split(key)
points = np.asarray(jr.normal(kp, (N, D), dtype=jnp.float64))
origin = points.min(axis=0)
scale = float((points.max(axis=0) - origin).max()) / LMAX
coords0 = np.clip(np.rint((points - origin) / scale), 0, LMAX).astype(np.uint32)
points_q = origin + scale * coords0.astype(np.float64)

# Build the real spatial graph (use the CUDA tree builder if available; falls back to CPU).
try:
    graph = gp.build_graph(jnp.asarray(points_q, dtype=jnp.float32), n0=n0, k=K, cuda=True)
except Exception:
    graph = gp.build_graph(jnp.asarray(points_q, dtype=jnp.float32), n0=n0, k=K)
M = N - n0

cov = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=1000, jitter=1e-3)
values = np.asarray(jr.normal(kx, (N,), dtype=jnp.float32))
coords = np.rint((np.asarray(graph.points, dtype=np.float64) - origin) / scale).astype(np.uint32)

np.savez(
    out,
    coords=coords,                                          # (N,d) uint32, tree order
    neighbors=np.asarray(graph.neighbors, dtype=np.int32),  # (M,k) 0-based
    offsets=np.asarray(graph.offsets, dtype=np.int64),
    n0=np.int64(n0),
    scale=np.float64(scale),
    cov_bins32=np.asarray(cov[0], dtype=np.float32),
    cov_vals32=np.asarray(cov[1], dtype=np.float32),
    values32=values,
)
print(f"dumped N={N} M={M} K={K} D={D} scale={scale:.6e} -> {out}")
