"""Real-graph GPU benchmark: build ONE gp.build_graph-derived graph, benchmark the pure-JAX
and custom graphgp-cuda paths on it, and dump it for the Julia kernels so all three
implementations run the *identical* spatially-coherent neighbor graph.

    python test/bench_realgraph.py N [K D] [dump_path.npz]

The graph is built on points quantized onto the 21-bit/axis integer lattice (the
representation GraphGP.jl consumes), then dequantized, so JAX/CUDA/Julia all evaluate the
same lattice points. With a dump path, writes coords (tree order) / neighbors / offsets /
scale / cov / values for bench_realgraph.jl.
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import graphgp as gp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
D = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dump_path = sys.argv[4] if len(sys.argv) > 4 else None
n0 = 1000

BITS = 21
LMAX = (1 << BITS) - 1


def quantize(points):
    points = np.asarray(points, dtype=np.float64)
    origin = points.min(axis=0)
    extent = float((points.max(axis=0) - origin).max())
    scale = extent / LMAX
    coords = np.rint((points - origin) / scale).astype(np.int64)
    coords = np.clip(coords, 0, LMAX).astype(np.uint32)
    return coords, origin, scale


key = jr.key(1)
kp, kx = jr.split(key)
points = jr.normal(kp, (N, D), dtype=jnp.float64)
coords0, origin, scale = quantize(points)
points_q = origin + scale * coords0.astype(np.float64)

# Real spatial graph (CUDA tree builder for speed at large N).
graph = gp.build_graph(jnp.asarray(points_q, dtype=jnp.float32), n0=n0, k=K, cuda=True)
M = N - n0

cov = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=1000, jitter=1e-3)
cov = (jnp.asarray(cov[0], jnp.float32), jnp.asarray(cov[1], jnp.float32))
values = jr.normal(kx, (N,), dtype=jnp.float32)


def timeit(f, reps=5):
    jax.block_until_ready(f())
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        best = min(best, time.perf_counter() - t0)
    return best


def run(label, cuda):
    ld = jax.jit(lambda c: gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, c, cuda=cuda)) if not cuda \
        else (lambda c: gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, c, cuda=True))
    iv = jax.jit(lambda v: gp.refine_inv(graph.points, graph.neighbors, graph.offsets, cov, v, cuda=cuda)) if not cuda \
        else (lambda v: gp.refine_inv(graph.points, graph.neighbors, graph.offsets, cov, v, cuda=True))
    t = timeit(lambda: ld(cov))
    print(f"  [{label}] refine_logdet : {1e3 * t:8.2f} ms   {M / t / 1e6:7.1f} M pts/s")
    t = timeit(lambda: iv(values))
    print(f"  [{label}] refine_inv    : {1e3 * t:8.2f} ms   {M / t / 1e6:7.1f} M pts/s")


print(f"REAL-graph bench: N={N} M={M} K={K} D={D} device={jax.devices()[0]} scale={scale:.3e}")
run("graphgp-cuda (ref)", cuda=True)
if N <= 5_000_000:  # pure-JAX materializes (M,K+1,K+1); skip where it OOMs
    run("JAX (pure)", cuda=False)
else:
    print("  [JAX (pure)] skipped (materializes (M,K+1,K+1); OOM at this N)")

if dump_path:
    # Integer lattice coords in tree/depth order (exact: points permuted, not modified).
    coords = np.rint((np.asarray(graph.points, dtype=np.float64) - origin) / scale).astype(np.uint32)
    np.savez(
        dump_path,
        coords=coords,                                            # (N,d) uint32 tree order
        neighbors=np.asarray(graph.neighbors, dtype=np.int32),   # (M,k) 0-based
        offsets=np.asarray(graph.offsets, dtype=np.int64),
        n0=np.int64(n0),
        scale=np.float64(scale),
        cov_bins32=np.asarray(cov[0], dtype=np.float32),
        cov_vals32=np.asarray(cov[1], dtype=np.float32),
        values32=np.asarray(values, dtype=np.float32),
    )
    print(f"  dumped real graph -> {dump_path}")
