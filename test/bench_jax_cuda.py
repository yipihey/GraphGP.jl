"""Benchmark the custom graphgp-cuda extension (cuda=True) — the reference implementation.

    python test/bench_jax_cuda.py [N] [K] [D]

Times refine_logdet, refine_inv and the logdet gradient w.r.t. cov_vals using the custom
CUDA kernels, matching the workload of bench_gpu.jl / bench_jax.py. f32, GPU.
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import jax.random as jr

import graphgp as gp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
D = int(sys.argv[3]) if len(sys.argv) > 3 else 3
n0 = 1000

key = jr.key(1)
kp, kx = jr.split(key)
points = jr.normal(kp, (N, D), dtype=jnp.float32)
# Build the graph with the CUDA tree builder so large-N setup is fast.
graph = gp.build_graph(points, n0=n0, k=K, cuda=True)
M = N - n0
cov = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=1000, jitter=1e-3)
cov = (jnp.asarray(cov[0], jnp.float32), jnp.asarray(cov[1], jnp.float32))
values = jr.normal(kx, (N,), dtype=jnp.float32)
offs = jnp.asarray(graph.offsets, dtype=graph.neighbors.dtype)


def timeit(f, reps=5):
    jax.block_until_ready(f())  # warmup / compile
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        best = min(best, time.perf_counter() - t0)
    return best


print(f"graphgp-cuda (reference) bench: N={N} M={M} K={K} D={D} device={jax.devices()[0]}")


def logdet(c):
    return gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, c, cuda=True)


def inv(v):
    return gp.refine_inv(graph.points, graph.neighbors, graph.offsets, cov, v, cuda=True)


t = timeit(lambda: logdet(cov))
print(f"  refine_logdet          : {1e3 * t:8.2f} ms   {M / t / 1e6:7.1f} M pts/s")
t = timeit(lambda: inv(values))
print(f"  refine_inv             : {1e3 * t:8.2f} ms   {M / t / 1e6:7.1f} M pts/s")

try:
    grad_ld = jax.jit(jax.grad(lambda vals: gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, (cov[0], vals), cuda=True)))
    t = timeit(lambda: grad_ld(cov[1]), reps=3)
    print(f"  refine_logdet_grad_vals: {1e3 * t:8.2f} ms   {M / t / 1e6:7.1f} M pts/s")
except Exception as e:
    print(f"  refine_logdet_grad_vals: FAILED ({type(e).__name__}: {str(e)[:120]})")
