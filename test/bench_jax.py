"""Benchmark the JAX GraphGP routines on CPU for comparison with the Julia kernels.

    python test/bench_jax.py [N] [K] [D]

Reports median wall time and throughput for refine_logdet, refine_inv and the logdet
gradient w.r.t. cov_vals, on a graph of N points (matching the Julia bench workload).
"""

import sys
import time

import jax

jax.config.update("jax_enable_x64", False)  # f32, to match the Julia default precision

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
graph = gp.build_graph(points, n0=n0, k=K)
M = N - n0
cov = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=1000, jitter=1e-3)
cov = (jnp.asarray(cov[0], jnp.float32), jnp.asarray(cov[1], jnp.float32))
values = jr.normal(kx, (N,), dtype=jnp.float32)


def timeit(f, reps=5):
    jax.block_until_ready(f())  # warmup / compile
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        best = min(best, time.perf_counter() - t0)
    return best


logdet = jax.jit(lambda c: gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, c))
inv = jax.jit(lambda v: gp.refine_inv(graph.points, graph.neighbors, graph.offsets, cov, v))
grad_ld = jax.jit(
    jax.grad(lambda vals: gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, (cov[0], vals)))
)

print(f"JAX CPU bench: N={N} M={M} K={K} D={D} device={jax.devices()[0]}")
t = timeit(lambda: logdet(cov))
print(f"  refine_logdet          : {1e3 * t:8.2f} ms   {M / t / 1e6:6.1f} M pts/s")
t = timeit(lambda: inv(values))
print(f"  refine_inv             : {1e3 * t:8.2f} ms   {M / t / 1e6:6.1f} M pts/s")
t = timeit(lambda: grad_ld(cov[1]), reps=3)
print(f"  refine_logdet_grad_vals: {1e3 * t:8.2f} ms   {M / t / 1e6:6.1f} M pts/s")
