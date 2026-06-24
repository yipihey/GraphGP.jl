"""Peak GPU memory of the graphgp CUDA EXTENSION (the fair baseline) on the FORWARD pass
(white noise xi -> correlated field). The extension is forward-only — it has no autodiff — so the
DERIVATIVES of the forward pass (d/dxi, d/dcov_vals) cannot be computed by it at all; those are a
GraphGP.jl-unique capability (run_gradmem.jl). We therefore baseline forward memory here and mark
derivatives as unsupported.

    python run_gradmem.py N [K]

On-demand allocator so peak is real (not XLA's 75% preallocation).
"""
import json
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import graphgp as gp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
D, n0 = 3, 256
dev = jax.devices()[0]

key = jr.key(0)
kp, kx = jr.split(key, 2)
points = jr.normal(kp, (N, D), dtype=jnp.float32)
graph = gp.build_graph(points, n0=n0, k=K, cuda=True)
cov = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=200, jitter=1e-3)
cov = (jnp.asarray(cov[0], jnp.float32), jnp.asarray(cov[1], jnp.float32))
xi = jr.normal(kx, (N,), dtype=jnp.float32)


def peak_mb():
    return (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6


# Forward generate via the CUDA extension (cuda=True). No jit needed; the ext is a custom call.
try:
    f = lambda x: gp.generate(graph, cov, x, cuda=True)
    jax.block_until_ready(f(xi))                  # warm
    t0 = time.perf_counter()
    jax.block_until_ready(f(xi))
    dt = time.perf_counter() - t0
    fwd = {"peak_mb": round(peak_mb(), 1), "ms": round(1e3 * dt, 1), "ok": True}
except Exception as e:
    fwd = {"peak_mb": None, "ms": None, "ok": False, "err": type(e).__name__}

res = {"path": "cuda-ext-gpu", "N": N, "K": K,
       "forward_generate": fwd,
       "d_dxi": "unsupported (extension has no autodiff)",
       "d_dvals": "unsupported (extension has no autodiff)"}
print("GRADMEM " + json.dumps(res))
