"""Time JAX `gp.build_graph` on CPU (steady-state; excludes the one-time JIT compile).

    JAX_PLATFORMS=cpu python build_bench.py [N] [K] [D]

Prints one `BUILD {...}` JSON line. Pair with build_bench.jl (run_build_compare.sh runs both
under matched cores). Needs a Python env with jax + graphgp.
"""
import sys
import time
import json

import numpy as np
import jax
import jax.numpy as jnp
import graphgp as gp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
D = int(sys.argv[3]) if len(sys.argv) > 3 else 3

pts = jnp.asarray(np.random.RandomState(1).randn(N, D).astype(np.float32))
g = gp.build_graph(pts, n0=256, k=K)
jax.block_until_ready(g.points)                          # warm + compile
t0 = time.perf_counter()
g = gp.build_graph(pts, n0=256, k=K)
jax.block_until_ready(g.points)
dt = time.perf_counter() - t0

print("BUILD " + json.dumps({"impl": "jax-cpu", "N": N, "K": K, "D": D,
                             "platform": jax.devices()[0].platform, "seconds": round(dt, 3)}))
