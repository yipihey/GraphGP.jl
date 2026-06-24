"""Run one graph (from dump_graph.py) through a graphgp JAX path and emit its outputs +
timings. Device is chosen by JAX_PLATFORMS (cpu/gpu) in the environment.

    python run_jax.py <graph.npz> <correctness|timing> <jax|cuda> <outdir>

  jax  : pure-JAX path  (refine_*(..., cuda=False)) -- materialises (M,k+1,k+1), has autodiff
  cuda : graphgp CUDA extension (cuda=True), GPU only, f32, no autodiff

correctness mode runs f64 and writes <outdir>/<label>.npz with logdet/xi/grad for the
element-wise cross-check; timing mode runs f32 (the production precision) and prints one JSON
line of milliseconds. Labels: jax-<device>, cuda-<device>.
"""

import json
import sys
import time

npz_path, mode, path, outdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
assert mode in ("correctness", "timing") and path in ("jax", "cuda")

import jax

X64 = mode == "correctness" and path == "jax"     # the CUDA ext is f32-only
jax.config.update("jax_enable_x64", X64)

import jax.numpy as jnp
import numpy as np

import graphgp as gp

dev = jax.devices()[0].platform
label = f"{path}-{dev}"
FT = jnp.float64 if X64 else jnp.float32

d = np.load(npz_path)
scale = float(d["scale"])
coords = d["coords"].astype(np.float64)
points = jnp.asarray(scale * coords, dtype=FT)            # dequantised lattice positions
neighbors = jnp.asarray(d["neighbors"].astype(np.int32))  # (M,k) 0-based
offsets = tuple(int(x) for x in d["offsets"])
bins = jnp.asarray(d["cov_bins32"], dtype=FT)
vals = jnp.asarray(d["cov_vals32"], dtype=FT)
cov = (bins, vals)
values = jnp.asarray(d["values32"], dtype=FT)
M = neighbors.shape[0]
CU = path == "cuda"


def logdet_of(vals_):
    return gp.refine_logdet(points, neighbors, offsets, (bins, vals_), cuda=CU)


def block(x):
    return jax.block_until_ready(x)


if mode == "correctness":
    logdet = float(block(logdet_of(vals)))
    xi = np.asarray(block(gp.refine_inv(points, neighbors, offsets, cov, values, cuda=CU)[1]))
    rec = dict(logdet=np.float64(logdet), xi=xi.astype(np.float64))
    if not CU:                                            # CUDA ext has no autodiff rule
        grad = np.asarray(block(jax.grad(lambda v: logdet_of(v))(vals)))
        rec["grad_logdet_vals"] = grad.astype(np.float64)
    np.savez(f"{outdir}/{label}.npz", **rec)
    print(f"{label}: correctness outputs written (logdet={logdet:.10g})")
else:
    def timeit(f, reps=5):
        block(f())
        return min((lambda t0: (block(f()), time.perf_counter() - t0)[1])(time.perf_counter())
                   for _ in range(reps))

    ld = jax.jit(logdet_of) if not CU else logdet_of
    iv = (jax.jit(lambda v: gp.refine_inv(points, neighbors, offsets, cov, v, cuda=False))
          if not CU else (lambda v: gp.refine_inv(points, neighbors, offsets, cov, v, cuda=True)))
    res = {"label": label, "M": int(M)}
    res["logdet_ms"] = 1e3 * timeit(lambda: ld(vals))
    res["inv_ms"] = 1e3 * timeit(lambda: iv(values))
    if not CU:
        gld = jax.jit(jax.grad(lambda v: logdet_of(v)))
        res["grad_ms"] = 1e3 * timeit(lambda: gld(vals), reps=3)
    print("TIMING " + json.dumps(res))
