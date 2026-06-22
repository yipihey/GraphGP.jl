"""Dump JAX reference inputs/outputs for the Julia GraphGP kernels.

Builds a GraphGP problem with the existing Python/JAX code, quantizes the points onto a
21-bit-per-axis integer lattice (the representation the Julia kernels consume), and dumps:

  * inputs   : integer lattice coords (tree order), neighbors, offsets, scale, cov bins/vals
  * f64 oracle : refine_logdet / refine_inv computed in float64 on the dequantized lattice
  * f32 ref    : the same computed in float32 (apples-to-apples target for Julia f32 kernels)

Run from the repo root:  python julia/GraphGP/test/dump_reference.py
"""

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import graphgp as gp

BITS = 21
LMAX = (1 << BITS) - 1
OUTDIR = os.path.join(os.path.dirname(__file__), "reference")


def quantize(points):
    """Map float points onto an isotropic integer lattice. Returns (coords_uint32, origin, scale)."""
    points = np.asarray(points, dtype=np.float64)
    origin = points.min(axis=0)
    extent = float((points.max(axis=0) - origin).max())
    scale = extent / LMAX
    coords = np.rint((points - origin) / scale).astype(np.int64)
    coords = np.clip(coords, 0, LMAX).astype(np.uint32)
    return coords, origin, scale


def build_case(name, *, n_points, n_dim, n0, k, seed=137):
    key = jr.key(seed)
    kp, kx = jr.split(key)
    points = jr.normal(kp, (n_points, n_dim), dtype=jnp.float64)

    # Quantize, then dequantize so JAX and Julia evaluate *identical* lattice points.
    coords0, origin, scale = quantize(points)
    points_q = origin + scale * coords0.astype(np.float64)

    graph = gp.build_graph(jnp.asarray(points_q), n0=n0, k=k)
    # Recover integer coords in tree/depth order (exact: points are permuted, not modified).
    coords = np.rint((np.asarray(graph.points) - origin) / scale).astype(np.uint32)

    N = coords.shape[0]
    M = N - n0
    neighbors = np.asarray(graph.neighbors, dtype=np.int64)  # (M, k), 0-based
    offsets = np.asarray(graph.offsets, dtype=np.int64)

    cov_bins64, cov_vals64 = gp.extras.rbf_kernel(
        variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=1000, jitter=1e-3
    )
    covariance64 = (cov_bins64, cov_vals64)

    # Random values for the inverse test.
    values64 = jr.normal(kx, (N,), dtype=jnp.float64)

    out = {
        "coords": coords,  # (N, d) uint32, tree order
        "neighbors": neighbors,  # (M, k) int64, 0-based
        "offsets": offsets,  # (B,) int64
        "n0": np.int64(n0),
        "scale": np.float64(scale),
        "cov_bins64": np.asarray(cov_bins64, dtype=np.float64),
        "cov_vals64": np.asarray(cov_vals64, dtype=np.float64),
        "cov_bins32": np.asarray(cov_bins64, dtype=np.float32),
        "cov_vals32": np.asarray(cov_vals64, dtype=np.float32),
        "values64": np.asarray(values64, dtype=np.float64),
        "values32": np.asarray(values64, dtype=np.float32),
    }

    # f64 oracle
    ld64 = gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, covariance64)
    _, xi64 = gp.refine_inv(graph.points, graph.neighbors, graph.offsets, covariance64, values64)
    out["logdet64"] = np.float64(ld64)
    out["xi64"] = np.asarray(xi64, dtype=np.float64)

    # f32 reference: same algorithm, float32 throughout.
    points32 = jnp.asarray(graph.points, dtype=jnp.float32)
    cov32 = (jnp.asarray(cov_bins64, dtype=jnp.float32), jnp.asarray(cov_vals64, dtype=jnp.float32))
    values32 = jnp.asarray(values64, dtype=jnp.float32)
    ld32 = gp.refine_logdet(points32, graph.neighbors, graph.offsets, cov32)
    _, xi32 = gp.refine_inv(points32, graph.neighbors, graph.offsets, cov32, values32)
    assert xi32.dtype == jnp.float32, f"expected f32, got {xi32.dtype}"
    out["logdet32"] = np.float32(ld32)
    out["xi32"] = np.asarray(xi32, dtype=np.float32)

    # Gradient of logdet w.r.t. cov_vals (f64 oracle): the training-relevant derivative.
    def logdet_of_vals(vals):
        return gp.refine_logdet(graph.points, graph.neighbors, graph.offsets, (cov_bins64, vals))

    grad_vals64 = jax.grad(logdet_of_vals)(cov_vals64)
    out["grad_logdet_vals64"] = np.asarray(grad_vals64, dtype=np.float64)

    # Gradient of the inverse-half loss 0.5*||xi||^2 w.r.t. cov_vals (f64 oracle).
    def inv_loss_of_vals(vals):
        _, xi = gp.refine_inv(graph.points, graph.neighbors, graph.offsets, (cov_bins64, vals), values64)
        return 0.5 * jnp.sum(xi ** 2)

    inv_loss = inv_loss_of_vals(cov_vals64)
    grad_inv_vals64 = jax.grad(inv_loss_of_vals)(cov_vals64)
    out["inv_loss64"] = np.float64(inv_loss)
    out["grad_inv_loss_vals64"] = np.asarray(grad_inv_vals64, dtype=np.float64)

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{name}.npz")
    np.savez(path, **out)
    print(f"wrote {path}: N={N} M={M} k={k} d={n_dim} scale={scale:.3e} "
          f"logdet64={float(ld64):.6f} logdet32={float(ld32):.6f}")


if __name__ == "__main__":
    build_case("small", n_points=1000, n_dim=3, n0=100, k=10, seed=137)
    build_case("medium", n_points=20000, n_dim=2, n0=200, k=10, seed=99)
