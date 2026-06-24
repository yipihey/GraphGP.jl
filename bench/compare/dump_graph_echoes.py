"""Dump a REAL ECHOES GraphGP graph in the shared bench schema (drop-in for ``dump_graph.py``),
so every implementation (pure-JAX CPU/GPU, the graphgp CUDA ext, GraphGP.jl CPU/GPU) evaluates the
identical neighbour graph that the ECHOES field pipeline actually builds — not a synthetic point
cloud. This is the input for the real-graph correctness + throughput tables.

    python dump_graph_echoes.py <boss|local> N [out.npz] [K] [ALPHA]

Geometry (matches production embeddings):
  boss   BOSS DR12 CMASS-South *randoms* (the candidate set the field is generated on), SGC
         footprint + CMASS z-cut, subsampled to N, embedded as (n̂, α·z)  — 4D, exactly as
         twopt_density/observed_ls.generate_catalogs_from_kernel (α only sets the radial graph
         scale; it cancels from the kernel). The data galaxies enter later via weights, not here.
  local  2M++ comoving xyz (3D). For N≈catalog size the real galaxies are used; for larger N the
         genuine candidate geometry (uniform-in-volume to the 2M++ depth, ZoA gap) is sampled.

Output keys are a superset of dump_graph.py: coords/neighbors/offsets/n0/scale/cov_*/values32
PLUS ``indices`` (the build↔original permutation) so the same NPZ also feeds run_graphgp.jl.
"""
import os
import sys

import numpy as np

ECHOES = os.environ.get("ECHOES_ROOT", os.path.expanduser("~/Projects/ECHOES"))
sys.path.insert(0, ECHOES)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", False)          # topology only — precision-insensitive
import jax.numpy as jnp                              # noqa: E402
import graphgp as gp                                 # noqa: E402

BITS = 21
LMAX = (1 << BITS) - 1
H0 = 68.1                                            # km/s/Mpc (2M++ loader convention)


def _radec_to_nhat(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=1)


def _boss_points(N, alpha, rng):
    """Real CMASS-South randoms → (n̂, α·z) 4D candidate embedding."""
    from echoes.surveys.boss import _read_boss_fits, _in_sgc_footprint, SIMBIG_CUTS

    rand_path = os.path.join(ECHOES, "data", "boss", "random0_DR12v5_CMASS_South.fits.gz")
    print(f"[boss] reading randoms {rand_path} (memmap) ...", flush=True)
    ra, dec, z, *_ = _read_boss_fits(rand_path, with_weight_fkp=False)
    cuts = SIMBIG_CUTS["CMASS"]
    m = (z >= cuts["z_min"]) & (z <= cuts["z_max"]) & _in_sgc_footprint(ra, dec, cuts["dec_min"])
    ra, dec, z = ra[m], dec[m], z[m]
    print(f"[boss] {len(ra):,} randoms in CMASS-South footprint; subsampling to {N:,}", flush=True)
    if N > len(ra):
        raise SystemExit(f"requested N={N:,} exceeds available randoms {len(ra):,}")
    sel = rng.choice(len(ra), size=N, replace=False)
    nhat = _radec_to_nhat(ra[sel], dec[sel])
    return np.hstack([nhat, (alpha * z[sel])[:, None]]).astype(np.float64)   # (N,4)


def _local_points(N, rng):
    """2M++ comoving xyz (3D). Real galaxies when N≈catalog size; else uniform-in-volume candidates
    (the field-evaluation geometry) within the 2M++ depth with a Zone-of-Avoidance |b|<5° gap."""
    from echoes.surveys.twompp import read_2mpp

    c = read_2mpp(os.path.join(ECHOES, "data", "local", "2mpp", "2mpp_catalog.fits"))
    d = c.vcmb / H0                                                          # comoving Mpc
    nhat = _radec_to_nhat(c.ra, c.dec)
    xyz_gal = nhat * d[:, None]
    if N <= int(1.2 * len(d)):
        sel = rng.choice(len(d), size=min(N, len(d)), replace=N > len(d))
        print(f"[local] using {N:,} real 2M++ galaxy positions (of {len(d):,})", flush=True)
        return xyz_gal[sel].astype(np.float64)
    # uniform-in-volume candidates to R_max, ZoA gap (Galactic |b|<5° excised)
    Rmax = float(np.percentile(d, 99.5))
    print(f"[local] sampling {N:,} uniform-in-volume candidates to R={Rmax:.0f} Mpc (ZoA gap)",
          flush=True)
    pts = np.empty((N, 3))
    n = 0
    while n < N:
        m = N - n
        u = rng.uniform(size=(2 * m, 3))
        r = Rmax * u[:, 0] ** (1.0 / 3.0)
        ct = 2 * u[:, 1] - 1.0
        ph = 2 * np.pi * u[:, 2]
        st = np.sqrt(1 - ct ** 2)
        gal_b = np.degrees(np.arcsin(ct))                                   # crude |b| proxy via z-axis
        keep = np.abs(gal_b) > 5.0
        cand = np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct], axis=1)[keep]
        take = min(len(cand), m)
        pts[n:n + take] = cand[:take]
        n += take
    return pts.astype(np.float64)


def main():
    survey = sys.argv[1] if len(sys.argv) > 1 else "boss"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
    out = sys.argv[3] if len(sys.argv) > 3 else "graph.npz"
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    rng = np.random.default_rng(1)

    if survey == "boss":
        points = _boss_points(N, alpha, rng)
    elif survey == "local":
        points = _local_points(N, rng)
    else:
        raise SystemExit("survey must be 'boss' or 'local'")
    N, D = points.shape
    n0 = min(1024, N // 2)

    origin = points.min(axis=0)
    scale = float((points.max(axis=0) - origin).max()) / LMAX
    coords0 = np.clip(np.rint((points - origin) / scale), 0, LMAX).astype(np.uint32)
    points_q = origin + scale * coords0.astype(np.float64)
    print(f"[{survey}] building graph N={N:,} D={D} K={K} n0={n0} ...", flush=True)
    try:
        graph = gp.build_graph(jnp.asarray(points_q, jnp.float32), n0=n0, k=K, cuda=True)
    except Exception:
        graph = gp.build_graph(jnp.asarray(points_q, jnp.float32), n0=n0, k=K)

    coords = np.rint((np.asarray(graph.points, np.float64) - origin) / scale).astype(np.uint32)
    idx = None if graph.indices is None else np.asarray(graph.indices, np.int64)
    # kernel: stretched-exp ξ(r) = A·exp(-(r/r0)^α_k) over the embedding's distance range (the
    # production tabulate_kernel form). r0 set to ~3% of the box so neighbours span the falloff.
    box = float((points.max(axis=0) - origin).max())
    r0 = 0.03 * box
    rb = np.linspace(0.0, 0.5 * box, 1000)
    A, alpha_k = 1.0, 1.5
    vals = A * np.exp(-((rb / r0) ** alpha_k))
    vals[0] *= 1.0 + 1e-3                                                    # jitter for PD
    values = rng.standard_normal(N).astype(np.float32)

    np.savez(
        out,
        coords=coords, neighbors=np.asarray(graph.neighbors, np.int32),
        offsets=np.asarray(graph.offsets, np.int64), n0=np.int64(n0), scale=np.float64(scale),
        cov_bins32=rb.astype(np.float32), cov_vals32=vals.astype(np.float32),
        values32=values,
        **({"indices": idx} if idx is not None else {}),
    )
    M = N - n0
    print(f"dumped {survey} N={N:,} M={M:,} K={K} D={D} scale={scale:.6e} r0={r0:.3f} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
