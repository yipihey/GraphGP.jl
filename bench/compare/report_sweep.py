"""Collate the per-(survey,N) results subdirs into the three real-graph tables:

  Table 1  correctness vs N   — Julia/CUDA-ext vs JAX-CPU (f64), L2-rel of logdet/xi/grad
  Table 2  throughput vs N    — M pts/s for each path (refine_logdet / refine_inv / grad)
  Table 3  end-to-end / OOM   — pure-JAX-GPU OOM boolean at each N (the no-OOM headline)

    python report_sweep.py <resultsdir> "boss:120000 boss:560000 ... local:1400000"
"""
import glob
import json
import os
import sys

import numpy as np

rdir = sys.argv[1]
sweep = sys.argv[2].split() if len(sys.argv) > 2 else None

if sweep:
    points = [(s.split(":")[0], int(s.split(":")[1])) for s in sweep]
else:
    points = []
    for d in sorted(glob.glob(os.path.join(rdir, "*_*"))):
        b = os.path.basename(d)
        sv, _, n = b.rpartition("_")
        if n.isdigit():
            points.append((sv, int(n)))


def load_outs(d):
    outs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        label = os.path.splitext(os.path.basename(f))[0]
        if label == "graph":
            continue
        outs[label] = dict(np.load(f))
    return outs


def load_timings(d):
    rows = {}
    p = os.path.join(d, "timings.jsonl")
    if os.path.exists(p):
        for line in open(d and p):
            line = line.strip()
            if line.startswith("TIMING "):
                line = line[len("TIMING "):]
            if line.startswith("{"):
                r = json.loads(line)
                rows[r["label"]] = r
    return rows


def l2rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


PATHS = ["jax-cpu", "julia-cpu", "cuda-gpu", "jax-gpu", "julia-gpu"]

# ---------- Table 1: correctness vs N ----------
print("## Table 1 — correctness on real graphs (Float64, vs `jax-cpu`)\n")
print("L2-relative difference of logdet / xi / grad(cov_vals). `—` = path absent.\n")
print("| survey | N | path | logdet | xi | grad |")
print("| --- | ---: | --- | ---: | ---: | ---: |")
for sv, N in points:
    d = os.path.join(rdir, f"{sv}_{N}")
    outs = load_outs(d)
    if "jax-cpu" not in outs:
        continue
    ref = outs["jax-cpu"]
    for label in PATHS:
        if label not in outs or label == "jax-cpu":
            continue
        o = outs[label]
        ld = abs(float(o["logdet"]) - float(ref["logdet"])) / max(abs(float(ref["logdet"])), 1e-30)
        xr = l2rel(o["xi"], ref["xi"])
        gr = l2rel(o["grad_logdet_vals"], ref["grad_logdet_vals"]) \
            if ("grad_logdet_vals" in o and "grad_logdet_vals" in ref) else None
        gcell = f"{gr:.1e}" if gr is not None else "n/a"
        print(f"| {sv} | {N:,} | `{label}` | {ld:.1e} | {xr:.1e} | {gcell} |")
print()

# ---------- Table 2: throughput vs N ----------
print("## Table 2 — throughput on real graphs (Float32, M pts/s; higher better)\n")
print("Per op: refine_logdet / refine_inv / grad(cov_vals). `oom` = ran out of GPU memory.\n")
print("| survey | N | " + " | ".join(f"`{p}`" for p in PATHS) + " |")
print("| --- | ---: | " + " | ".join("---" for _ in PATHS) + " |")


def cell(r):
    if r is None:
        return "—"
    if r.get("oom"):
        return "**oom**"
    M = r.get("M")
    def mps(k):
        if k not in r or not M:
            return "n/a"
        v = M / (r[k] / 1e3) / 1e6
        return f"{v:.1f}" if v >= 1 else f"{v:.2g}"
    return f"{mps('logdet_ms')} / {mps('inv_ms')} / {mps('grad_ms')}"


for sv, N in points:
    rows = load_timings(os.path.join(rdir, f"{sv}_{N}"))
    print(f"| {sv} | {N:,} | " + " | ".join(cell(rows.get(p)) for p in PATHS) + " |")
print()

# ---------- Table 3: OOM matrix ----------
print("## Table 3 — pure-JAX-GPU out-of-memory at k=30 (the no-OOM headline)\n")
print("Pure-JAX GPU materialises the dense `(M, k+1, k+1)` tensor; Julia/CUDA-ext never do.\n")
print("| survey | N | M | pure-JAX-GPU | Julia GPU |")
print("| --- | ---: | ---: | --- | --- |")
for sv, N in points:
    rows = load_timings(os.path.join(rdir, f"{sv}_{N}"))
    jg = rows.get("jax-gpu")
    ju = rows.get("julia-gpu")
    jg_s = "OOM" if (jg and jg.get("oom")) else ("ran" if jg else "—")
    ju_s = "ran" if (ju and not ju.get("oom")) else ("OOM" if (ju and ju.get("oom")) else "—")
    M = (jg or ju or {}).get("M", N)
    print(f"| {sv} | {N:,} | {M:,} | {jg_s} | {ju_s} |")
print()
