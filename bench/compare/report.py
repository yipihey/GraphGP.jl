"""Aggregate the per-path correctness outputs (*.npz) and timing lines (timings.jsonl) from a
results dir into two Markdown tables: a correctness cross-check (element-wise, vs a reference
implementation) and a throughput comparison.

    python report.py <resultsdir>

Correctness is checked in Float64 against a reference path (pure-JAX CPU if present, else the
first available) so the comparison is "same answer", independent of f32 summation order.
"""

import glob
import json
import os
import sys

import numpy as np

rdir = sys.argv[1]

# ---- correctness ----
outs = {}
for f in sorted(glob.glob(os.path.join(rdir, "*.npz"))):
    label = os.path.splitext(os.path.basename(f))[0]
    if label == "graph":          # the shared input, not an output
        continue
    outs[label] = dict(np.load(f))

ref_label = next((l for l in ("jax-cpu", "jax-gpu") if l in outs), None) or (sorted(outs)[0] if outs else None)


def reldiff(a, b):
    # L2-norm relative error (robust: per-element relative error is meaningless where the
    # reference element is ~0, which xi has). Also report max-abs for scale.
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / max(nb, 1e-30)), float(np.max(np.abs(a - b)))


print("## Correctness cross-check (Float64)\n")
if ref_label:
    ref = outs[ref_label]
    print(f"Reference: `{ref_label}`. L2-norm relative / max-absolute difference vs reference:\n")
    print("| path | logdet (rel) | xi (rel, abs) | grad cov_vals (rel, abs) |")
    print("| --- | --- | --- | --- |")
    for label in sorted(outs):
        o = outs[label]
        ld_r = abs(float(o["logdet"]) - float(ref["logdet"])) / max(abs(float(ref["logdet"])), 1e-30)
        xr, xa = reldiff(o["xi"], ref["xi"])
        if "grad_logdet_vals" in o and "grad_logdet_vals" in ref:
            gr, ga = reldiff(o["grad_logdet_vals"], ref["grad_logdet_vals"])
            gcell = f"{gr:.1e}, {ga:.1e}"
        else:
            gcell = "n/a (no autodiff)"
        tag = "  *(ref)*" if label == ref_label else ""
        print(f"| `{label}`{tag} | {ld_r:.1e} | {xr:.1e}, {xa:.1e} | {gcell} |")
print()

# ---- timing ----
tpath = os.path.join(rdir, "timings.jsonl")
rows = []
if os.path.exists(tpath):
    for line in open(tpath):
        line = line.strip()
        if line.startswith("TIMING "):
            line = line[len("TIMING "):]
        if line.startswith("{"):
            rows.append(json.loads(line))

print("## Throughput (Float32, M pts/s; higher is better)\n")
if rows:
    print("| path | refine_logdet | refine_inv | grad (cov_vals) |")
    print("| --- | --- | --- | --- |")

    def mps(r, key):
        if key not in r:
            return "n/a"
        return f"{r['M'] / (r[key] / 1e3) / 1e6:.1f}"

    order = ["jax-cpu", "jax-gpu", "cuda-gpu", "julia-cpu", "julia-gpu"]
    rows.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 99)
    for r in rows:
        print(f"| `{r['label']}` | {mps(r, 'logdet_ms')} | {mps(r, 'inv_ms')} | {mps(r, 'grad_ms')} |")
print()
