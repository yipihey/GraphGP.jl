"""
Basic GP hyperparameter training in Python using graphgp + JAX.

Identical setup to train_compare.jl: same N, D, n0, k, true hyperparameters,
same random seed via numpy. Maximises the log-marginal-likelihood with
unit-gradient ascent and reports timing for comparison with Julia.

Run from repo root:
  python julia/GraphGP/examples/train_compare.py
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import graphgp
from graphgp.extras import rbf_kernel

# ── Problem setup ────────────────────────────────────────────────────────────
N, D        = 1000, 2
N0_REQ, K   = 50, 8
R_MIN, R_MAX, N_BINS = 1e-4, 5.0, 300
JITTER      = 1e-3
N_ITER      = 40
STEP_SIZE   = 0.02   # fixed step in the unit-gradient direction (gradient ascent)

TRUE_VARIANCE = 1.5
TRUE_SCALE    = 0.4

print("graphgp Python/JAX training example")
print(f"  N={N}  D={D}  n0≈{N0_REQ}  k={K}  n_bins={N_BINS}")

rng = np.random.default_rng(42)
pts = rng.random((N, D))

# Build graph once (structure fixed; only kernel vals change during training)
t0 = time.time()
cov_init = rbf_kernel(variance=TRUE_VARIANCE, scale=TRUE_SCALE,
                      r_min=R_MIN, r_max=R_MAX, n_bins=N_BINS, jitter=JITTER)
graph = graphgp.build_graph(pts, n0=N0_REQ, k=K)
t_build = time.time() - t0
n0_actual = len(graph.points) - len(graph.neighbors)
print(f"Graph build: {t_build*1000:.1f} ms  (n0={n0_actual}, N={N}, k={K})")

# Generate synthetic observations from the true GP
xi_true = rng.standard_normal(N)
data = jnp.array(graphgp.generate(graph, cov_init, jnp.array(xi_true)))
print(f"Data generated: length={len(data)}, std={float(jnp.std(data)):.3f}")

# ── Loss and gradient ─────────────────────────────────────────────────────────
# loss(hp) = -logdet(K) - 0.5*||xi||^2  =  log p(data|hp) + const
# Maximise → hp += step * g / ‖g‖

@jax.jit
def loss_fn(hp):
    variance, scale = hp[0], hp[1]
    cov = rbf_kernel(variance=variance, scale=scale,
                     r_min=R_MIN, r_max=R_MAX, n_bins=N_BINS, jitter=JITTER)
    xi  = graphgp.generate_inv(graph, cov, data)
    ld  = graphgp.generate_logdet(graph, cov)
    return -ld - 0.5 * jnp.sum(xi ** 2)

value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# Warm-up JIT
hp_init = jnp.array([1.0, 0.7])
_loss, _g = value_and_grad(hp_init)
_loss.block_until_ready()

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nTraining (unit-gradient ascent on log-likelihood, step={STEP_SIZE}):")
print(f"  Start: variance=1.0  scale=0.7   True: {TRUE_VARIANCE}  {TRUE_SCALE}")
print()
print("  iter │    loss    │ variance │  scale  │  ‖g‖    │  ms/iter")
print("  ─────┼────────────┼──────────┼─────────┼─────────┼─────────")

hp = jnp.array([1.0, 0.7])

for i in range(N_ITER):
    t0 = time.time()
    loss, g = value_and_grad(hp)
    loss.block_until_ready()
    elapsed = time.time() - t0

    gnorm = float(jnp.linalg.norm(g))
    g_unit = g / (gnorm + 1e-12)       # unit gradient direction
    hp = hp + STEP_SIZE * g_unit        # gradient ASCENT
    hp = jnp.clip(hp, 1e-3, None)

    it = i + 1
    if it == 1 or it % 5 == 0 or it == N_ITER:
        print(f"  {it:4d} │ {float(loss):10.3f} │ {float(hp[0]):8.4f} │ "
              f"{float(hp[1]):7.4f} │ {gnorm:7.3f} │ {elapsed*1000:6.1f}")

print()
print(f"  Final:  variance={float(hp[0]):.4f}  scale={float(hp[1]):.4f}")
print(f"  True:   variance={TRUE_VARIANCE:.4f}  scale={TRUE_SCALE:.4f}")
