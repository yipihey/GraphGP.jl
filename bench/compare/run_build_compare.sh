#!/usr/bin/env bash
# build_graph wall-time: GraphGP.jl (parallel CPU) vs JAX gp.build_graph (CPU), on matched cores.
#
#   ./run_build_compare.sh [N] [K] [D] [NTHREADS]
#
# Environment knobs (same as run_all.sh):
#   PY     python with jax + graphgp   (default: python)
#   JL     julia                       (default: julia)
#   PROJ   julia project               (default: ..  = the bench env)
#   CORES  taskset core list           (default: 0-$((NTHREADS-1)))
#
# Both processes are confined to CORES with taskset and given the same thread budget, so the
# comparison is fair under machine load. The Julia build is byte-identical to gp.build_graph.
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-1000000}; K=${2:-10}; D=${3:-3}; NT=${4:-$(nproc)}
PY=${PY:-python}; JL=${JL:-julia}; PROJ=${PROJ:-..}
CORES=${CORES:-0-$((NT - 1))}

echo ">> JAX gp.build_graph (CPU, $NT cores: $CORES)"
env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS="$NT" OMP_NUM_THREADS="$NT" \
    taskset -c "$CORES" "$PY" build_bench.py "$N" "$K" "$D"

echo ">> GraphGP.jl build_graph (CPU, -t $NT, cores $CORES)"
taskset -c "$CORES" "$JL" -t "$NT" --project="$PROJ" build_bench.jl "$N" "$K" "$D"
