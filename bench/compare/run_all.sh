#!/usr/bin/env bash
# Reproducible, apples-to-apples comparison of the graphgp implementations on ONE shared graph:
# correctness cross-check (Float64, element-wise) + throughput (Float32). Every path consumes
# the identical dumped graph (same lattice points, same neighbours, same covariance table).
#
#   ./run_all.sh [N] [K] [D] [NTHREADS]
#
# Environment knobs:
#   PY        python with jax + graphgp        (default: python; set to your jax+graphgp venv)
#   JL        julia                            (default: julia)
#   PROJ      julia project                    (default: the bench env two levels up)
#   CORES     taskset core list for CPU runs   (default: 0-$((NTHREADS-1)))
#   PIN       ThreadPinning strategy hint       (informational; pinning done in-script if set)
#
# CPU runs are confined to CORES with taskset so JAX-CPU and Julia-CPU get the SAME cores
# (fair under machine load). GPU runs use the default CUDA device.
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-200000}; K=${2:-10}; D=${3:-3}; NT=${4:-32}
PY=${PY:-python}
JL=${JL:-julia}
PROJ=${PROJ:-..}
CORES=${CORES:-0-$((NT - 1))}
RDIR=results
mkdir -p "$RDIR"; : > "$RDIR/timings.jsonl"
GRAPH="$RDIR/graph.npz"

echo ">> dumping shared graph (N=$N K=$K D=$D)"
$PY dump_graph.py "$N" "$K" "$D" "$GRAPH"

have_gpu() { JAX_PLATFORMS=cuda $PY -c "import jax,sys; sys.exit(0 if jax.devices() else 1)" 2>/dev/null; }

run_jax() {  # $1=mode $2=path $3=platform(cpu|gpu)
  local plat=$3 pre="" blas="" jp="cuda"
  if [ "$plat" = cpu ]; then
    jp="cpu"
    pre="taskset -c $CORES"
    # Cap LAPACK/OpenBLAS threads to the core budget: JAX's Cholesky calls OpenBLAS, which
    # otherwise spawns a thread per logical core and oversubscribes XLA's own pool (-> crashes).
    blas="OPENBLAS_NUM_THREADS=$NT OMP_NUM_THREADS=$NT"
  fi
  env JAX_PLATFORMS=$jp $blas $pre $PY run_jax.py "$GRAPH" "$1" "$2" "$RDIR" \
    | { [ "$1" = timing ] && grep '^TIMING' >> "$RDIR/timings.jsonl" || cat; }
}
run_julia() { # $1=mode $2=cpu|gpu
  local pre=""
  [ "$2" = cpu ] && pre="taskset -c $CORES"
  $pre $JL -t "$NT" --project="$PROJ" run_julia.jl "$GRAPH" "$1" "$2" "$RDIR" \
    | { [ "$1" = timing ] && grep '^TIMING' >> "$RDIR/timings.jsonl" || cat; }
}

echo ">> JAX CPU";      run_jax correctness jax cpu;  run_jax timing jax cpu
echo ">> Julia CPU (-t $NT, cores $CORES)"; run_julia correctness cpu; run_julia timing cpu
if [ "${GPU:-auto}" != off ] && have_gpu; then
  echo ">> graphgp CUDA ext (GPU)"; run_jax correctness cuda gpu; run_jax timing cuda gpu || true
  echo ">> JAX pure (GPU)";        run_jax correctness jax  gpu || true; run_jax timing jax gpu || true
  echo ">> Julia GPU";             run_julia correctness gpu; run_julia timing gpu
else
  echo ">> (no JAX GPU device; skipping GPU paths)"
fi

echo; echo ">> report"; $PY report.py "$RDIR" | tee "$RDIR/report.md"
