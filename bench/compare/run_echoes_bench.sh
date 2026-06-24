#!/usr/bin/env bash
# Real-ECHOES-graph benchmark: dump each (survey,N) graph the field pipeline actually builds, then
# run every implementation (JAX-CPU, Julia-CPU, graphgp CUDA-ext, pure-JAX GPU, Julia GPU) on the
# IDENTICAL graph for the correctness cross-check (f64) and throughput (f32). The pure-JAX GPU path
# materialises the (M,k+1,k+1) tensor, so it is allowed to OOM at large N (k=30) — that failure is
# the headline "no-OOM" result, recorded as a boolean, not a crash.
#
#   ./run_echoes_bench.sh ["survey:N survey:N ..."]
#
# Each point goes to results/<survey>_<N>/ (graph.npz + per-path npz + timings.jsonl + report.md).
set -uo pipefail
cd "$(dirname "$0")"

PY_BUILD=${PY_BUILD:-$HOME/.venv/k3d/bin/python3}                 # echoes + graphgp (CPU build/dump)
PY_GPU=${PY_GPU:-$HOME/Projects/graphgp-julia/.venv-gpu/bin/python}  # graphgp_cuda + jax-cuda
JL=${JL:-$HOME/.juliaup/bin/julia}
PROJ=${PROJ:-..}
NT=${NT:-16}
K=${K:-30}
CORES=${CORES:-0-$((NT - 1))}
SWEEP=${1:-"boss:120000 boss:560000 boss:2400000 local:70000 local:1400000"}

jax_cpu_env="JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=$NT OMP_NUM_THREADS=$NT"

for spec in $SWEEP; do
  survey=${spec%:*}; N=${spec#*:}
  RD="results/${survey}_${N}"
  mkdir -p "$RD"; : > "$RD/timings.jsonl"
  G="$RD/graph.npz"
  echo "================ $survey  N=$N ================"
  if [ ! -f "$G" ]; then
    JAX_PLATFORMS=cpu "$PY_BUILD" dump_graph_echoes.py "$survey" "$N" "$G" "$K" || { echo "DUMP FAILED"; continue; }
  else
    echo "(reusing existing $G)"
  fi

  echo ">> JAX-CPU";   env $jax_cpu_env taskset -c "$CORES" "$PY_BUILD" run_jax.py "$G" correctness jax "$RD" 2>>"$RD/err.log"
  env $jax_cpu_env taskset -c "$CORES" "$PY_BUILD" run_jax.py "$G" timing jax "$RD" 2>>"$RD/err.log" | grep '^TIMING' >> "$RD/timings.jsonl" || true

  echo ">> Julia-CPU"; taskset -c "$CORES" "$JL" -t "$NT" --project="$PROJ" run_julia.jl "$G" correctness cpu "$RD" 2>>"$RD/err.log"
  taskset -c "$CORES" "$JL" -t "$NT" --project="$PROJ" run_julia.jl "$G" timing cpu "$RD" 2>>"$RD/err.log" | grep '^TIMING' >> "$RD/timings.jsonl" || true

  echo ">> CUDA-ext (GPU)"; JAX_PLATFORMS=cuda "$PY_GPU" run_jax.py "$G" correctness cuda "$RD" 2>>"$RD/err.log" || echo "   cuda-ext correctness failed"
  JAX_PLATFORMS=cuda "$PY_GPU" run_jax.py "$G" timing cuda "$RD" 2>>"$RD/err.log" | grep '^TIMING' >> "$RD/timings.jsonl" || echo "   cuda-ext timing failed"

  echo ">> pure-JAX GPU (may OOM @ k=30 — that's the result)"
  if JAX_PLATFORMS=cuda "$PY_GPU" run_jax.py "$G" timing jax "$RD" 2>"$RD/jaxgpu.err" | grep '^TIMING' >> "$RD/timings.jsonl"; then
    echo "   jax-gpu OK"
  else
    echo "{\"label\":\"jax-gpu\",\"M\":$((N)),\"oom\":true}" >> "$RD/timings.jsonl"
    echo "   jax-gpu OOM/failed (recorded oom=true)"; tail -1 "$RD/jaxgpu.err" 2>/dev/null
  fi

  echo ">> Julia GPU"; "$JL" -t "$NT" --project="$PROJ" run_julia.jl "$G" timing gpu "$RD" 2>>"$RD/err.log" | grep '^TIMING' >> "$RD/timings.jsonl" || echo "   julia-gpu failed"

  "$PY_BUILD" report.py "$RD" > "$RD/report.md" 2>/dev/null || true
  echo ">> wrote $RD/report.md"
done

echo "================ SWEEP DONE ================"
"$PY_BUILD" report_sweep.py results "$SWEEP" || true
