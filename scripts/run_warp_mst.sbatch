#!/bin/bash
# ---------------------------------------------------------------------------
# Runs the CUDA Boruvka MST, WARP PER VERTEX version, on the graphs listed in
# "Graph selection" below (files in Graphs/egr_graphs) and writes an
# Nsight Systems (nsys) report per graph.
#
#   source     : src/Boruvka_intermediate_warp_v1.cu
#   executable : bin/mst_warp
#   idea       : one warp (32 threads) per vertex in cheapest_find_kernel, the lanes
#                share the vertex's edges and combine with a warp min reduction.
#
# Other versions: run_gpu_mst.sbatch (baseline), run_atomic_guard.sbatch,
# run_warp_mst.sbatch. ECL-MST for comparison: run_ecl_mst.sbatch.
#
# For each graph:
#   1. RUNS plain runs -> correctness check + chrono timings (no profiler overhead),
#                         one CSV row per run (take the median later)
#   2. nsys run        -> <graph>.nsys-rep (open in the Nsight Systems GUI), once per graph
#   3. nsys stats      -> per-kernel / memcpy / memset / API times as CSV
#
# "Boruvka loop" time excludes the host<->device copies, like ECL-MST's "Dev" time.
#   throughput (Medges/s) = edges / loop time / 1e6
#
# Output: reports/<job_id>/
#   boruvka_runs.csv                     one row per graph per run
# and reports/<job_id>/<graph>/
#   run<k>.txt                           program output of plain run k
#   <graph>.nsys-rep                     full timeline
#   <graph>_cuda_gpu_kern_sum.csv        time per kernel (summed over all rounds)
#   <graph>_cuda_gpu_mem_time_sum.csv    memcpy H2D/D2H + memset times
#   <graph>_cuda_api_sum.csv             host-side CUDA API times
#
# The binary is compiled on the LOGIN node (compute-node compilation is broken
# for now) with the Makefile in the project root:
#   cd ~/BTP_2026_CUDA/Boruvka_Algorithms_CUDA
#   module load cuda-12.4 gcc-12.4.0
#   make bin/mst_warp
# Don't add -DDEBUG for timing runs (DEBUG syncs after every kernel).
#
# Submit from the project root (so logs/ resolves correctly):
#   cd ~/BTP_2026_CUDA/Boruvka_Algorithms_CUDA
#   mkdir -p logs
#   sbatch scripts/run_warp_mst.sbatch
#   logs -> logs/boruvka_warp_<job_id>.out / .err
# ---------------------------------------------------------------------------

#SBATCH --job-name=boruvka_warp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                  # host RAM: big graphs + random-weight generation
#SBATCH --time=06:00:00            # all graphs, RUNS plain runs + 1 nsys run each
#SBATCH --output=logs/%x_%j.out    # logs/ must exist BEFORE sbatch (SLURM won't create it)
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
BTP_DIR="$HOME/BTP_2026_CUDA"
PROJECT_DIR="$BTP_DIR/Boruvka_Algorithms_CUDA"
GRAPH_DIR="$BTP_DIR/Graphs/egr_graphs"

BIN="$PROJECT_DIR/bin/mst_warp"
RUNS=9                             # plain runs per graph, report the median later
REPORT_DIR="$PROJECT_DIR/reports/${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
CSV="$REPORT_DIR/boruvka_runs.csv"

# ---- Environment ----------------------------------------------------------
module purge
module load cuda-12.4
module load gcc-12.4.0

echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# ---- Binary / tool checks -------------------------------------------------
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found or not executable. Compile it on the login node first." >&2
    exit 1
fi
echo "Using $BIN (built $(date -r "$BIN"))"
# which build produced this report: binary name + build time (no commas, it goes in the CSV)
BINARY_TAG="$(basename "$BIN")@$(date -r "$BIN" +%Y%m%d_%H%M%S)"

# nsys ships with the CUDA toolkit; override with NSYS=/path/to/nsys if needed.
NSYS="${NSYS:-$(command -v nsys || true)}"
if [[ -z "$NSYS" || ! -x "$NSYS" ]]; then
    echo "ERROR: nsys not found. Load the Nsight Systems module or set NSYS=/path/to/nsys." >&2
    exit 1
fi
echo "Using $NSYS ($("$NSYS" --version))"

if [[ ! -d "$GRAPH_DIR" ]]; then
    echo "ERROR: graph directory $GRAPH_DIR not found" >&2
    exit 1
fi

mkdir -p "$REPORT_DIR"
echo "Reports -> $REPORT_DIR"

# ---- Graph selection ------------------------------------------------------
# One graph per line, file name inside GRAPH_DIR. Put # in front to skip one.
graphs=(
    "2d-2e20.sym.egr"
    "amazon0601.egr"
    "as-skitter.egr"
    "citationCiteseer.egr"
    "cit-Patents.egr"
    "coPapersDBLP.egr"
    "delaunay_n24.egr"
    "europe_osm.egr"
    "in-2004.egr"
    "internet.egr"
    "r4-2e23.sym.egr"
    "rmat16.sym.egr"
    "rmat22.sym.egr"
    "soc-LiveJournal1.egr"
    "USA-road-d.NY.egr"
    "USA-road-d.USA.egr"
)

if [[ ${#graphs[@]} -eq 0 ]]; then
    echo "ERROR: graph list is empty" >&2
    exit 1
fi
echo "Running ${#graphs[@]} graphs, $RUNS runs each"

echo "graph,nodes,edges,run,binary,loop_seconds,throughput_medges_s,verify" > "$CSV"

# ---- Run ------------------------------------------------------------------
for g in "${graphs[@]}"; do
    name="$(basename "$g" .egr)"
    file="$GRAPH_DIR/$g"

    echo ""
    echo "=================================================="
    echo ">>> $name"
    echo "=================================================="

    if [[ ! -f "$file" ]]; then
        echo "!!! File $file not found, skipping"
        continue
    fi

    # .egr header = two 32-bit ints: nodes, edges
    read -r nodes edges < <(od -An -t d4 -N 8 "$file")
    echo "Nodes: $nodes  Edges: $edges"

    out="$REPORT_DIR/$name"
    mkdir -p "$out"

    # 1. Plain runs: correctness + chrono timings without profiler overhead.
    #    Keep going if one run fails (e.g. out of GPU memory).
    ok_runs=0
    for run in $(seq 1 "$RUNS"); do
        run_out="$out/run$run.txt"

        if ! "$BIN" "$file" > "$run_out" 2>&1; then
            echo "!!! run $run FAILED on $name (see $run_out)"
            continue
        fi

        # e.g. "Boruvka loop      : 0.001807 s"
        secs="$(grep -m1 '^Boruvka loop' "$run_out" | awk '{print $4}' || true)"
        if [[ -z "$secs" ]]; then
            echo "!!! run $run: no 'Boruvka loop' time found in $run_out"
            continue
        fi
        tput="$(awk -v e="$edges" -v s="$secs" 'BEGIN { printf "%.2f", (s > 0 ? e / s / 1e6 : 0) }')"
        # both weight and edge count must match the CPU reference
        if [[ "$(grep -c '(match)' "$run_out")" -eq 2 ]] && ! grep -q MISMATCH "$run_out"; then
            verify="ok"
        else
            verify="MISMATCH"
        fi

        echo "$name,$nodes,$edges,$run,$BINARY_TAG,$secs,$tput,$verify" >> "$CSV"
        printf "run %d: loop %s s  ->  %10s Medges/s  [%s]\n" "$run" "$secs" "$tput" "$verify"
        ok_runs=$((ok_runs + 1))
    done

    if [[ $ok_runs -eq 0 ]]; then
        echo "!!! all runs FAILED on $name, skipping nsys"
        continue
    fi

    # 2. Profile: trace CUDA only (no CPU sampling) to keep overhead low.
    echo "--- nsys profile ---"
    if ! "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
            --force-overwrite=true -o "$out/$name" "$BIN" "$file" > /dev/null; then
        echo "!!! nsys profile FAILED on $name"
        continue
    fi

    # 3. Summaries: CSV files for analysis + kernel/memory tables in the log.
    "$NSYS" stats --force-export=true \
        --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
        --format csv --output "$out/$name" "$out/$name.nsys-rep" > /dev/null \
        || echo "!!! nsys stats (csv) FAILED on $name"
    "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
        "$out/$name.nsys-rep" \
        || echo "!!! nsys stats FAILED on $name"
done

echo ""
echo "All graphs done at $(date)"
echo "Reports in $REPORT_DIR"
echo "Results in $CSV"
echo ""
column -s, -t < "$CSV" || cat "$CSV"
