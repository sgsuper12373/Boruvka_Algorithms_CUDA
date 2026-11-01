#!/bin/bash
# ---------------------------------------------------------------------------
# Runs ECL-MST (reference GPU MST code) on the graphs listed below and records
# its throughput, for comparison with our Boruvka (run_gpu_mst.sbatch).
#
# ECL-MST prints its GPU time as "Dev+: <sec> s" (filtered variant, used when
# avg degree >= 4) or "Dev-: <sec> s" (non-filtered). That time excludes the
# host<->device copies, like our "Boruvka loop" time.
#   throughput (Medges/s) = edges / GPU time / 1e6
# "edges" = CSR edge count from the .egr header (same definition as our code).
#
# Output: reports/ecl_<job_id>/
#   ecl_throughput.csv        one row per graph per run
#   <graph>/run<k>.txt        full ECL-MST output of each run
#
# Compile on the LOGIN node (H100 = sm_90):
#   cd ~/BTP_2026_CUDA/Boruvka_Algorithms_CUDA
#   nvcc -O3 -arch=sm_90 ../ECL-MST/source/ECL-MST_10.cu -o bin/ecl_mst
#
# Submit from the project root (so logs/ resolves correctly):
#   cd ~/BTP_2026_CUDA/Boruvka_Algorithms_CUDA
#   mkdir -p logs
#   sbatch scripts/run_ecl_mst.sbatch
# ---------------------------------------------------------------------------

#SBATCH --job-name=ecl_mst
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                  # host RAM: big graphs + ECL's CPU (Kruskal) check
#SBATCH --time=04:00:00            # RUNS runs per graph, each with ECL's CPU check
#SBATCH --output=logs/%x_%j.out    # logs/ must exist BEFORE sbatch (SLURM won't create it)
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
BTP_DIR="$HOME/BTP_2026_CUDA"
PROJECT_DIR="$BTP_DIR/Boruvka_Algorithms_CUDA"
GRAPH_DIR="$BTP_DIR/Graphs/egr_graphs"

BIN="$PROJECT_DIR/bin/ecl_mst"
RUNS=9                             # runs per graph, report the median later
REPORT_DIR="$PROJECT_DIR/reports/ecl_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
CSV="$REPORT_DIR/ecl_throughput.csv"

# ---- Environment ----------------------------------------------------------
module purge
module load cuda-12.4
module load gcc-12.4.0

echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# ---- Binary check (compiled on the login node) ----------------------------
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found or not executable. Compile it on the login node first." >&2
    exit 1
fi
echo "Using $BIN (built $(date -r "$BIN"))"

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

echo "graph,nodes,edges,run,variant,gpu_seconds,throughput_medges_s,verify" > "$CSV"

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

    mkdir -p "$REPORT_DIR/$name"

    for run in $(seq 1 "$RUNS"); do
        out="$REPORT_DIR/$name/run$run.txt"

        # Keep going if one run fails (e.g. out of GPU memory).
        if ! "$BIN" "$file" > "$out" 2>&1; then
            echo "!!! run $run FAILED on $name (see $out)"
            continue
        fi

        # e.g. "Dev+:  0.012345678 s"
        line="$(grep -m1 -E '^Dev[+-]:' "$out" || true)"
        if [[ -z "$line" ]]; then
            echo "!!! run $run: no 'Dev' time found in $out"
            continue
        fi
        if [[ "$line" == Dev+* ]]; then variant="filtered"; else variant="unfiltered"; fi
        secs="$(awk '{print $2}' <<< "$line")"
        tput="$(awk -v e="$edges" -v s="$secs" 'BEGIN { printf "%.2f", (s > 0 ? e / s / 1e6 : 0) }')"
        if grep -q "all good" "$out"; then verify="ok"; else verify="MISMATCH"; fi

        echo "$name,$nodes,$edges,$run,$variant,$secs,$tput,$verify" >> "$CSV"
        printf "run %d: %-10s GPU %s s  ->  %10s Medges/s  [%s]\n" "$run" "$variant" "$secs" "$tput" "$verify"
    done
done

echo ""
echo "All graphs done at $(date)"
echo "Results in $CSV"
echo ""
column -s, -t < "$CSV" || cat "$CSV"
