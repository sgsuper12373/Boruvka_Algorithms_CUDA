#!/bin/bash
# ---------------------------------------------------------------------------
# Runs the CUDA Boruvka MST (v1) on every .egr graph in Graphs/egr_graphs and
# writes an Nsight Systems (nsys) report per graph.
#
# For each graph:
#   1. plain run      -> correctness check + chrono timings (no profiler overhead)
#   2. nsys run       -> <graph>.nsys-rep (open in the Nsight Systems GUI)
#   3. nsys stats     -> per-kernel / memcpy / memset / API times as CSV
#
# Output: reports/<job_id>/<graph>/
#   <graph>.txt                          program output of the plain run
#   <graph>.nsys-rep                     full timeline
#   <graph>_cuda_gpu_kern_sum.csv        time per kernel (summed over all rounds)
#   <graph>_cuda_gpu_mem_time_sum.csv    memcpy H2D/D2H + memset times
#   <graph>_cuda_api_sum.csv             host-side CUDA API times
#
# The binary is compiled on the LOGIN node (compute-node compilation is broken
# for now). Compile WITHOUT -DDEBUG for timing runs (DEBUG syncs after every
# kernel), e.g.:
#   nvcc -O3 -arch=sm_XX -Isrc src/Boruvka_intermediate_cuda.cu -o bin/mst
#
# Submit from the project root (so logs/ resolves correctly):
#   cd ~/BTP_2026/Boruvka_Algorithms_CUDA
#   mkdir -p logs
#   sbatch scripts/run_gpu_mst.sbatch
# ---------------------------------------------------------------------------

#SBATCH --job-name=boruvka_cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                  # host RAM: big graphs + random-weight generation
#SBATCH --time=03:00:00            # all graphs, each run twice (plain + nsys)
#SBATCH --output=logs/%x_%j.out    # logs/ must exist BEFORE sbatch (SLURM won't create it)
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
BTP_DIR="$HOME/BTP_2026_CUDA"
PROJECT_DIR="$BTP_DIR/Boruvka_Algorithms_CUDA"
GRAPH_DIR="$BTP_DIR/Graphs/egr_graphs"

BIN="$PROJECT_DIR/bin/mst"
REPORT_DIR="$PROJECT_DIR/reports/${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"

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
# All graphs:
shopt -s nullglob
graphs=("$GRAPH_DIR"/*.egr)
shopt -u nullglob

# Selected graphs (bare names, resolved against GRAPH_DIR below):
# graphs=( "amazon0601.egr" "in-2004.egr" "rmat16.sym.egr" "USA-road-d.NY.egr" "internet.egr" )

if [[ ${#graphs[@]} -eq 0 ]]; then
    echo "ERROR: no .egr graphs found in $GRAPH_DIR" >&2
    exit 1
fi
echo "Found ${#graphs[@]} graphs"

# ---- Run ------------------------------------------------------------------
for g in "${graphs[@]}"; do
    name="$(basename "$g" .egr)"
    # Works for both bare names (selected list) and full paths (glob).
    if [[ "$g" == /* ]]; then file="$g"; else file="$GRAPH_DIR/$g"; fi

    echo ""
    echo "=================================================="
    echo ">>> $name"
    echo "=================================================="

    if [[ ! -f "$file" ]]; then
        echo "!!! File $file not found, skipping"
        continue
    fi

    out="$REPORT_DIR/$name"
    mkdir -p "$out"

    # 1. Plain run: correctness + chrono timings without profiler overhead.
    #    Keep going if one graph fails (e.g. out of GPU memory).
    if ! "$BIN" "$file" 2>&1 | tee "$out/$name.txt"; then
        echo "!!! FAILED on $name, skipping nsys"
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
