#!/bin/bash
# ---------------------------------------------------------------------------
# Runs the CUDA Boruvka MST (v1) on selected .egr graphs in Graphs/egr_graphs.
#
# The binary is compiled on the LOGIN node (compute-node compilation is broken
# for now). Compile first, e.g.:
#   nvcc -O3 -arch=sm_XX -DDEBUG -Isrc src/Boruvka_intermediate_cuda.cu -o bin/mst_debug
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
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out    # logs/ must exist BEFORE sbatch (SLURM won't create it)
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
BTP_DIR="$HOME/BTP_2026"
PROJECT_DIR="$BTP_DIR/Boruvka_Algorithms_CUDA"
GRAPH_DIR="$BTP_DIR/Graphs/egr_graphs"

BIN="$PROJECT_DIR/bin/mst_debug"

# ---- Environment ----------------------------------------------------------
module purge
module load cuda-12.4
module load gcc-12.4.0

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# ---- Binary check (compiled on the login node) ----------------------------
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found or not executable. Compile it on the login node first." >&2
    exit 1
fi
echo "Using $BIN (built $(date -r "$BIN"))"

# ---- Graph selection ------------------------------------------------------
# All graphs:
# shopt -s nullglob
# graphs=("$GRAPH_DIR"/*.egr)
# shopt -u nullglob

# Selected graphs (bare names, resolved against GRAPH_DIR below):
graphs=( "amazon0601.egr" "in-2004.egr" "rmat16.sym.egr" "USA-road-d.NY.egr" "internet.egr" )

if [[ ${#graphs[@]} -eq 0 ]]; then
    echo "ERROR: graph list is empty" >&2
    exit 1
fi

# ---- Run ------------------------------------------------------------------
for g in "${graphs[@]}"; do
    name="$(basename "$g")"
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

    # Keep going if one graph fails (e.g. out of GPU memory).
    "$BIN" "$file" || echo "!!! FAILED on $name (exit $?)"
done

echo ""
echo "All graphs done at $(date)"
