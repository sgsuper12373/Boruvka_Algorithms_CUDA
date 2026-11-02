#!/bin/bash
# ---------------------------------------------------------------------------
# Submits all benchmark jobs ONE AFTER ANOTHER: every job waits until the
# previous one has ended (Slurm --dependency=afterany), so our jobs never
# share a GPU node.
#
# Why: when jobs 92933 (atomic guard) and 92934 (ECL) ran together on gpu01,
# the ECL job finishing also killed the atomic-guard job
# (sacct: both ended at 05:18:34, the guard's batch step was CANCELLED).
#
# afterany = start after the previous job ends for ANY reason, so one failed
# job doesn't stop the rest of the chain.
#
# Usage (on the login node, from the project root, after `make all`):
#   cd ~/BTP_2026_CUDA/Boruvka_Algorithms_CUDA
#   bash scripts/submit_all.sh
# Waiting jobs show as "PD (Dependency)" in `squeue -u $USER`, that is normal.
# ---------------------------------------------------------------------------

set -euo pipefail

# Jobs in the order they run. Add the script of a new version here.
jobs=(
    "scripts/run_gpu_mst.sbatch"         # base          -> reports/<id>/
    "scripts/run_atomic_guard.sbatch"    # atomic guard  -> reports/<id>/
    "scripts/run_warp_mst.sbatch"        # warp          -> reports/<id>/
    "scripts/run_ecl_mst.sbatch"         # ECL-MST       -> reports/ecl_<id>/
)

# sbatch scripts write to logs/, which must exist before submitting
if [[ ! -d scripts ]]; then
    echo "ERROR: run this from Boruvka_Algorithms_CUDA/ (scripts/ not found here)" >&2
    exit 1
fi
mkdir -p logs

prev=""     # job id of the previous job in the chain
for script in "${jobs[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "ERROR: $script not found" >&2
        exit 1
    fi

    if [[ -z "$prev" ]]; then
        id="$(sbatch --parsable "$script")"                                 # first job starts right away
    else
        id="$(sbatch --parsable --dependency=afterany:"$prev" "$script")"   # waits for the previous job
    fi
    id="${id%%;*}"      # --parsable may print "<id>;<cluster>", keep only the id

    if [[ -z "$prev" ]]; then
        printf "%-36s job %s\n" "$script" "$id"
    else
        printf "%-36s job %s  (waits for %s)\n" "$script" "$id" "$prev"
    fi
    prev="$id"
done

echo ""
echo "Submitted ${#jobs[@]} jobs. Note the job ids above: they tell which reports/ folder is which version."
echo "Check progress with: squeue -u \$USER"
