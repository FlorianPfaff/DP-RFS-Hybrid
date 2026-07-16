#!/usr/bin/env bash
set -euo pipefail

seed_count="${1:-100}"
chunk_size="${2:-10}"
seed_start_base="${3:-0}"
project_root="${PROJECT_ROOT:-$HOME/dp-rfs-benchmark}"
mtt_root="${MTT_ROOT:-$HOME/MTT-reference}"
tcl_root="${TCL_ROOT:-$HOME/TCL-reference}"
python_bin="${PYTHON_BIN:-$HOME/venvs/dp-rfs/bin/python}"
octave_image="${OCTAVE_IMAGE:-docker://gnuoctave/octave:9.2.0}"
seed_end=$((seed_start_base + seed_count - 1))
if ((seed_start_base == 0)); then
    campaign_dir="$project_root/results/pmbm_chunks"
    merged_output="$project_root/results/recurring_birth_pmbm_seed${seed_count}.csv"
else
    campaign_dir="$project_root/results/pmbm_chunks_${seed_start_base}_${seed_end}"
    merged_output="$project_root/results/recurring_birth_pmbm_seed${seed_start_base}_${seed_end}.csv"
fi

mkdir -p "$campaign_dir"
pids=()
csv_files=()
for ((seed_offset = 0; seed_offset < seed_count; seed_offset += chunk_size)); do
    seed_start=$((seed_start_base + seed_offset))
    remaining=$((seed_count - seed_offset))
    this_chunk=$chunk_size
    if ((remaining < chunk_size)); then
        this_chunk=$remaining
    fi
    dataset="$campaign_dir/dataset_${seed_start}.mat"
    output="$campaign_dir/metrics_${seed_start}.csv"
    log="$campaign_dir/run_${seed_start}.log"
    "$python_bin" "$project_root/experiments/export_recurring_birth_pmbm.py" \
        --seed-start "$seed_start" --seeds "$this_chunk" --scans 96 \
        --output "$dataset"
    apptainer exec "$octave_image" octave --quiet --eval \
        "addpath('$project_root/experiments'); official_pmbm_recurring_birth('$dataset', '$output', '$mtt_root', '$tcl_root');" \
        >"$log" 2>&1 &
    pids+=("$!")
    csv_files+=("$output")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

"$python_bin" "$project_root/experiments/merge_metric_csv.py" \
    --input "${csv_files[@]}" \
    --output "$merged_output"
