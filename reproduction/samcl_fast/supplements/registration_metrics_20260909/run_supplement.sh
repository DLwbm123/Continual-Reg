#!/usr/bin/env bash
set -euo pipefail
root="${SAMCL_EXPERIMENT_ROOT:?Set SAMCL_EXPERIMENT_ROOT to the existing experiment root}"
out="$root/supplements/registration_metrics_20260909"
source "$root/corrections/nlst_native_v4_20260831/env.sh"
mkdir -p "$out/logs"
exec 9>"$out/pipeline.lock"
flock -n 9 || { echo 'Supplement pipeline already running' >&2; exit 1; }
[[ ! -f "$out/jobs.tsv" ]] || { echo 'Existing job ledger; inspect before restarting' >&2; exit 1; }
for gpu in 2 3; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    required=20000; [[ "$gpu" != 2 ]] || required=33000
    (( free >= required )) || { echo "GPU $gpu insufficient free memory: $free MiB" >&2; exit 1; }
done
printf 'task\tgpu\tpid\n' > "$out/jobs.tsv"
printf 'running\n' > "$out/pipeline.status"
python_bin=$(command -v python)

reference() {
    local task="$1" gpu="$2" name="$3"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c 'exec -a "$1" "$2" "$3" --task "$4" --data-root "$5" --run-dir "$6"' \
        _ "$name" "$python_bin" "$out/run_reference.py" "$task" "$LEARN2REG_ROOT" "$out/references/$task" \
        > "$out/logs/reference_$task.stdout.log" 2> "$out/logs/reference_$task.stderr.log" &
    last_pid=$!
    printf '%s\t%s\t%s\n' "$task" "$gpu" "$last_pid" >> "$out/jobs.tsv"
}

# CTCT smoke measured 10.5 GiB. Reserve 33 GiB for two plain-update workers.
reference ctct 2 ref2c; ctct_pid=$last_pid
reference mrct 2 ref2m; mrct_pid=$last_pid
failed=0
CUDA_VISIBLE_DEVICES=3 bash -c 'exec -a prf3 "$1" "$2" --experiment-root "$3" --data-root "$4" --output "$5"' \
    _ "$python_bin" "$out/profile_resources.py" "$root" "$LEARN2REG_ROOT" "$out/results/resource_profile.json" \
    > "$out/logs/profile.stdout.log" 2> "$out/logs/profile.stderr.log" &
profile_pid=$!
printf 'resource_profile\t3\t%s\n' "$profile_pid" >> "$out/jobs.tsv"
wait "$profile_pid" || failed=1
# Keep GPU3 uncontended during the timing comparison; then fill its next slot.
reference nlst 3 ref3n; nlst_pid=$last_pid
wait "$ctct_pid" || failed=1
wait "$mrct_pid" || failed=1
wait "$nlst_pid" || failed=1
"$python_bin" "$out/collect_metrics.py" --root "$out" > "$out/logs/collect.stdout.log" 2> "$out/logs/collect.stderr.log" || failed=1
if (( failed )); then
    printf 'failed; inspect logs\n' > "$out/pipeline.status"
    exit 1
fi
printf 'complete\n' > "$out/pipeline.status"
