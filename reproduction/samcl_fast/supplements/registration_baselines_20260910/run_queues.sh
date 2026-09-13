#!/usr/bin/env bash
set -euo pipefail
if [[ ${QUEUE_ALIAS_READY:-0} != 1 ]]; then
    export QUEUE_SCRIPT="$(readlink -f "$0")" QUEUE_ALIAS_READY=1
    exec -a ctl /bin/bash -c 'source "$QUEUE_SCRIPT"'
fi
root=${RUN_ROOT:?Set RUN_ROOT to the existing deployment root}
out="$root/supplements/registration_baselines_20260910"
source "$root/corrections/nlst_native_v4_20260831/env.sh"
export OPENBLAS_NUM_THREADS=4
mkdir -p "$out/logs"
exec 9>"$out/pipeline.lock"
flock -n 9 || { echo 'Existing baseline controller'; exit 1; }
[[ ! -f "$out/jobs.tsv" ]] || { echo 'Job ledger exists; inspect before restart'; exit 1; }
python_bin=$(command -v python)
printf 'method\tgpu\tpid\n' > "$out/jobs.tsv"
printf 'running\n' > "$out/pipeline.status"
queue() {
    local gpu="$1"; shift
    local queue_failed=0
    for method in "$@"; do
        local required=22000
        [[ "$method" != gpm ]] || required=28000
        local available
        available=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        if (( available < required )); then
            printf 'blocked: %s requires %s MiB, available %s\n' "$method" "$required" "$available" > "$out/queue_gpu${gpu}.status"
            return 1
        fi
        printf 'running: %s\n' "$method" > "$out/queue_gpu${gpu}.status"
        local resume=()
        [[ ! -f "$out/runs/$method/checkpoints/latest.pt" ]] || resume=(--resume)
        (
            export CUDA_VISIBLE_DEVICES="$gpu" WORKER_ALIAS="w0$gpu" RUN_SCRIPT="$out/run_baseline.py"
            export RUN_ARGUMENTS
            printf -v RUN_ARGUMENTS '%s\n' --method "$method" --data-root "$LEARN2REG_ROOT" \
                --run-dir "$out/runs/$method" \
                --reference-root "$root/supplements/registration_metrics_20260909/references" "${resume[@]}"
            exec -a "$WORKER_ALIAS" "$python_bin" -c 'import os,sys,runpy,ctypes; ctypes.CDLL(None).prctl(15,os.environ["WORKER_ALIAS"].encode(),0,0,0); p=os.environ["RUN_SCRIPT"]; sys.path.insert(0,os.path.dirname(p)); sys.argv=[p]+os.environ["RUN_ARGUMENTS"].splitlines(); runpy.run_path(p,run_name="__main__")'
        ) > "$out/logs/$method.stdout.log" 2> "$out/logs/$method.stderr.log" &
        local pid=$!
        printf '%s\t%s\t%s\n' "$method" "$gpu" "$pid" >> "$out/jobs.tsv"
        local code=0
        wait "$pid" || code=$?
        printf '%s\n' "$code" > "$out/logs/$method.exitcode"
        if (( code )); then
            printf 'failed: %s\n' "$method" > "$out/queue_gpu${gpu}.status"
            queue_failed=1
        fi
    done
    if (( queue_failed )); then
        printf 'finished_with_failures\n' > "$out/queue_gpu${gpu}.status"
        return 1
    fi
    printf 'complete\n' > "$out/queue_gpu${gpu}.status"
}
queue 2 sequential si & q2=$!
queue 3 ewc gpm & q3=$!
failed=0
wait "$q2" || failed=1
wait "$q3" || failed=1
if (( failed )); then printf 'failed or blocked; inspect queue status and logs\n' > "$out/pipeline.status"; exit 1; fi
printf 'complete\n' > "$out/pipeline.status"
