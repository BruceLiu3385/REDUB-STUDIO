#!/usr/bin/env bash
# Usage: bash 04_run_sample.sh /absolute/source.mp4 /absolute/driving.wav [seconds<=15] [label]
# TBDUB_PREPROCESS_ONLY=1 saves the aligned face preview without loading the generator.
# TBDUB_CPU_OFFLOAD=0 keeps models on a GPU with enough VRAM (for example 84GB).
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
use_tbdub_env
[[ $# -ge 2 && $# -le 4 ]] || { printf 'Usage: %s SOURCE_VIDEO DRIVING_AUDIO [SECONDS<=15] [LABEL]\n' "$0" >&2; exit 2; }
source_video="$(realpath -- "$1")"
driving_audio="$(realpath -- "$2")"
sample_seconds="${3:-15}"
sample_label="${4:-sample}"
[[ -f "$source_video" && -f "$driving_audio" ]]
[[ "$sample_label" =~ ^[a-zA-Z0-9_-]+$ ]] || { printf 'LABEL: use letters, digits, _ or -.\n' >&2; exit 2; }
python - "$sample_seconds" <<'PY'
import math, sys
x = float(sys.argv[1])
assert math.isfinite(x) and 0 < x <= 15, 'First-round samples must be 0 < seconds <= 15'
PY
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$TBDUB_ROOT/results/${sample_label}_${run_stamp}"
mkdir "$run_dir"
ffmpeg -nostdin -hide_banner -loglevel warning -y -i "$source_video" -t "$sample_seconds" \
  -map 0:v:0 -an -vf fps=25 -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p "$run_dir/source.mp4"
ffmpeg -nostdin -hide_banner -loglevel warning -y -i "$driving_audio" -t "$sample_seconds" \
  -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le "$run_dir/driving.wav"
python - "$run_dir/driving.wav" <<'PY'
import sys, wave
with wave.open(sys.argv[1]) as f:
    duration = f.getnframes()/f.getframerate()
assert duration > 0.25, 'Driving audio is empty or too short'
print(f'Actual driving audio: {duration:.3f}s')
PY
cd "$TBDUB_REPO"
args=(inference.py --video "$run_dir/source.mp4" --audio "$run_dir/driving.wav"
  --checkpoint-dir "$TBDUB_CHECKPOINTS" --inference-mode student --num-student-steps 2
  --sigma-shift 1.0 --motion-from-latents --seed 42
  --sample-name "$sample_label" --output-dir "$run_dir"
  --preprocess-report "$run_dir/face_detection.json" --save-comparison)
if [[ "${TBDUB_CPU_OFFLOAD:-1}" == 1 ]]; then args+=(--cpu-offload); fi
if [[ "${TBDUB_PREPROCESS_ONLY:-0}" == 1 ]]; then args+=(--preprocess-only); fi
printf '%q ' python "${args[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
# Measure sampled GPU memory (whole visible GPU), including process startup/loading.
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 1 > "$run_dir/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
start_epoch="$(date +%s)"
set +e
python "${args[@]}" 2>&1 | tee "$run_dir/inference.log"
status="${PIPESTATUS[0]}"
set -e
end_epoch="$(date +%s)"
printf 'exit_code=%s\nwall_seconds=%s\n' "$status" "$((end_epoch-start_epoch))" > "$run_dir/timing.txt"
printf 'Results: %s\n' "$run_dir"
exit "$status"
