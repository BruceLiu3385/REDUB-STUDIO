#!/usr/bin/env bash
# Deliberately private: use authenticated Jupyter Server Proxy or an SSH tunnel.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${REDUB_PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd)}"
app_dir="$project_root/apps/redub_studio"
export REDUB_WEB_ROOT="${REDUB_WEB_ROOT:-/root/autodl-tmp/redub-web}"
export REDUB_WEB_ENV="${REDUB_WEB_ENV:-$REDUB_WEB_ROOT/env}"
export REDUB_TBDUB_ROOT="${REDUB_TBDUB_ROOT:-/root/autodl-tmp/tbdub}"
export REDUB_TBDUB_ENV="${REDUB_TBDUB_ENV:-$REDUB_TBDUB_ROOT/env}"
export REDUB_VOICE_ROOT="${REDUB_VOICE_ROOT:-/root/autodl-tmp/redub-voice}"
export REDUB_VOICE_ENV="${REDUB_VOICE_ENV:-$REDUB_VOICE_ROOT/env}"
export REDUB_SEPARATOR_ROOT="${REDUB_SEPARATOR_ROOT:-/root/autodl-tmp/redub-separator}"
export REDUB_SEPARATOR_ENV="${REDUB_SEPARATOR_ENV:-$REDUB_SEPARATOR_ROOT/env}"
export REDUB_DATA_DIR="${REDUB_DATA_DIR:-/root/autodl-tmp/redub-data}"
export REDUB_UPSCALE_MODEL="${REDUB_UPSCALE_MODEL:-/root/autodl-tmp/redub-upscale/RealESRGAN_x2plus.pth}"
export REDUB_PORT="${REDUB_PORT:-7860}"
export REDUB_FFMPEG="${REDUB_FFMPEG:-$REDUB_TBDUB_ENV/bin/ffmpeg}"
probe_binary="$REDUB_TBDUB_ENV/bin/ffprobe"
if [[ ! -x "$probe_binary" ]]; then probe_binary="$(command -v ffprobe || true)"; fi
export REDUB_FFPROBE="${REDUB_FFPROBE:-$probe_binary}"
export TORCH_HOME="$REDUB_SEPARATOR_ROOT/model-cache"
export PYTHONNOUSERSITE=1
export PATH="$REDUB_TBDUB_ENV/bin:$PATH"
export PYTHONPATH="$app_dir"
[[ -x "$REDUB_WEB_ENV/bin/python" ]] || { printf 'Run 00_install_web.sh first.\n' >&2; exit 2; }
[[ -f "$app_dir/redub_studio/server.py" ]] || { printf 'Missing app tree; set REDUB_PROJECT_ROOT.\n' >&2; exit 2; }
[[ -x "$REDUB_FFMPEG" && -x "$REDUB_FFPROBE" ]] || {
  printf 'Need FFmpeg and ffprobe in TBDub env/bin, or explicit REDUB_FFMPEG / REDUB_FFPROBE paths.\n' >&2
  exit 2
}
"$REDUB_WEB_ENV/bin/python" - "$REDUB_PORT" <<'PY'
import sys
port = int(sys.argv[1])
assert 1024 <= port <= 65535, 'REDUB_PORT must be 1024..65535'
PY
mkdir -p "$REDUB_DATA_DIR"
chmod 700 "$REDUB_DATA_DIR"
printf 'Listening only at http://127.0.0.1:%s; no public listener or public share URL.\n' "$REDUB_PORT"
printf 'With the authenticated Jupyter Server Proxy extension installed, open your existing Jupyter base URL + proxy/%s/.\n' "$REDUB_PORT"
printf 'Use the AutoDL SSH host/port from your console on your own computer:\n'
printf '  ssh -N -L %s:127.0.0.1:%s -p <SSH_PORT> root@<SSH_HOST>\n' "$REDUB_PORT" "$REDUB_PORT"
printf 'Then open http://127.0.0.1:%s locally. Stop with Ctrl+C.\n' "$REDUB_PORT"
printf 'This prototype has no multi-user authentication or billing; do not reverse-proxy it publicly.\n'
cd -- "$app_dir"
# server.main hardcodes host='127.0.0.1'; no --host override is accepted here.
exec "$REDUB_WEB_ENV/bin/python" -m redub_studio.server
