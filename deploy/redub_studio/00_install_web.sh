#!/usr/bin/env bash
# Lightweight API/UI only. No CUDA package or model is installed here.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${REDUB_PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd)}"
app_dir="$project_root/apps/redub_studio"
web_root="${REDUB_WEB_ROOT:-/root/autodl-tmp/redub-web}"
web_env="${REDUB_WEB_ENV:-$web_root/env}"
mkdir -p "$web_root/logs" "$web_root/tmp" "$web_root/cache"
web_root="$(realpath "$web_root")"
web_env="$(realpath -m "$web_env")"
[[ "$web_env" != / ]] || { printf 'Environment may not be the filesystem root.\n' >&2; exit 2; }
for protected_env in "${REDUB_TBDUB_ENV:-${REDUB_TBDUB_ROOT:-/root/autodl-tmp/tbdub}/env}" "${REDUB_VOICE_ENV:-${REDUB_VOICE_ROOT:-/root/autodl-tmp/redub-voice}/env}" "${REDUB_SEPARATOR_ENV:-${REDUB_SEPARATOR_ROOT:-/root/autodl-tmp/redub-separator}/env}"; do
  protected_env="$(realpath -m "$protected_env")"
  if [[ "$web_env" == "$protected_env" || "$web_env" == "$protected_env/"* || "$protected_env" == "$web_env/"* ]]; then
    printf 'Refusing to install web dependencies into another component environment.\n' >&2
    exit 2
  fi
done
[[ -f "$app_dir/requirements.txt" && -f "$app_dir/redub_studio/server.py" ]] || {
  printf 'Upload the apps/ and deploy/ trees together, or set REDUB_PROJECT_ROOT.\n' >&2
  exit 2
}
export TMPDIR="$web_root/tmp"
export UV_CACHE_DIR="$web_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$web_root/cache/python"
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH
bootstrap_python="$(command -v python3 || command -v python)"
uv_bin="$web_root/bootstrap/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
  "$bootstrap_python" -m pip install --target "$web_root/bootstrap" 'uv==0.9.4'
fi

# Starlette 1.8.0 requires Python >=3.11. Use 3.12, the locally tested runtime.
# Keep the new interpreter under web_root instead of changing TBDub's Python 3.10.
if [[ ! -x "$web_env/bin/python" ]]; then
  if [[ -n "${REDUB_WEB_PYTHON:-}" ]]; then
    "$uv_bin" venv --python "$REDUB_WEB_PYTHON" "$web_env"
  elif "$bootstrap_python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3,12))'; then
    "$uv_bin" venv --python "$bootstrap_python" "$web_env"
  else
    "$uv_bin" python install 3.12
    "$uv_bin" venv --python 3.12 "$web_env"
  fi
fi
"$web_env/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
"$uv_bin" pip install --python "$web_env/bin/python" -r "$app_dir/requirements.txt"
"$uv_bin" pip check --python "$web_env/bin/python"
PYTHONPATH="$app_dir" "$web_env/bin/python" - "$web_root" <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys
from redub_studio.server import app
assert app is not None
record = {
    'status': 'web_imports_passed', 'python': sys.version,
    'versions': {name: md.version(name) for name in (
        'fastapi', 'starlette', 'pydantic', 'uvicorn', 'python-multipart')},
    'gpu_inference_tested': False,
    'note': 'Installation/import check only. Existing local media/API tests are separate.'
}
(Path(sys.argv[1])/'logs'/'runtime-check.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
PY
"$uv_bin" pip freeze --python "$web_env/bin/python" > "$web_root/logs/pip-freeze.txt"
printf 'Web environment ready. Start: bash %q\n' "$script_dir/03_start_web.sh"
printf 'No GPU model was loaded or tested by this script.\n'
