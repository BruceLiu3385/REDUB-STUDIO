#!/usr/bin/env bash
# Default: read-only checks and plan. --execute clones offline unless network is
# explicitly allowed with the additional --allow-conda-network flag.
# Never installs into or moves the source/target environments.
set -Eeuo pipefail
execute=0
allow_conda_network=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) execute=1 ;;
    --allow-conda-network) allow_conda_network=1 ;;
    --help|-h)
      printf 'Usage: bash 00b_clone_voice_env.sh [--execute [--allow-conda-network]]\n'
      printf 'Default: check-only. --execute: offline clone.\n'
      printf '%s\n' '--allow-conda-network requires --execute; permits conda to fetch missing cached packages.'
      printf 'REDUB_TBDUB_ENV: source conda env; REDUB_VOICE_ROOT/ENV: clone destination.\n'
      printf 'Nonempty target directories are always refused; move any old env aside yourself.\n'
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
if [[ "$allow_conda_network" -eq 1 && "$execute" -ne 1 ]]; then
  printf '%s\n' '--allow-conda-network requires --execute; no operation performed.' >&2
  exit 2
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ "$allow_conda_network" -eq 1 ]]; then
  export CONDA_OFFLINE=false
  clone_mode='explicitly allowed conda network'
else
  export CONDA_OFFLINE=true
  clone_mode='offline'
fi
unset PYTHONPATH
source_env="$(realpath -e "${REDUB_TBDUB_ENV:-${REDUB_TBDUB_ROOT:-/root/autodl-tmp/tbdub}/env}")"
voice_root="$(realpath -m "${REDUB_VOICE_ROOT:-/root/autodl-tmp/redub-voice}")"
target_env="$(realpath -m "${REDUB_VOICE_ENV:-$voice_root/env}")"
[[ -x "$source_env/bin/python" && -d "$source_env/conda-meta" ]] || {
  printf 'Source must be an existing conda environment containing Python.\n' >&2; exit 2;
}
if [[ "$target_env" == / || "$target_env" == "$source_env" || "$target_env" == "$source_env/"* || "$source_env" == "$target_env/"* ]]; then
  printf 'Source and target must be distinct, non-overlapping environment directories.\n' >&2
  exit 2
fi
[[ ! -L "${REDUB_VOICE_ENV:-$voice_root/env}" ]] || {
  printf 'Target may not be a symlink.\n' >&2; exit 2;
}
conda_bin="${REDUB_CONDA_BIN:-}"
if [[ -z "$conda_bin" ]]; then conda_bin="$(type -P conda || true)"; fi
for candidate in /root/miniconda3/bin/conda /opt/conda/bin/conda; do
  if [[ -z "$conda_bin" && -x "$candidate" ]]; then conda_bin="$candidate"; fi
done
[[ -n "$conda_bin" && -x "$conda_bin" ]] || {
  printf 'Missing conda executable; set REDUB_CONDA_BIN.\n' >&2; exit 2;
}
source_bytes="$(du -sb -- "$source_env" | cut -f1)"
# Check without creating directories or importing models in the default mode.
"$source_env/bin/python" - "$source_env" "$target_env" "$voice_root" "$source_bytes" <<'PY'
import json, os
from pathlib import Path
import shutil
import sys
import torch
source, target, root = map(Path, sys.argv[1:4])
source_bytes = int(sys.argv[4])
assert sys.version_info[:2] == (3,10), sys.version
assert torch.__version__ == '2.9.0+cu128', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert Path(torch.__file__).resolve().is_relative_to(source), 'Torch is not contained in source env'
if target.exists():
    if not target.is_dir() or next(target.iterdir(), None) is not None:
        raise SystemExit('Target exists and is nonempty: move it to a recoverable backup before retrying; nothing was changed.')
ancestor = target.parent
while not ancestor.exists():
    ancestor = ancestor.parent
free = shutil.disk_usage(ancestor).free
required = source_bytes + 5*2**30
if free < required:
    raise SystemExit(f'Insufficient destination space: need source size + 5GiB ({required/2**30:.2f}GiB), available {free/2**30:.2f}GiB')
print(json.dumps({'status':'preflight_passed','source':str(source),'target':str(target),
    'source_bytes':source_bytes,'free_bytes':free,'required_free_bytes':required,
    'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda}, indent=2))
PY
clone_command=("$conda_bin" create --yes --copy --prefix "$target_env" --clone "$source_env")
if [[ "$allow_conda_network" -ne 1 ]]; then clone_command+=(--offline); fi
printf 'Planned copy clone, mode: %s (source remains unchanged):\n' "$clone_mode"
printf '  '
printf '%q ' "${clone_command[@]}"
printf '\n'
if [[ "$execute" -ne 1 ]]; then
  printf 'Check-only mode. Re-run with --execute to perform this clone. No files were created.\n'
  exit 0
fi
mkdir -p "$voice_root/logs" "$voice_root/tmp"
export TMPDIR="$voice_root/tmp"
# Reuse existing caches. Offline mode fails if required conda packages are
# absent; explicitly allowed network mode may retrieve them. Neither mode
# silently falls back to the other. Cache writes stay outside the source env.
clone_cache="$voice_root/cache/clone-conda-pkgs"
tbdub_cache="${REDUB_TBDUB_ROOT:-/root/autodl-tmp/tbdub}/cache/conda-pkgs"
conda_base_cache="$(dirname -- "$(dirname -- "$conda_bin")")/pkgs"
export CONDA_PKGS_DIRS="$clone_cache:${REDUB_CLONE_PACKAGE_CACHES:-$tbdub_cache:$conda_base_cache}"
mkdir -p "$clone_cache"
receipt="$voice_root/logs/cloned-source.json"
source_history_before="$(sha256sum "$source_env/conda-meta/history" | cut -d' ' -f1)"
"$source_env/bin/python" - "$receipt" "$source_env" "$target_env" "$source_bytes" "$source_history_before" "$allow_conda_network" <<'PY'
import datetime,json
from pathlib import Path
import sys
record={'status':'clone_started','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source':sys.argv[2],'target':sys.argv[3],'source_bytes':int(sys.argv[4]),
        'source_history_sha256_before':sys.argv[5],'copy':True,'offline':sys.argv[6]!='1',
        'conda_network_explicitly_allowed':sys.argv[6]=='1',
        'source_modified_by_script':False,'voice_dependencies_installed':False}
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
PY
if ! "${clone_command[@]}" \
    2>&1 | tee "$voice_root/logs/clone-conda.log"; then
  "$source_env/bin/python" - "$receipt" <<'PY'
import json
from pathlib import Path
import sys
path=Path(sys.argv[1]); record=json.loads(path.read_text())
record.update(status='clone_failed', note='Partial target retained for inspection; no environment was deleted or moved.')
path.write_text(json.dumps(record,indent=2)+'\n')
PY
  printf 'Clone failed (%s). Partial target and log retained; no automatic cleanup or network fallback.\n' "$clone_mode" >&2
  exit 1
fi
source_history_after="$(sha256sum "$source_env/conda-meta/history" | cut -d' ' -f1)"
[[ "$source_history_before" == "$source_history_after" ]] || {
  printf 'Source conda history changed during cloning; stop and investigate concurrent operations.\n' >&2
  exit 1
}
"$target_env/bin/python" - "$receipt" "$target_env" "$source_history_after" <<'PY'
import json
from pathlib import Path
import sys
import torch
target=Path(sys.argv[2])
assert sys.version_info[:2] == (3,10), sys.version
assert Path(sys.prefix).resolve() == target.resolve(), (sys.prefix,target)
assert torch.__version__ == '2.9.0+cu128' and torch.version.cuda == '12.8'
assert Path(torch.__file__).resolve().is_relative_to(target), torch.__file__
assert torch.cuda.is_available(), 'CUDA unavailable in cloned env'
value=torch.randn((64,64),device='cuda')
assert torch.isfinite(value@value).all().item()
torch.cuda.synchronize()
path=Path(sys.argv[1]); record=json.loads(path.read_text())
record.update(status='cloned_source_validated', source_history_sha256_after=sys.argv[3],
              python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0),torch_file=torch.__file__,
              voice_dependencies_installed=False,voice_inference_tested=False)
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
printf 'Clone ready at %q. Run 01_install_voice.sh to install the matching torchaudio and voice dependencies.\n' "$target_env"
printf 'Keep torch==2.9.0+cu128 and torchaudio==2.9.0+cu128 constrained for subsequent installs.\n'
