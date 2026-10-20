#!/usr/bin/env bash
# Optional BGM separation, isolated from both TBDub and voice dependencies.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
separator_root="${REDUB_SEPARATOR_ROOT:-/root/autodl-tmp/redub-separator}"
separator_env="${REDUB_SEPARATOR_ENV:-$separator_root/env}"
voice_root="${REDUB_VOICE_ROOT:-/root/autodl-tmp/redub-voice}"
tbdub_env="${REDUB_TBDUB_ENV:-${REDUB_TBDUB_ROOT:-/root/autodl-tmp/tbdub}/env}"
mkdir -p "$separator_root/logs" "$separator_root/tmp" "$separator_root/cache" "$separator_root/model-cache"
separator_root="$(realpath "$separator_root")"
separator_env="$(realpath -m "$separator_env")"
[[ "$separator_env" != / ]] || { printf 'Environment may not be the filesystem root.\n' >&2; exit 2; }
for protected_env in "$tbdub_env" "${REDUB_VOICE_ENV:-$voice_root/env}" "${REDUB_WEB_ENV:-${REDUB_WEB_ROOT:-/root/autodl-tmp/redub-web}/env}"; do
  protected_env="$(realpath -m "$protected_env")"
  if [[ "$separator_env" == "$protected_env" || "$separator_env" == "$protected_env/"* || "$protected_env" == "$separator_env/"* ]]; then
    printf 'Refusing to change another component environment.\n' >&2
    exit 2
  fi
done
# Same disk + same CUDA wheels as voice: uv can reuse cached artifacts/hardlinks.
export UV_CACHE_DIR="$voice_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$separator_root/cache/python"
export TMPDIR="$separator_root/tmp" TORCH_HOME="$separator_root/model-cache"
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset PYTHONPATH
mkdir -p "$UV_CACHE_DIR"
df -h "$separator_root" "$UV_CACHE_DIR"
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bootstrap_python="$(command -v python3 || command -v python)"
uv_bin="$voice_root/bootstrap/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
  uv_bin="$separator_root/bootstrap/bin/uv"
  if [[ ! -x "$uv_bin" ]]; then
    "$bootstrap_python" -m pip install --target "$separator_root/bootstrap" 'uv==0.9.4'
  fi
fi
if [[ ! -x "$separator_env/bin/python" ]]; then
  python_seed="${REDUB_SEPARATOR_PYTHON:-}"
  if [[ -z "$python_seed" && -x "$tbdub_env/bin/python" ]]; then
    python_seed="$tbdub_env/bin/python"
  fi
  if [[ -n "$python_seed" ]]; then
    "$uv_bin" venv --python "$python_seed" "$separator_env"
  else
    "$uv_bin" python install 3.10
    "$uv_bin" venv --python 3.10 "$separator_env"
  fi
fi
"$separator_env/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
if ! "$separator_env/bin/python" -c 'import torch,torchaudio; assert torch.__version__=="2.9.0+cu128" and torchaudio.__version__=="2.9.0+cu128"' 2>/dev/null; then
  "$uv_bin" pip install --python "$separator_env/bin/python" \
    'torch==2.9.0+cu128' 'torchaudio==2.9.0+cu128' \
    --default-index https://download.pytorch.org/whl/cu128 \
    --index "${REDUB_PYPI_INDEX:-https://pypi.org/simple}" --index-strategy unsafe-best-match
fi
cat > "$separator_root/runtime-requirements.txt" <<'REQ'
demucs==4.0.1
numpy==1.26.4
scipy==1.15.3
soundfile==0.13.1
hydra-core==1.3.2
omegaconf==2.3.0
einops==0.8.1
julius==0.2.7
lameenc==1.8.1
openunmix==1.3.0
dora-search==0.1.12
REQ
printf '%s\n' 'torch==2.9.0+cu128' 'torchaudio==2.9.0+cu128' > "$separator_root/cuda-constraints.txt"
"$uv_bin" pip install --python "$separator_env/bin/python" -c "$separator_root/cuda-constraints.txt" -r "$separator_root/runtime-requirements.txt"
"$uv_bin" pip check --python "$separator_env/bin/python"

# Explicit installation-time prefetch. Requests must not fetch weights later.
# Official files.txt supplies a hash prefix; the full deployed hash is locked here.
"$separator_env/bin/python" - "$separator_root" <<'PY'
import hashlib, importlib.metadata as md, json, os
from pathlib import Path
import sys
import torch, torchaudio, soundfile
from demucs.pretrained import get_model
root = Path(sys.argv[1])
assert torch.__version__.split('+')[0] == '2.9.0'
assert torchaudio.__version__.split('+')[0] == '2.9.0'
assert torch.version.cuda == '12.8' and torch.cuda.is_available()
value = torch.randn((64,64), device='cuda')
assert torch.isfinite(value @ value).all().item()
torch.cuda.synchronize()
url = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th'
path = Path(torch.hub.get_dir())/'checkpoints'/'955717e8-8726e21a.th'
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    partial = path.with_suffix('.th.part')
    torch.hub.download_url_to_file(url, str(partial), hash_prefix='8726e21a', progress=True)
    partial.replace(path)
digest = hashlib.sha256()
with path.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        digest.update(block)
actual = digest.hexdigest()
assert actual == '8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4', 'Locked htdemucs SHA-256 mismatch'
model = get_model('htdemucs')
assert model.samplerate == 44100 and model.audio_channels == 2
assert 'vocals' in model.sources
record = {
    'status': 'prefetch_cpu_model_load_and_cuda_math_passed',
    'prefetch_completed': True, 'separation_inference_tested': False,
    'model': 'htdemucs', 'url': url, 'cache_file': str(path),
    'size': path.stat().st_size, 'sha256': actual,
    'upstream_sha256_prefix': '8726e21a', 'demucs': md.version('demucs'),
    'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
    'gpu': torch.cuda.get_device_name(0),
    'audio_io': 'project worker uses FFmpeg/SoundFile; no torchaudio.load/save or TorchCodec'
}
(root/'logs'/'runtime-check.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
PY
"$uv_bin" pip freeze --python "$separator_env/bin/python" > "$separator_root/logs/pip-freeze.txt"
if [[ -f "$script_dir/model_sources.json" ]]; then
  cp "$script_dir/model_sources.json" "$separator_root/model_sources.json"
fi
printf 'Separator prefetch finished; real separation quality/performance still requires a sample.\n'
printf 'Configure REDUB_SEPARATOR_ROOT=%q REDUB_SEPARATOR_ENV=%q\n' "$separator_root" "$separator_env"
