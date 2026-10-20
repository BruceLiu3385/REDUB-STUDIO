#!/usr/bin/env bash
# Run on the rented Linux CUDA instance. This never installs into TBDub's env.
# REDUB_VOICE_ROOT defaults to /root/autodl-tmp/redub-voice; all caches stay there.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
voice_root="${REDUB_VOICE_ROOT:-/root/autodl-tmp/redub-voice}"
voice_env="${REDUB_VOICE_ENV:-$voice_root/env}"
code_ref='5de7a54aa4e5e2baadb0182dde554908b48b85c2'
model_ref='5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'
code_root="$voice_root/src/chatterbox"
mkdir -p "$voice_root/logs" "$voice_root/checkpoints" "$voice_root/tmp" "$voice_root/cache"
voice_root="$(realpath "$voice_root")"
voice_env="$(realpath -m "$voice_env")"
tbdub_env="$(realpath -m "${REDUB_TBDUB_ENV:-${REDUB_TBDUB_ROOT:-/root/autodl-tmp/tbdub}/env}")"
[[ "$voice_env" != / ]] || { printf 'Environment may not be the filesystem root.\n' >&2; exit 2; }
for protected_env in "$tbdub_env" "${REDUB_WEB_ENV:-${REDUB_WEB_ROOT:-/root/autodl-tmp/redub-web}/env}" "${REDUB_SEPARATOR_ENV:-${REDUB_SEPARATOR_ROOT:-/root/autodl-tmp/redub-separator}/env}"; do
  protected_env="$(realpath -m "$protected_env")"
  if [[ "$voice_env" == "$protected_env" || "$voice_env" == "$protected_env/"* || "$protected_env" == "$voice_env/"* ]]; then
    printf 'Refusing to install voice dependencies into an overlapping component environment.\n' >&2
    exit 2
  fi
done
export TMPDIR="$voice_root/tmp"
export UV_CACHE_DIR="$voice_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$voice_root/cache/python"
export HF_HOME="$voice_root/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset PYTHONPATH
command -v nvidia-smi >/dev/null
command -v git >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$voice_root"
bootstrap_python="$(command -v python3 || command -v python)"
"$bootstrap_python" - "$voice_root" "$voice_env" <<'PY'
import pathlib, shutil, sys
root, env = map(pathlib.Path, sys.argv[1:])
free = shutil.disk_usage(root).free
if not env.exists() and free < 18*2**30:
    raise SystemExit('Need >=18GiB free for an isolated CUDA environment, cache and VC checkpoint')
PY

# Pinned uv bootstrap is scoped to voice_root and does not change the base env.
uv_bin="$voice_root/bootstrap/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
  "$bootstrap_python" -m pip install --target "$voice_root/bootstrap" 'uv==0.9.4'
fi
if [[ ! -x "$voice_env/bin/python" ]]; then
  python_seed="${REDUB_VOICE_PYTHON:-}"
  if [[ -z "$python_seed" && -x "$tbdub_env/bin/python" ]]; then
    python_seed="$tbdub_env/bin/python"
  fi
  if [[ -n "$python_seed" ]]; then
    "$uv_bin" venv --python "$python_seed" "$voice_env"
  else
    "$uv_bin" python install 3.10
    "$uv_bin" venv --python 3.10 "$voice_env"
  fi
fi
"$voice_env/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
if [[ ! -d "$code_root/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/resemble-ai/chatterbox.git "$code_root"
fi
[[ "$(git -C "$code_root" remote get-url origin)" == 'https://github.com/resemble-ai/chatterbox.git' ]]
if [[ -n "$(git -C "$code_root" status --porcelain 2>/dev/null)" ]] && git -C "$code_root" rev-parse --verify HEAD >/dev/null 2>&1; then
  # A brand-new --no-checkout clone lists tracked deletions; allow that state once.
  if [[ -f "$code_root/pyproject.toml" ]]; then
    printf 'Voice source contains local changes; review them before installing.\n' >&2
    exit 1
  fi
fi
git -C "$code_root" fetch origin "$code_ref"
git -C "$code_root" checkout --detach "$code_ref"
git -C "$code_root" rev-parse HEAD > "$voice_root/logs/code_revision.txt"

if ! "$voice_env/bin/python" -c 'import torch,torchaudio; assert torch.__version__=="2.9.0+cu128" and torchaudio.__version__=="2.9.0+cu128"' 2>/dev/null; then
  # Generic dependencies can use a regional PyPI mirror; the CUDA build remains exact.
  "$uv_bin" pip install --python "$voice_env/bin/python" \
    'torch==2.9.0+cu128' 'torchaudio==2.9.0+cu128' \
    --default-index https://download.pytorch.org/whl/cu128 \
    --index "${REDUB_PYPI_INDEX:-https://pypi.org/simple}" --index-strategy unsafe-best-match
fi

# Keep CUDA 12.8 instead of upstream's older torch==2.6 metadata pin.
# Diffusers 0.36 / safetensors 0.7 overrides are deliberate for the current Hub API.
# These are an integration configuration; actual VC quality still needs a sample run.
cat > "$voice_root/runtime-requirements.txt" <<'REQ'
# PerTH 1.0.1 imports pkg_resources, removed in setuptools 82.
setuptools==80.9.0
numpy==1.26.4
scipy==1.15.3
librosa==0.11.0
soundfile==0.13.1
s3tokenizer==0.3.0
transformers==5.2.0
diffusers==0.36.0
safetensors==0.7.0
huggingface-hub==1.4.1
conformer==0.3.2
omegaconf==2.3.0
resemble-perth==1.0.1
spacy-pkuseg==1.0.1
pykakasi==2.3.0
gradio==6.8.0
pyloudnorm==0.1.1
imageio-ffmpeg==0.6.0
REQ
printf '%s\n' 'torch==2.9.0+cu128' 'torchaudio==2.9.0+cu128' > "$voice_root/cuda-constraints.txt"
"$uv_bin" pip install --python "$voice_env/bin/python" -c "$voice_root/cuda-constraints.txt" -r "$voice_root/runtime-requirements.txt"
"$uv_bin" pip install --python "$voice_env/bin/python" --no-deps --editable "$code_root"

"$voice_env/bin/python" - "$voice_root" "$model_ref" <<'PY'
import hashlib, json, os, pathlib, sys, time
root, revision = pathlib.Path(sys.argv[1]), sys.argv[2]
expected_size = 1056484620
expected_sha = '2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e'
path = root/'checkpoints'/'s3gen.safetensors'
def checksum(file):
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()
already_valid = path.is_file() and path.stat().st_size == expected_size and checksum(path) == expected_sha
if not already_valid:
    from huggingface_hub import hf_hub_download
    path = pathlib.Path(hf_hub_download('ResembleAI/chatterbox', 's3gen.safetensors',
                        revision=revision, local_dir=root/'checkpoints'))
assert path.stat().st_size == expected_size, 'Incomplete VC checkpoint'
assert checksum(path) == expected_sha, 'VC checkpoint SHA-256 mismatch'
receipt = {'repo':'ResembleAI/chatterbox', 'revision':revision, 'file':path.name,
           'size':expected_size, 'sha256':expected_sha, 'license':'MIT'}
(root/'logs'/'checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Exact VC checkpoint verified; no TTS checkpoints downloaded.')
PY

"$voice_env/bin/python" - "$voice_root" <<'PY'
import json, os, pathlib, sys
import importlib.metadata as md
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
root=pathlib.Path(sys.argv[1])
import numpy, soundfile, torch, torchaudio, transformers, diffusers, safetensors
from chatterbox.vc import ChatterboxVC
assert torch.__version__.split('+')[0]=='2.9.0',torch.__version__
assert torchaudio.__version__.split('+')[0]=='2.9.0',torchaudio.__version__
assert torch.version.cuda=='12.8' and torch.cuda.is_available()
x=torch.randn((64,64),device='cuda')
assert torch.isfinite(x@x).all().item()
torch.cuda.synchronize()
model=ChatterboxVC.from_local(root/'checkpoints',device='cuda')
assert model.sr==24000
record={'status':'imports_cuda_and_checkpoint_load_passed',
        'voice_inference_tested':False, 'chinese_quality_tested':False,
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'cuda':torch.version.cuda,'chatterbox':md.version('chatterbox-tts'),
        'intentional_upstream_metadata_overrides':['torch','torchaudio','diffusers','safetensors','resemble-perth git@master -> PyPI 1.0.1']}
(root/'logs'/'runtime-check.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
"$uv_bin" pip freeze --python "$voice_env/bin/python" > "$voice_root/logs/pip-freeze.txt"
if [[ -f "$script_dir/model_sources.json" ]]; then
  cp "$script_dir/model_sources.json" "$voice_root/model_sources.json"
fi
printf 'Configure the web app with:\nexport REDUB_VOICE_ROOT=%q\nexport REDUB_VOICE_ENV=%q\n' "$voice_root" "$voice_env"
printf 'Environment loaded; run a real Chinese/reference sample before treating quality as verified.\n'
