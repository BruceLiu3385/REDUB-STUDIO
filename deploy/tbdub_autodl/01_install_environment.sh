#!/usr/bin/env bash
# Execute on the rented GPU instance, not on the local workstation.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
tbdub_env_path="$(realpath -m "$TBDUB_ENV")"
[[ "$tbdub_env_path" != / ]] || { printf 'Environment may not be the filesystem root.\n' >&2; exit 2; }
for protected_env in "${REDUB_WEB_ENV:-${REDUB_WEB_ROOT:-/root/autodl-tmp/redub-web}/env}" "${REDUB_VOICE_ENV:-${REDUB_VOICE_ROOT:-/root/autodl-tmp/redub-voice}/env}" "${REDUB_SEPARATOR_ENV:-${REDUB_SEPARATOR_ROOT:-/root/autodl-tmp/redub-separator}/env}"; do
  protected_env="$(realpath -m "$protected_env")"
  if [[ "$tbdub_env_path" == "$protected_env" || "$tbdub_env_path" == "$protected_env/"* || "$protected_env" == "$tbdub_env_path/"* ]]; then
    printf 'Refusing to install TBDub dependencies into an overlapping component environment.\n' >&2
    exit 2
  fi
done
bash "$TBDUB_SCRIPT_DIR/00_check_hardware.sh"
command -v git >/dev/null
TBDUB_GIT_REF="${TBDUB_GIT_REF:-19e18e4ba8e2a19e6716cb38110ee2de1dae9bb3}"
[[ "$TBDUB_GIT_REF" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'TBDUB_GIT_REF must be an immutable 40-character commit SHA.\n' >&2; exit 2;
}
if [[ ! -d "$TBDUB_REPO/.git" ]]; then
  mkdir -p "$TBDUB_ROOT/src"
  git clone --filter=blob:none --no-checkout https://github.com/TaoLiveAIGC/TBDub.git "$TBDUB_REPO"
  git -C "$TBDUB_REPO" fetch origin "$TBDUB_GIT_REF"
  git -C "$TBDUB_REPO" checkout --detach "$TBDUB_GIT_REF"
else
  [[ "$(git -C "$TBDUB_REPO" remote get-url origin)" == 'https://github.com/TaoLiveAIGC/TBDub.git' ]]
  [[ "$(git -C "$TBDUB_REPO" rev-parse HEAD)" == "$TBDUB_GIT_REF" ]] || {
    printf 'Existing TBDub clone is at a different revision; review/migrate it explicitly. No checkout was changed.\n' >&2
    exit 2
  }
  [[ -z "$(git -C "$TBDUB_REPO" status --porcelain --untracked-files=no)" ]] || {
    printf 'Existing TBDub tracked files contain local changes; review them before installing.\n' >&2
    exit 2
  }
fi
# Do not silently replace a reviewed dependency list or update an existing clone.
diff -u -B "$TBDUB_SCRIPT_DIR/requirements.official-v1.1.txt" "$TBDUB_REPO/requirements.txt"
git -C "$TBDUB_REPO" rev-parse HEAD > "$TBDUB_ROOT/logs/code_revision.txt"

if [[ ! -x "$TBDUB_ENV/bin/python" ]]; then
  conda_bin="$(command -v conda || true)"
  for candidate in /root/miniconda3/bin/conda /opt/conda/bin/conda; do
    if [[ -z "$conda_bin" && -x "$candidate" ]]; then conda_bin="$candidate"; fi
  done
  [[ -n "$conda_bin" ]] || { printf 'Conda is required from the AutoDL base image.\n' >&2; exit 1; }
  "$conda_bin" create --yes --prefix "$TBDUB_ENV" --override-channels -c conda-forge python=3.10.16 pip=25.3
fi
use_tbdub_env
python -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$TBDUB_SCRIPT_DIR/requirements.official-v1.1.txt"
python -m pip check
python - "$TBDUB_ENV" <<'PY'
import importlib.metadata as md, os, pathlib, subprocess, sys
import imageio_ffmpeg
env = pathlib.Path(sys.argv[1])
os.environ.pop('IMAGEIO_FFMPEG_EXE', None)
binary = pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
version = subprocess.check_output([str(binary), '-version'], text=True).splitlines()[0]
assert '7.0.2' in version, f'Unexpected bundled ffmpeg: {version}'
link = env/'bin'/'ffmpeg'
if link.is_symlink():
    assert link.resolve() == binary, f'Unexpected ffmpeg symlink: {link}'
elif link.exists():
    raise SystemExit(f'Will not overwrite existing executable: {link}')
else:
    link.symlink_to(binary)
installed = {d.metadata['Name'].lower().replace('_','-') for d in md.distributions()}
conflicts = installed & {'opencv-python','opencv-python-headless','opencv-contrib-python-headless'}
assert not conflicts, f'Conflicting cv2 packages: {conflicts}'
print(version)
PY
python - <<'PY'
import cv2, jax, mediapipe, torch, torchvision, transformers
assert torch.__version__.split('+')[0] == '2.9.0'
assert torchvision.__version__.split('+')[0] == '0.24.0'
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert all(d.platform == 'cpu' for d in jax.devices()), jax.devices()
x = torch.randn((128,128), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print('Validated CUDA/BF16, CPU JAX, MediaPipe, OpenCV and Transformers imports.')
PY
python -m pip freeze > "$TBDUB_ROOT/logs/pip-freeze.txt"
printf 'Environment ready: %s\nCode revision: ' "$TBDUB_ENV"
cat "$TBDUB_ROOT/logs/code_revision.txt"
df -h "$TBDUB_ROOT"
