#!/usr/bin/env bash
# Source this file; all large assets live on AutoDL's persistent data disk.
set -Eeuo pipefail
TBDUB_ROOT="${TBDUB_ROOT:-${REDUB_TBDUB_ROOT:-/root/autodl-tmp/tbdub}}"
TBDUB_ENV="${TBDUB_ENV:-${REDUB_TBDUB_ENV:-$TBDUB_ROOT/env}}"
TBDUB_REPO="$TBDUB_ROOT/src/TBDub"
TBDUB_CHECKPOINTS="$TBDUB_ROOT/checkpoints"
TBDUB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TBDUB_ROOT TBDUB_ENV TBDUB_REPO TBDUB_CHECKPOINTS
export HF_HOME="$TBDUB_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_DISABLE_XET=1
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CONDA_PKGS_DIRS="$TBDUB_ROOT/cache/conda-pkgs"
export TMPDIR="$TBDUB_ROOT/tmp"
export PYTHONNOUSERSITE=1
# Notebook kernels export their own inline backend, which is absent from the
# isolated inference environment. MediaPipe only needs a non-interactive backend.
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p "$TBDUB_ROOT" "$TMPDIR" "$TBDUB_ROOT/logs" "$TBDUB_ROOT/results"

use_tbdub_env() {
  if [[ ! -x "$TBDUB_ENV/bin/python" ]]; then
    printf 'Missing environment: %s. Run 01_install_environment.sh first.\n' "$TBDUB_ENV" >&2
    return 1
  fi
  export PATH="$TBDUB_ENV/bin:$PATH"
  export IMAGEIO_FFMPEG_EXE="$TBDUB_ENV/bin/ffmpeg"
}
