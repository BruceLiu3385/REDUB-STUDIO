#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
use_tbdub_env
python "$TBDUB_SCRIPT_DIR/02_download_student.py" 2>&1 | tee "$TBDUB_ROOT/logs/download.log"
