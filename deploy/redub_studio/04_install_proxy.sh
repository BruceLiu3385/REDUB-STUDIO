#!/usr/bin/env bash
# Check the real Jupyter Server environment. Default: inspect + pip dry-run only.
# --install installs the pinned extension; never restarts services or edits auth.
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: bash 04_install_proxy.sh --python /absolute/jupyter-env/bin/python [--install]

Set REDUB_JUPYTER_PYTHON instead of --python if desired.
Use the interpreter running Jupyter Server, not the Web/model kernel environment.
Default: report package/extension status, then pip --dry-run; no package install.
--install: apply the reviewed package install, then report extension status.
No Jupyter/service restart, authentication change, public listener or model setup.
HELP
}

jupyter_python="${REDUB_JUPYTER_PYTHON:-}"
install=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      jupyter_python="$2"; shift 2 ;;
    --install) install=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$jupyter_python" == /* && -x "$jupyter_python" ]] || {
  printf 'Provide the verified absolute Jupyter Server Python path with --python.\n' >&2
  exit 2
}
# Do not realpath this Python symlink: that would bypass its virtualenv.
scratch="$(mktemp -d "${TMPDIR:-/tmp}/redub-proxy-check.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT
constraints="$scratch/existing-jupyter-constraints.txt"

"$jupyter_python" - "$constraints" <<'PY'
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit('jupyter-server-proxy 4.6.0 requires Python >=3.10.')
if importlib.util.find_spec('jupyter_server') is None:
    raise SystemExit('No jupyter_server in this interpreter. Verify the running Server environment first.')
packages = ('jupyter-server', 'notebook', 'jupyterlab', 'jupyterlab-server',
            'jupyter-core', 'jupyter-client', 'tornado', 'traitlets',
            'aiohttp', 'simpervisor', 'jupyter-server-proxy')
versions = {}
for name in packages:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None
protected = {'jupyter-server', 'notebook', 'jupyterlab', 'jupyterlab-server',
             'jupyter-core', 'jupyter-client', 'tornado', 'traitlets'}
Path(sys.argv[1]).write_text(''.join(
    f'{name}=={version}\n' for name, version in versions.items()
    if name in protected and version), encoding='utf-8')
print(json.dumps({'python': sys.executable, 'prefix': sys.prefix, 'versions': versions}, indent=2))
print('Existing Jupyter core versions are constrained; an incompatible plan will fail instead of upgrading them.')
PY

"$jupyter_python" -m pip --version
printf 'Current extension state (this does not load an extension into the running server):\n'
"$jupyter_python" -m jupyter_server extension list
printf 'Dependency plan only; existing Jupyter core versions remain fixed:\n'
"$jupyter_python" -m pip install --disable-pip-version-check --no-cache-dir \
  --only-binary=:all: --constraint "$constraints" --dry-run 'jupyter-server-proxy==4.6.0'

if [[ "$install" != 1 ]]; then
  printf 'Dry-run complete. No packages installed; no service restarted.\n'
  printf 'After reviewing the plan, rerun with the same --python and --install.\n'
  exit 0
fi

"$jupyter_python" -m pip install --disable-pip-version-check --no-cache-dir \
  --only-binary=:all: --constraint "$constraints" 'jupyter-server-proxy==4.6.0'
"$jupyter_python" -c 'import importlib.metadata as m; assert m.version("jupyter-server-proxy")=="4.6.0"; print("Installed jupyter-server-proxy 4.6.0")'
"$jupyter_python" -m jupyter_server extension list
printf 'No service was restarted and no authentication settings were changed.\n'
printf 'If extension list shows disabled, explicitly enable it with the SAME interpreter:\n'
printf '  %q -m jupyter_server extension enable --sys-prefix --py jupyter_server_proxy\n' "$jupyter_python"
printf 'Save work and inspect the exact service manager before restarting only Jupyter Server.\n'
printf 'Then sign in through the original Jupyter entry and open <jupyter-base>/proxy/7860/.\n'
printf 'Keep Redub on 127.0.0.1:7860; keep existing auth and XSRF checks enabled.\n'
