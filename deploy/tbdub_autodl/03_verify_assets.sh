#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
use_tbdub_env
python - "$TBDUB_ROOT" <<'PY'
import hashlib, importlib.util, json, pathlib, sys
from safetensors import safe_open
root = pathlib.Path(sys.argv[1])
models = root/'checkpoints'
receipt = json.loads((root/'logs'/'download_receipt.json').read_text())
assert len(receipt['files']) == 8, 'Download incomplete'
for item in receipt['files']:
    p = models/item['path']
    assert p.is_file() and p.stat().st_size == item['size'], p
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    assert digest.hexdigest() == item['sha256'], f'Hash mismatch: {p}'
    print('Verified', item['path'])
spec = importlib.util.spec_from_file_location('tbdub_runtime_config', root/'src'/'TBDub'/'runtime_config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.validate_manifest(json.loads((models/'config.json').read_text()), 'student')
assert json.loads((models/'hubert-large-ll60k'/'config.json').read_text())['model_type'] == 'hubert'
with safe_open(models/'tbdub_student.safetensors', framework='pt', device='cpu') as model:
    assert len(model.keys()) == 1245, len(model.keys())
    assert all(model.get_slice(k).get_dtype() == 'BF16' for k in model.keys())
print('Student safetensors header, manifest, HuBERT config and all file hashes verified.')
PY
cd "$TBDUB_REPO"
python inference.py --help > "$TBDUB_ROOT/logs/inference-help.txt"
df -h "$TBDUB_ROOT"
