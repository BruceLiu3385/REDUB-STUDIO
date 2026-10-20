#!/usr/bin/env python3
"""Exact upstream file downloads. No Teacher, snapshot, or full Hub cache copy."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request
from types import SimpleNamespace

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

ROOT = Path(os.environ.get('TBDUB_ROOT', '/root/autodl-tmp/tbdub')).resolve()
MODELS = ROOT / 'checkpoints'
REPO = ROOT / 'src' / 'TBDub'
STUDENT_SIZE = 12_591_523_048
STUDENT_SHA = '4e86900f9744b3650bfe58451ae89b3afba758233a3edf5947516268c1ea11e6'
LANDMARK_SHA = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
LANDMARK_SIZE = 3_758_596
LANDMARK_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
                'face_landmarker/float16/1/face_landmarker.task')
SOURCES = json.loads(Path(__file__).with_name('sources.json').read_text(encoding='utf-8'))
MODEL_REVISIONS = {item['id']: item['revision'] for item in SOURCES['model_repositories']}
EXPECTED_FILES = {item['path']: item for item in SOURCES['files']}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_asset_metadata(api, repo_id, revision, names, siblings):
    """Use per-path metadata when model_info omits optional sibling fields."""
    indexed = {item.rfilename: SimpleNamespace(size=getattr(item, 'size', None),
                                               lfs=getattr(item, 'lfs', None))
               for item in siblings}
    # get_paths_info returns RepoFile.path rather than RepoSibling.rfilename.
    # Request all selected files so LFS hashes survive incomplete model_info data.
    try:
        paths = api.get_paths_info(repo_id=repo_id, paths=names,
                                   revision=revision, repo_type='model')
    except Exception as exc:
        print(f'Per-path metadata unavailable for {repo_id} ({type(exc).__name__}); '
              'using file metadata fallback where needed.', flush=True)
        paths = []
    for item in paths:
        name = getattr(item, 'path', None)
        if name not in names:
            continue
        current = indexed.setdefault(name, SimpleNamespace(size=None, lfs=None))
        size = getattr(item, 'size', None)
        if isinstance(size, int) and size > 0:
            current.size = size
        if getattr(item, 'lfs', None):
            current.lfs = item.lfs
    for name in names:
        current = indexed.setdefault(name, SimpleNamespace(size=None, lfs=None))
        if not isinstance(current.size, int) or current.size <= 0:
            metadata = get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename=name,
                                                      revision=revision, endpoint=api.endpoint))
            assert isinstance(metadata.size, int) and metadata.size > 0, (
                f'Missing file size after per-path and HEAD metadata: {repo_id}/{name}')
            current.size = metadata.size
            # LFS ETags are SHA-256; ordinary Git-file ETags are not.
            etag = (metadata.etag or '').strip('"')
            if not current.lfs and re.fullmatch(r'[0-9a-fA-F]{64}', etag):
                current.lfs = {'sha256': etag.lower()}
        assert isinstance(current.size, int) and current.size > 0, (
            f'Missing size: {repo_id}/{name}')
    return indexed


def main() -> None:
    started = time.monotonic()
    MODELS.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    # Use the model revisions and file checksums captured from the deployed release.
    api = HfApi()
    specs = [
        ('TaoLiveAIGC/TBDub', MODELS,
         ['config.json', 'null_prompt_emb.pt', 'tbdub_student.safetensors']),
        ('KlingTeam/X-Dub', MODELS, ['Wan2.2_VAE.safetensors']),
        ('facebook/hubert-large-ll60k', MODELS / 'hubert-large-ll60k',
         ['config.json', 'preprocessor_config.json', 'pytorch_model.bin']),
    ]
    receipt_path = ROOT / 'logs' / 'download_receipt.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt = {'repositories': [], 'files': []}
    infos = []
    remaining = 0
    for repo_id, folder, files in specs:
        # Do not silently mix a previous installation with the locked release.
        prior = next((x for x in previous.get('repositories', []) if x['id'] == repo_id), None)
        revision = MODEL_REVISIONS[repo_id]
        if prior and prior['revision'] != revision:
            raise RuntimeError(f'Existing download receipt uses a different revision for {repo_id}; '
                               'review/migrate that installation explicitly before retrying.')
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        assert info.sha == revision, f'Unexpected upstream revision: {repo_id}'
        indexed = resolve_asset_metadata(api, repo_id, info.sha, files, info.siblings)
        receipt['repositories'].append({'id': repo_id, 'revision': info.sha, 'files': files})
        for name in files:
            assert name in indexed, f'Missing upstream asset: {repo_id}/{name}'
            size = indexed[name].size
            assert isinstance(size, int) and size > 0, f'Missing size: {repo_id}/{name}'
            relative = str((folder / name).relative_to(MODELS))
            assert size == EXPECTED_FILES[relative]['size'], f'Unexpected locked asset size: {relative}'
            existing = folder / name
            if not existing.exists() or existing.stat().st_size != size:
                remaining += size
        infos.append((repo_id, folder, files, info.sha, indexed))
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    # Leave room for outputs and temporary media after all pending weights.
    free = shutil.disk_usage(MODELS).free
    assert free >= remaining + 3 * 2**30, (
        f'Insufficient disk: free={free/2**30:.2f}GiB, pending={remaining/2**30:.2f}GiB + 3GiB reserve')

    def download(repo_id, folder, name, revision, indexed):
        path = Path(hf_hub_download(repo_id=repo_id, filename=name, revision=revision,
                                    local_dir=folder))
        assert path.stat().st_size == indexed[name].size, f'Incorrect size: {path}'
        digest = sha256(path)
        relative = str(path.relative_to(MODELS))
        assert digest == EXPECTED_FILES[relative]['sha256'], f'Locked SHA-256 mismatch: {path}'
        lfs = indexed[name].lfs
        expected = (lfs.get('sha256') if isinstance(lfs, dict) else
                    getattr(lfs, 'sha256', None)) if lfs else None
        if expected:
            assert digest == expected, f'Upstream LFS SHA-256 mismatch: {path}'
        receipt['files'].append({'path': str(path.relative_to(MODELS)),
                                 'size': path.stat().st_size, 'sha256': digest,
                                 'upstream_lfs_sha256': expected})
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'OK {name}: {path.stat().st_size:,} bytes', flush=True)
        return path

    # Fetch and validate the locked configuration before the 12.59GB weight.
    spec = infos[0]
    manifest_path = download(spec[0], spec[1], 'config.json', spec[3], spec[4])
    manifest = json.loads(manifest_path.read_text())
    module_spec = importlib.util.spec_from_file_location('tbdub_runtime_config', REPO/'runtime_config.py')
    assert module_spec and module_spec.loader, 'Missing official runtime_config.py'
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.validate_manifest(manifest, 'student')
    assert manifest['variants']['student']['checkpoints'] == ['tbdub_student.safetensors']
    for repo_id, folder, files, revision, indexed in infos:
        for name in files:
            if repo_id == spec[0] and name == 'config.json':
                continue
            download(repo_id, folder, name, revision, indexed)

    student = MODELS/'tbdub_student.safetensors'
    assert student.stat().st_size == STUDENT_SIZE
    assert next(x['sha256'] for x in receipt['files']
                if x['path'] == student.name) == STUDENT_SHA, 'Unexpected Student release'
    landmark = MODELS/'face_landmarker.task'
    if not landmark.exists() or landmark.stat().st_size != LANDMARK_SIZE or sha256(landmark) != LANDMARK_SHA:
        temporary = MODELS/'face_landmarker.task.part'
        with urllib.request.urlopen(LANDMARK_URL, timeout=120) as response, temporary.open('wb') as target:
            shutil.copyfileobj(response, target)
        assert temporary.stat().st_size == LANDMARK_SIZE
        assert sha256(temporary) == LANDMARK_SHA
        temporary.replace(landmark)
    receipt['files'].append({'path': landmark.name, 'size': LANDMARK_SIZE, 'sha256': LANDMARK_SHA})
    receipt['elapsed_seconds'] = round(time.monotonic() - started, 2)
    receipt['total_bytes'] = sum(x['size'] for x in receipt['files'])
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'total_gb': receipt['total_bytes']/1e9,
                      'elapsed_seconds': receipt['elapsed_seconds'],
                      'receipt': str(receipt_path)}, indent=2))


if __name__ == '__main__':
    main()
