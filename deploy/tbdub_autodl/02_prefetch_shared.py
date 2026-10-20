#!/usr/bin/env python3
"""Prefetch distinct shared assets; the main downloader still verifies all files."""
import concurrent.futures
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

root = Path(os.environ.get('TBDUB_ROOT', '/root/autodl-tmp/tbdub'))
receipt = json.loads((root / 'logs' / 'download_receipt.json').read_text())
destinations = {
    'KlingTeam/X-Dub': root / 'checkpoints',
    'facebook/hubert-large-ll60k': root / 'checkpoints' / 'hubert-large-ll60k',
}


def fetch(repository):
    destination = destinations[repository['id']]
    for name in repository['files']:
        path = hf_hub_download(repo_id=repository['id'], filename=name,
                               revision=repository['revision'], local_dir=destination)
        print(f"Prefetched {repository['id']}/{name}: {Path(path).stat().st_size} bytes", flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(fetch, entry) for entry in receipt['repositories']
               if entry['id'] in destinations]
    assert len(futures) == 2, 'Start the main downloader first to pin repository revisions.'
    for future in futures:
        future.result()
print('Shared assets prefetched; final verification remains in the main downloader.', flush=True)
