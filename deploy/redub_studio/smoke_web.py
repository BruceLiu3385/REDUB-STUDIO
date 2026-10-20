"""Exercise the actual private HTTP workflow with explicitly supplied test media.

Requires the installed GPU backends. This never powers an instance on or off.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--voice', type=Path)
    parser.add_argument('--portrait', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:7860')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--keep-background', action='store_true')
    parser.add_argument('--no-voice-conversion', action='store_true')
    parser.add_argument('--chunk-seconds', type=int, choices=[5, 10, 15], default=15)
    args = parser.parse_args()
    endpoint = urlsplit(args.endpoint)
    if not (endpoint.scheme == 'http' and endpoint.hostname in ('127.0.0.1', 'localhost')
            and endpoint.username is None and endpoint.password is None
            and endpoint.port is not None and 1024 <= endpoint.port <= 65535
            and endpoint.path in ('', '/') and not endpoint.query and not endpoint.fragment):
        raise ValueError('Smoke tests intentionally target only the local private app')
    with httpx.Client(base_url=args.endpoint, timeout=120, trust_env=False) as client:
        health = client.get('/api/health').raise_for_status().json()
        if not (health.get('gpu_ready') and (args.no_voice_conversion or health.get('voice_ready')) and health.get('ffprobe')):
            raise RuntimeError(f'Model environment not ready: {health}')
        if not args.no_voice_conversion and not args.voice:
            raise ValueError('--voice is required when voice conversion is enabled')
        assets = {}
        for kind in ('source', 'voice', 'portrait'):
            path = getattr(args, kind)
            if path is None:
                continue
            with path.open('rb') as stream:
                response = client.post('/api/assets', data={'kind': kind}, files={'file': (path.name, stream)})
            assets[kind] = response.raise_for_status().json()
        response = client.post('/api/jobs', json={f'{kind}_id': asset['id'] for kind, asset in assets.items()} |
            {'options': {'voice_conversion': not args.no_voice_conversion, 'keep_background': args.keep_background, 'chunk_seconds': args.chunk_seconds, 'fit_video': True}})
        job = response.raise_for_status().json()
        started, last = time.monotonic(), None
        while job['status'] not in ('completed', 'failed', 'cancelled'):
            key = (job['status'], job['stage'], job['message'])
            if key != last:
                print(json.dumps({field: job.get(field) for field in ('id', 'status', 'stage', 'progress', 'message', 'error')}, ensure_ascii=False), flush=True)
                last = key
            if time.monotonic() - started > args.timeout:
                client.post(f"/api/jobs/{job['id']}/cancel").raise_for_status()
                raise TimeoutError('Full workflow timed out; cancellation requested')
            time.sleep(2)
            job = client.get(f"/api/jobs/{job['id']}").raise_for_status().json()
        receipt = {'health': health, 'assets': assets, 'job': job, 'elapsed_seconds': time.monotonic()-started,
                   'note': 'Actual HTTP/model test; perceptual quality still requires review.'}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps(job, ensure_ascii=False, indent=2), flush=True)
        if job['status'] != 'completed':
            raise RuntimeError(job.get('error') or 'Workflow did not complete')
        # Check the download endpoint too, not only a server-side path.
        video = client.get(job['download_urls']['video']).raise_for_status()
        if len(video.content) < 1000:
            raise RuntimeError('Downloaded video is unexpectedly small')
        args.receipt.with_suffix('.mp4').write_bytes(video.content)


if __name__ == '__main__':
    main()
