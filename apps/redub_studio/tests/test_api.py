"""API isolation tests. Stub runners test orchestration, not GPU/media quality."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid
import wave

from fastapi.testclient import TestClient
import pytest

from redub_studio import server


def seed(manager, kind):
    asset_id = uuid.uuid4().hex
    folder = manager.assets_dir / asset_id
    folder.mkdir()
    (folder / 'media.mp4').write_bytes(b'api-test-fixture')
    server.write_json(folder / 'asset.json', {'id': asset_id, 'kind': kind,
        'file': 'media.mp4', 'name': 'fixture.mp4', 'bytes': 16,
        'duration': 10, 'width': 512, 'height': 512, 'has_audio': True})
    return asset_id


def request_body(manager):
    return {'source_id': seed(manager, 'source'), 'portrait_id': seed(manager, 'portrait'),
            'voice_id': seed(manager, 'voice'), 'options': {}}


def wait_status(client, job_id, statuses):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] in statuses:
            return job
        time.sleep(.01)
    pytest.fail(f'job did not reach {statuses}: {job}')


def test_health_without_config_does_not_claim_gpu(tmp_path, monkeypatch):
    for key in ('REDUB_TBDUB_ROOT', 'REDUB_TBDUB_ENV', 'REDUB_VOICE_ROOT', 'REDUB_VOICE_ENV'):
        monkeypatch.delenv(key, raising=False)
    app = server.create_app(tmp_path)
    with TestClient(app) as client:
        data = client.get('/api/health').json()
        assert data['status'] == 'ok'
        assert data['gpu_ready'] is False
        assert data['voice_ready'] is False
        assert data['limits']['max_duration'] == 600
        assert data['limits']['chunk_seconds'] == 15


def test_upload_uses_uuid_paths_and_does_not_trust_filename(tmp_path, monkeypatch):
    # Isolate upload persistence; this does not test real ffprobe or inference.
    monkeypatch.setattr(server, 'probe_media', lambda path, kind: {
        'duration': 10, 'width': 512, 'height': 512, 'has_audio': True})
    app = server.create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/assets', data={'kind': 'source'},
                               files={'file': ('../../evil.mp4', b'test-media', 'video/mp4')})
        assert response.status_code == 201, response.text
        asset = response.json()
        assert asset['name'] == 'evil.mp4'
        assert server.ID_RE.fullmatch(asset['id'])
        assert (tmp_path / 'assets' / asset['id'] / 'media.mp4').read_bytes() == b'test-media'
        assert not (tmp_path / 'evil.mp4').exists()
        assert 'file' not in asset
        response = client.post('/api/assets', data={'kind': 'portrait'},
                               files={'file': ('face.png', b'not-a-video', 'image/png')})
        assert response.status_code == 422


def test_upload_limit_and_missing_probe_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'MAX_UPLOAD', 12)
    app = server.create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/assets', data={'kind': 'source'},
                               files={'file': ('large.mp4', b'x' * 13, 'video/mp4')})
        assert response.status_code == 413
        assert list((tmp_path / 'assets').iterdir()) == []
        monkeypatch.setattr(server, 'executable', lambda *args: None)
        response = client.post('/api/assets', data={'kind': 'source'},
                               files={'file': ('small.mp4', b'x', 'video/mp4')})
        assert response.status_code == 503
        assert list((tmp_path / 'assets').iterdir()) == []


def test_jobs_validate_asset_kind_and_voice_requirement(tmp_path):
    app = server.create_app(tmp_path)
    with TestClient(app) as client:
        manager = app.state.manager
        body = request_body(manager)
        body['portrait_id'] = body['source_id']
        assert client.post('/api/jobs', json=body).status_code == 422
        body['portrait_id'] = seed(manager, 'portrait')
        body['voice_id'] = None
        assert client.post('/api/jobs', json=body).status_code == 422
        body['options']['unexpected_command'] = 'execute'
        assert client.post('/api/jobs', json=body).status_code == 422
        assert client.get('/api/jobs/../../etc/passwd').status_code == 404


def test_queue_serializes_cancel_and_retry(tmp_path):
    first_started = threading.Event()
    active = []
    seen = []

    def runner(folder, assets, options, report, cancelled):
        active.append(folder.name)
        assert len(active) == 1
        seen.append(folder.name)
        if len(seen) == 1:
            first_started.set()
            while not cancelled():
                time.sleep(.01)
            active.pop()
            raise RuntimeError('cancelled test runner')
        (folder / 'final.mp4').write_bytes(b'test-output')
        (folder / 'quality_report.json').write_text('{"full_decode": "passed"}')
        active.pop()
        return {'video': 'final.mp4', 'report': 'quality_report.json'}

    app = server.create_app(tmp_path, runner)
    with TestClient(app) as client:
        body = request_body(app.state.manager)
        first = client.post('/api/jobs', json=body).json()['id']
        assert first_started.wait(2)
        second = client.post('/api/jobs', json=body).json()['id']
        assert client.post(f'/api/jobs/{first}/retry').status_code == 409
        assert client.post(f'/api/jobs/{second}/cancel').json()['status'] == 'cancelled'
        assert client.post(f'/api/jobs/{second}/retry').json()['status'] == 'queued'
        client.post(f'/api/jobs/{first}/cancel')
        assert wait_status(client, first, {'cancelled'})['outputs'] == {}
        done = wait_status(client, second, {'completed'})
        assert seen == [first, second]
        assert done['outputs'] == {'video': 'final.mp4', 'report': 'quality_report.json'}
        assert client.get(done['download_urls']['video']).content == b'test-output'
        assert client.get(done['download_urls']['report']).json()['full_decode'] == 'passed'
        assert client.get(f'/api/jobs/{second}/download/final.mp4').content == b'test-output'
        assert client.get(f'/api/jobs/{second}/download/job.json').status_code == 404
        assert client.post(f'/api/jobs/{second}/retry').status_code == 409


def test_retry_reuses_job_directory_and_clears_error(tmp_path):
    attempts = []

    def runner(folder, assets, options, report, cancelled):
        attempts.append(folder)
        partial = folder / 'saved-chunk.bin'
        if len(attempts) == 1:
            partial.write_bytes(b'preserved')
            raise RuntimeError('deliberate API queue test failure')
        assert partial.read_bytes() == b'preserved'
        (folder / 'final.mp4').write_bytes(b'done')
        return {'video': 'final.mp4'}

    app = server.create_app(tmp_path, runner)
    with TestClient(app) as client:
        job_id = client.post('/api/jobs', json=request_body(app.state.manager)).json()['id']
        assert wait_status(client, job_id, {'failed'})['error']
        assert client.post(f'/api/jobs/{job_id}/retry').status_code == 200
        done = wait_status(client, job_id, {'completed'})
        assert attempts[0] == attempts[1]
        assert done['error'] is None
        assert len(client.get('/api/jobs').json()) == 1


@pytest.mark.parametrize('escape', ['../outside.mp4', '/tmp/absolute.mp4', 'linked.mp4'])
def test_pipeline_cannot_publish_escape_paths(tmp_path, escape):
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'private')

    def runner(folder, assets, options, report, cancelled):
        if escape == 'linked.mp4':
            (folder / escape).symlink_to(outside)
        return {'video': escape}

    app = server.create_app(tmp_path, runner)
    with TestClient(app) as client:
        job_id = client.post('/api/jobs', json=request_body(app.state.manager)).json()['id']
        job = wait_status(client, job_id, {'failed'})
        assert job['outputs'] == {}
        assert client.get(f'/api/jobs/{job_id}/download/video').status_code == 404


def test_audio_marker_allows_prelisten_and_rejects_arbitrary_files(tmp_path):
    ready = threading.Event()

    def runner(folder, assets, options, report, cancelled):
        (folder / 'source.wav').write_bytes(b'original-audio')
        (folder / 'converted.wav').write_bytes(b'converted-audio')
        server.write_json(folder / 'audio_ready.json', {
            'source_audio': 'source.wav', 'audio': 'converted.wav', 'private': 'job.json'})
        report('lipsync', 50, 'API orchestration test only')
        ready.set()
        while not cancelled():
            time.sleep(.01)
        raise RuntimeError('cancelled')

    app = server.create_app(tmp_path, runner)
    with TestClient(app) as client:
        job_id = client.post('/api/jobs', json=request_body(app.state.manager)).json()['id']
        assert ready.wait(2)
        job = client.get(f'/api/jobs/{job_id}').json()
        assert job['status'] == 'running'
        assert job['outputs'] == {'source_audio': 'source.wav', 'audio': 'converted.wav'}
        assert client.get(job['download_urls']['audio']).content == b'converted-audio'
        assert client.get(f'/api/jobs/{job_id}/download/private').status_code == 404
        client.post(f'/api/jobs/{job_id}/cancel')
        job = wait_status(client, job_id, {'cancelled'})
        assert client.get(job['download_urls']['source_audio']).content == b'original-audio'


def test_restart_does_not_automatically_repeat_inference(tmp_path):
    folder = tmp_path / 'jobs' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    server.write_json(folder / 'job.json', {
        'id': folder.name, 'status': 'running', 'stage': 'lipsync', 'progress': 50,
        'message': 'interrupted', 'error': None, 'outputs': {}, 'events': [],
        'created_at': server.now(), 'updated_at': server.now(), 'assets': {}, 'options': {}})
    with TestClient(server.create_app(tmp_path)) as client:
        job = client.get(f'/api/jobs/{folder.name}').json()
        assert job['status'] == 'failed'
        assert job['stage'] == 'interrupted'


@pytest.mark.parametrize('kind,container,video,audio,expected_status', [
    ('source', 20, '20', '10', 201),
    ('source', 700, '700', '10', 201),
    ('source', 20, '20', '601', 422),
    ('portrait', 700, '15', '700', 201),
    ('portrait', 20, '661', '10', 422),
    ('voice', 100, '100', '10', 201),
    ('voice', 10, '10', '61', 422),
])
def test_upload_limits_use_relevant_stream_duration(tmp_path, monkeypatch, kind,
                                                   container, video, audio, expected_status):
    # Stub only ffprobe JSON so this exercises API validation, not media decoding.
    raw = {'format': {'format_name': 'mov,mp4,m4a,3gp,3g2,mj2', 'duration': str(container)},
           'streams': [{'codec_type': 'video', 'duration': video, 'width': 512, 'height': 512},
                       {'codec_type': 'audio', 'duration': audio}]}
    monkeypatch.setattr(server, 'executable', lambda *args: '/test/ffprobe')
    def probe(command, **kwargs):
        assert command[command.index('-protocol_whitelist') + 1] == 'file,pipe'
        return SimpleNamespace(returncode=0, stdout=json.dumps(raw))
    monkeypatch.setattr(server.subprocess, 'run', probe)
    with TestClient(server.create_app(tmp_path)) as client:
        response = client.post('/api/assets', data={'kind': kind},
                               files={'file': ('fixture.mp4', b'api-fixture', 'video/mp4')})
        assert response.status_code == expected_status, response.text
        if expected_status == 201:
            asset = response.json()
            assert asset['duration'] == container
            assert asset['video_duration'] == float(video)
            assert asset['audio_duration'] == float(audio)


@pytest.mark.parametrize('stream,expected', [
    ({'codec_type': 'audio', 'duration': 'N/A', 'duration_ts': 480000, 'time_base': '1/48000'}, 10),
    ({'codec_type': 'audio', 'tags': {'DURATION': '00:00:10.500000000'}}, 10.5),
    ({'codec_type': 'video', 'nb_frames': '375', 'avg_frame_rate': '25/1', 'time_base': '1/12800'}, 15),
    ({'codec_type': 'video', 'nb_frames': '375', 'avg_frame_rate': '0/0', 'r_frame_rate': '25/1'}, 15),
])
def test_stream_duration_fallbacks(stream, expected):
    assert server.stream_duration(stream) == expected


def test_pcm_wave_uses_exact_samples_instead_of_rounded_probe(tmp_path, monkeypatch):
    path = tmp_path / 'reference.wav'
    with wave.open(str(path), 'wb') as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(8000)
        target.writeframes(b'\0\0' * 40000)
    raw = {'format': {'format_name': 'wav', 'duration': '59'},
           'streams': [{'codec_type': 'audio', 'duration': '59'}]}
    monkeypatch.setattr(server, 'executable', lambda *args: '/test/ffprobe')
    monkeypatch.setattr(server.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(raw)))
    data = server.probe_media(path, 'voice')
    assert data['audio_duration'] == 5
    assert data['duration'] == 5
    assert data['video_duration'] == 0


def test_source_revision_is_checked_against_actual_git_head(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=API Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '--allow-empty',
                    '-qm', 'fixture source revision'], check=True)
    revision = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    assert server.source_revision(tmp_path, revision)['verified'] is True
    assert server.source_revision(tmp_path, '0' * 40)['verified'] is False
    assert server.source_revision(tmp_path, None)['verified'] is False


def test_voice_health_does_not_accept_unversioned_source(tmp_path, monkeypatch):
    root = tmp_path / 'voice'
    (root / 'env/bin').mkdir(parents=True)
    (root / 'env/bin/python').symlink_to(sys.executable)
    (root / 'checkpoints').mkdir()
    (root / 'checkpoints/s3gen.safetensors').write_bytes(b'not-a-model')
    (root / 'src/chatterbox/src/chatterbox').mkdir(parents=True)
    (root / 'src/chatterbox/src/chatterbox/vc.py').write_text('# API fixture only')
    monkeypatch.setenv('REDUB_VOICE_ROOT', str(root))
    monkeypatch.setenv('REDUB_VOICE_ENV', str(root / 'env'))
    monkeypatch.setattr(server, 'module_present', lambda *args: True)
    with TestClient(server.create_app(tmp_path / 'data')) as client:
        health = client.get('/api/health').json()
        assert health['voice_ready'] is False
        assert health['source_revisions']['voice']['verified'] is False


def test_environment_python_preserves_virtualenv_symlink_path(tmp_path, monkeypatch):
    environment = tmp_path / 'voice-env'
    python = environment / 'bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv('REDUB_VOICE_ENV', str(environment))
    selected = server.environment_python('REDUB_VOICE_ENV')
    assert selected == python.absolute()
    assert selected != python.resolve(), 'Resolving this link would bypass virtualenv package discovery.'
