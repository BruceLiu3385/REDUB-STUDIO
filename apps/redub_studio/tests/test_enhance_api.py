"""Enhancement orchestration uses small fixtures, without loading GPU models."""
from pathlib import Path
import threading
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from redub_studio import pipeline, server


def completed_source(manager, outputs=None):
    job_id = uuid.uuid4().hex
    folder = manager.jobs_dir / job_id
    folder.mkdir()
    (folder / 'final.mp4').write_bytes(b'original-video')
    job = {'id': job_id, 'status': 'completed', 'stage': 'completed', 'progress': 100,
           'message': 'done', 'error': None, 'created_at': server.now(),
           'updated_at': server.now(), 'events': [], 'source_name': 'fixture.mp4',
           'assets': {'source': uuid.uuid4().hex, 'portrait': uuid.uuid4().hex, 'voice': None},
           'options': server.JobOptions(voice_conversion=False).model_dump(),
           'outputs': {'video': 'final.mp4'} if outputs is None else outputs}
    manager.jobs[job_id] = job
    manager.cancellations[job_id] = threading.Event()
    server.write_json(folder / 'job.json', job)
    return job_id, folder


def wait_status(client, job_id, statuses):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] in statuses:
            return job
        time.sleep(.01)
    pytest.fail(f'job did not reach {statuses}: {job}')


def basic_body(manager):
    ids = {}
    for kind in ('source', 'portrait'):
        asset_id = uuid.uuid4().hex
        folder = manager.assets_dir / asset_id
        folder.mkdir()
        (folder / 'media.mp4').write_bytes(b'api-fixture')
        server.write_json(folder / 'asset.json', {'id': asset_id, 'kind': kind,
                          'file': 'media.mp4', 'name': 'fixture.mp4'})
        ids[kind + '_id'] = asset_id
    return {**ids, 'options': {'voice_conversion': False}}


@pytest.mark.parametrize('value', [1, 0, 'true', 'false', None, [], {}])
def test_enhancement_option_requires_a_real_boolean(tmp_path, value):
    with TestClient(server.create_app(tmp_path)) as client:
        body = basic_body(client.app.state.manager)
        body['options']['enhance_video'] = value
        assert client.post('/api/jobs', json=body).status_code == 422
        body['options'] = {'voice_conversion': False, 'enhance_from': 'x'}
        assert client.post('/api/jobs', json=body).status_code == 422
        body['options'] = {'voice_conversion': False, 'source_final_path': '/tmp/secret.mp4'}
        assert client.post('/api/jobs', json=body).status_code == 422


def test_old_requests_default_to_no_enhancement(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'upscale_ready', lambda *args: False)
    seen = []

    def runner(folder, assets, options, report, cancelled):
        seen.append(options)
        (folder / 'final.mp4').write_bytes(b'original')
        return {'video': 'final.mp4'}

    with TestClient(server.create_app(tmp_path, runner)) as client:
        body = basic_body(client.app.state.manager)
        response = client.post('/api/jobs', json=body)
        assert response.status_code == 201
        assert response.json()['options']['enhance_video'] is False
        wait_status(client, response.json()['id'], {'completed'})
        assert seen[0]['enhance_video'] is False
        body['options']['enhance_video'] = True
        assert client.post('/api/jobs', json=body).status_code == 503


def test_enhance_rejects_missing_model_source_and_extra_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'upscale_ready', lambda *args: False)
    with TestClient(server.create_app(tmp_path)) as client:
        source_id, _ = completed_source(client.app.state.manager)
        endpoint = f'/api/jobs/{source_id}/enhance'
        assert client.post(endpoint).status_code == 503
        monkeypatch.setattr(server, 'upscale_ready', lambda *args: True)
        assert client.post(endpoint, json={'source_final_path': '/tmp/secret.mp4'}).status_code == 422
        assert client.post(endpoint, json={'enhance_video': 'true'}).status_code == 422
        assert client.post(f'/api/jobs/{uuid.uuid4().hex}/enhance').status_code == 404
        assert client.post('/api/jobs/invalid-id/enhance').status_code == 404


@pytest.mark.parametrize('status', ['running', 'queued', 'failed', 'cancelled'])
def test_enhance_requires_a_completed_source(tmp_path, monkeypatch, status):
    monkeypatch.setattr(server, 'upscale_ready', lambda *args: True)
    with TestClient(server.create_app(tmp_path)) as client:
        manager = client.app.state.manager
        source_id, _ = completed_source(manager)
        manager.update(source_id, status=status)
        assert client.post(f'/api/jobs/{source_id}/enhance').status_code == 409


@pytest.mark.parametrize('relative', [None, 'missing.mp4', 'audio.wav', '../outside.mp4',
                                     '/tmp/outside.mp4', 'job.json'])
def test_enhance_rejects_missing_nonvideo_and_escaping_outputs(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(server, 'upscale_ready', lambda *args: True)
    with TestClient(server.create_app(tmp_path)) as client:
        outputs = {} if relative is None else {'video': relative}
        source_id, folder = completed_source(client.app.state.manager, outputs)
        (folder / 'audio.wav').write_bytes(b'audio')
        (folder.parent / 'outside.mp4').write_bytes(b'private-video')
        assert client.post(f'/api/jobs/{source_id}/enhance').status_code == 422


def test_enhancement_joins_serial_queue_and_supports_cancel_retry_and_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'upscale_ready', lambda *args: True)
    started = threading.Event()
    active = []
    enhanced = []

    def runner(folder, assets, options, report, cancelled):
        active.append('normal')
        assert active == ['normal']
        started.set()
        try:
            while not cancelled():
                time.sleep(.01)
            raise RuntimeError('cancelled fixture')
        finally:
            active.pop()

    def enhance(folder, source_video, report, cancelled):
        active.append('enhance')
        assert active == ['enhance'], 'enhancement must share the existing serial worker'
        try:
            enhanced.append(source_video)
            assert source_video.read_bytes() == b'original-video'
            report('upscale', 50, 'API fixture enhancement')
            (folder / 'final.mp4').write_bytes(source_video.read_bytes())
            (folder / 'final_hd.mp4').write_bytes(b'hd-video')
            return {'video': 'final.mp4', 'video_hd': 'final_hd.mp4'}
        finally:
            active.pop()

    monkeypatch.setattr(pipeline, 'run_enhancement', enhance)
    with TestClient(server.create_app(tmp_path, runner)) as client:
        manager = client.app.state.manager
        source_id, source_folder = completed_source(manager)
        original = client.get(f'/api/jobs/{source_id}').json()
        normal_id = client.post('/api/jobs', json=basic_body(manager)).json()['id']
        assert started.wait(2)
        response = client.post(f'/api/jobs/{source_id}/enhance')
        assert response.status_code == 201, response.text
        job = response.json()
        assert job['id'] != source_id and job['enhance_from'] == source_id
        assert job['options']['enhance_video'] is True
        assert job['assets'] == original['assets']
        assert 'path' not in str(job['options'])
        assert job['status'] == 'queued'
        assert not enhanced
        assert client.post(f"/api/jobs/{job['id']}/cancel").json()['status'] == 'cancelled'
        assert client.post(f"/api/jobs/{job['id']}/retry").json()['status'] == 'queued'
        client.post(f'/api/jobs/{normal_id}/cancel')
        wait_status(client, normal_id, {'cancelled'})
        done = wait_status(client, job['id'], {'completed'})
        assert enhanced == [(source_folder / 'final.mp4').resolve()]
        assert any(event['stage'] == 'upscale' for event in done['events'])
        assert client.get(done['download_urls']['video']).content == b'original-video'
        assert client.get(done['download_urls']['video_hd']).content == b'hd-video'
        assert client.get(f"/api/jobs/{job['id']}/download/final_hd.mp4").content == b'hd-video'
        assert client.get(f'/api/jobs/{source_id}').json() == original
        assert (source_folder / 'final.mp4').read_bytes() == b'original-video'


def test_running_enhancement_can_cancel_and_retry_without_source_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'upscale_ready', lambda *args: True)
    started = threading.Event()
    attempts = []

    def enhance(folder, source_video, report, cancelled):
        attempts.append(folder)
        if len(attempts) == 1:
            started.set()
            while not cancelled():
                time.sleep(.01)
            raise RuntimeError('cancelled fixture')
        (folder / 'final.mp4').write_bytes(source_video.read_bytes())
        (folder / 'final_hd.mp4').write_bytes(b'retried-hd')
        return {'video': 'final.mp4', 'video_hd': 'final_hd.mp4'}

    monkeypatch.setattr(pipeline, 'run_enhancement', enhance)
    with TestClient(server.create_app(tmp_path)) as client:
        source_id, folder = completed_source(client.app.state.manager)
        job_id = client.post(f'/api/jobs/{source_id}/enhance').json()['id']
        assert started.wait(2)
        assert client.post(f'/api/jobs/{job_id}/retry').status_code == 409
        client.post(f'/api/jobs/{job_id}/cancel')
        assert wait_status(client, job_id, {'cancelled'})['outputs'] == {}
        monkeypatch.setattr(server, 'upscale_ready', lambda *args: False)
        assert client.post(f'/api/jobs/{job_id}/retry').status_code == 503
        monkeypatch.setattr(server, 'upscale_ready', lambda *args: True)
        (folder / 'final.mp4').unlink()
        assert client.post(f'/api/jobs/{job_id}/retry').status_code == 422
        (folder / 'final.mp4').write_bytes(b'original-video')
        assert client.post(f'/api/jobs/{job_id}/retry').status_code == 200
        done = wait_status(client, job_id, {'completed'})
        assert attempts[0] == attempts[1]
        assert client.get(done['download_urls']['video_hd']).content == b'retried-hd'


def test_upscale_readiness_discovers_modules_without_loading_them(tmp_path, monkeypatch):
    model = tmp_path / 'RealESRGAN_x2plus.pth'
    python = tmp_path / 'bin/python'
    monkeypatch.setenv('REDUB_UPSCALE_MODEL', str(model))
    monkeypatch.setattr(server, 'environment_python', lambda variable: python)
    seen = []
    monkeypatch.setattr(server, 'module_present', lambda interpreter, module: seen.append(module) or True)
    assert server.upscale_ready() is False
    assert seen == []
    model.write_bytes(b'fixture-weight-path-only')
    assert server.upscale_ready() is True
    assert seen == ['torch', 'cv2', 'numpy']
    monkeypatch.setattr(server, 'module_present', lambda interpreter, module: module != 'cv2')
    assert server.upscale_ready() is False
