"""Real container/timeline checks with only the AI upscale subprocess replaced.

These tests exercise FFmpeg, ffprobe, cache invalidation and cancellation. They
do not assess neural model quality or suggest that a scale filter is AI upscale.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from redub_studio import pipeline
from redub_studio.media import CancelledError, count_video_frames, executable, probe


def media_tools_available():
    try:
        executable('ffmpeg')
        executable('ffprobe')
        return True
    except RuntimeError:
        return False


real_media = pytest.mark.skipif(
    not media_tools_available(), reason='requires real FFmpeg and ffprobe')


def ff(*args):
    subprocess.run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y',
                    *map(str, args)], check=True, capture_output=True)


def make_video(path, frequency=440):
    # A non-round half-second duration yields 13 real frames on the 25 fps grid.
    ff('-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=25:duration=0.52',
       '-f', 'lavfi', '-i',
       f'sine=frequency={frequency}:sample_rate=48000:duration=0.52',
       '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-pix_fmt',
       'yuv420p', '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', path)


def decoded_audio_hash(path):
    result = subprocess.run(
        [executable('ffmpeg'), '-nostdin', '-v', 'error', '-i', str(path),
         '-map', '0:a:0', '-c:a', 'pcm_s16le', '-f', 'hash', '-hash',
         'sha256', '-'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def media_setup(tmp_path, monkeypatch):
    if not media_tools_available():
        pytest.skip('requires real FFmpeg and ffprobe')
    source = tmp_path / 'source.mp4'
    make_video(source)
    model = tmp_path / 'model.pth'
    model.write_bytes(b'fixture-model-001')
    monkeypatch.setenv('REDUB_UPSCALE_MODEL', str(model))
    monkeypatch.setattr(
        pipeline, 'model_environment',
        lambda name: (tmp_path, Path(sys.executable), os.environ.copy()))
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'logs').mkdir()
    return source, model, job


def install_worker(monkeypatch, *, mode='valid', after_write=None):
    """Replace only the AI process; retain real validation and media tools."""
    calls = []
    original_run = pipeline.run

    def worker(command, **kwargs):
        if len(command) < 2 or Path(command[1]).name != 'upscale.py':
            return original_run(command, **kwargs)
        arguments = list(map(str, command))
        get = lambda flag: Path(arguments[arguments.index(flag) + 1])
        source, target = get('--input'), get('--output')
        calls.append(source)
        if mode == 'cancel':
            raise CancelledError('fixture worker cancelled')
        if mode == 'corrupt':
            target.write_bytes(b'this is not a playable video')
        else:
            filters = 'scale=iw*2:ih*2'
            if mode == 'short_frames':
                filters = 'trim=end_frame=12,setpts=PTS-STARTPTS,' + filters
            args = ['-i', source, '-map', '0:v:0', '-vf', filters,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p']
            if mode == 'no_audio':
                args += ['-an']
            elif mode == 'short_audio':
                args += ['-map', '0:a:0', '-af',
                         'atrim=duration=0.12,asetpts=PTS-STARTPTS', '-c:a', 'aac']
            else:
                args += ['-map', '0:a:0', '-c:a', 'copy']
            ff(*args, '-movflags', '+faststart', target)
        get('--progress').write_text(json.dumps({'frames': 13}))
        get('--report').write_text(
            '{' if mode == 'bad_report' else json.dumps({'method': 'test-only FFmpeg scale'}))
        if kwargs.get('heartbeat'):
            kwargs['heartbeat']()
        if after_write:
            after_write()
        return ''

    monkeypatch.setattr(pipeline, 'run', worker)
    return calls


def test_missing_model_has_actionable_error_before_starting_worker(tmp_path, monkeypatch):
    monkeypatch.setenv('REDUB_UPSCALE_MODEL', str(tmp_path / 'missing.pth'))
    events = []
    with pytest.raises(RuntimeError, match='高清模型尚未安装'):
        pipeline.enhance_video_file(tmp_path, tmp_path / 'unused.mp4',
                                    lambda *event: events.append(event), lambda: False)
    assert not events
    assert not (tmp_path / 'upscale-complete.json').exists()


@real_media
def test_enhance_existing_video_keeps_original_frames_and_exact_audio(media_setup, monkeypatch):
    source, _, job = media_setup
    calls = install_worker(monkeypatch)
    events = []
    outputs = pipeline.run_enhancement(
        job, source, lambda *event: events.append(event), lambda: False)
    assert outputs == {'video': 'final.mp4', 'video_hd': 'final_hd.mp4',
                       'report': 'quality_report.json'}
    original, enhanced = job / outputs['video'], job / outputs['video_hd']
    assert original.read_bytes() == source.read_bytes()
    assert calls == [original], 'enhancement must not invoke voice or lip-sync inference'
    assert count_video_frames(original) == count_video_frames(enhanced) == 13
    info = probe(enhanced)
    assert (info['width'], info['height']) == (128, 96)
    assert info['has_audio']
    assert decoded_audio_hash(enhanced) == decoded_audio_hash(original)
    report = json.loads((job / outputs['report']).read_text())
    assert report['full_decode'] == 'passed'
    assert report['audio_duration_checked'] is True
    assert report['frames'] == 13 and report['fps'] == 25
    assert events[-1][0:2] == ('completed', 100)
    assert (job / 'upscale-complete.json').is_file()


@real_media
@pytest.mark.parametrize('mode', ['short_frames', 'no_audio', 'short_audio', 'corrupt'])
def test_reject_invalid_worker_output_without_marking_complete(media_setup, monkeypatch, mode):
    source, _, job = media_setup
    calls = install_worker(monkeypatch, mode=mode)
    events = []
    with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
        pipeline.run_enhancement(
            job, source, lambda *event: events.append(event), lambda: False)
    assert len(calls) == 1
    assert (job / 'final.mp4').read_bytes() == source.read_bytes()
    assert not (job / 'upscale-complete.json').exists()
    assert not (job / 'quality_report.json').exists()
    assert all(event[0] != 'completed' for event in events)


@real_media
def test_cached_result_is_reused_but_still_fully_decoded(media_setup, monkeypatch):
    source, _, job = media_setup
    calls = install_worker(monkeypatch)
    first, _ = pipeline.enhance_video_file(job, source, lambda *_: None, lambda: False)
    decode_checks = []
    original_ffmpeg = pipeline.ffmpeg

    def verify(args, log, cancelled):
        decode_checks.append(list(args))
        return original_ffmpeg(args, log, cancelled)

    monkeypatch.setattr(pipeline, 'ffmpeg', verify)
    cached, quality = pipeline.enhance_video_file(job, source, lambda *_: None, lambda: False)
    assert first == cached
    assert len(calls) == 1
    assert quality['full_decode'] == 'passed'
    assert any('-xerror' in args and 'null' in args for args in decode_checks)


@real_media
@pytest.mark.parametrize('changed', ['model', 'source'])
def test_cache_cannot_reuse_different_model_or_source_with_same_stat(
        media_setup, monkeypatch, changed):
    source, model, job = media_setup
    replacement = source.parent / ('other-model.pth' if changed == 'model' else 'other-source.mp4')
    original = model if changed == 'model' else source
    if changed == 'model':
        replacement.write_bytes(b'fixture-model-002')
    else:
        make_video(replacement, frequency=880)
        # Both remain valid MP4s; equal file size and timestamp must not identify
        # two different videos. Padding only affects bytes after the MP4 boxes.
        size = max(source.stat().st_size, replacement.stat().st_size)
        for path in (source, replacement):
            with path.open('ab') as stream:
                stream.write(b'\0' * (size - path.stat().st_size))
    stat = original.stat()
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert replacement.stat().st_size == original.stat().st_size
    assert replacement.stat().st_mtime_ns == original.stat().st_mtime_ns
    assert replacement.read_bytes() != original.read_bytes()
    calls = install_worker(monkeypatch)
    pipeline.enhance_video_file(job, source, lambda *_: None, lambda: False)
    if changed == 'model':
        monkeypatch.setenv('REDUB_UPSCALE_MODEL', str(replacement))
    else:
        source = replacement
    pipeline.enhance_video_file(job, source, lambda *_: None, lambda: False)
    assert len(calls) == 2, 'a completed result belongs to its actual source and model'


@real_media
def test_corrupt_cached_video_is_regenerated(media_setup, monkeypatch):
    source, _, job = media_setup
    calls = install_worker(monkeypatch)
    target, _ = pipeline.enhance_video_file(job, source, lambda *_: None, lambda: False)
    target.write_bytes(b'damaged cached video')
    target, quality = pipeline.enhance_video_file(job, source, lambda *_: None, lambda: False)
    assert len(calls) == 2
    assert count_video_frames(target) == 13
    assert quality['full_decode'] == 'passed'


@real_media
@pytest.mark.parametrize('when', ['worker', 'after_worker', 'decode'])
def test_cancellation_propagates_without_publishing_completion(media_setup, monkeypatch, when):
    source, _, job = media_setup
    state = {'cancelled': False}
    events = []
    install_worker(
        monkeypatch, mode='cancel' if when == 'worker' else 'valid',
        after_write=(lambda: state.update(cancelled=True)) if when == 'after_worker' else None)

    def report(*event):
        events.append(event)
        if when == 'decode' and event[0:2] == ('upscale', 99):
            state['cancelled'] = True

    with pytest.raises(CancelledError):
        pipeline.run_enhancement(job, source, report, lambda: state['cancelled'])
    assert (job / 'final.mp4').read_bytes() == source.read_bytes()
    assert not (job / 'upscale-complete.json').exists()
    assert not (job / 'quality_report.json').exists()
    assert all(event[0] != 'completed' for event in events)


@real_media
def test_invalid_worker_report_cannot_leave_complete_marker(media_setup, monkeypatch):
    source, _, job = media_setup
    install_worker(monkeypatch, mode='bad_report')
    events = []
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.run_enhancement(
            job, source, lambda *event: events.append(event), lambda: False)
    assert not (job / 'upscale-complete.json').exists()
    assert not (job / 'quality_report.json').exists()
    assert all(event[0] != 'completed' for event in events)
