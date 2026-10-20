"""Actual FFmpeg tests; AI is deliberately replaced and is NOT validated here."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pytest

from redub_studio.media import executable, plan_segments, probe
from redub_studio import pipeline


def available():
    try:
        executable('ffmpeg')
        executable('ffprobe')
        return True
    except RuntimeError:
        return False


def ff(*args):
    subprocess.run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y', *map(str, args)], check=True)


def count_frames(path):
    result = subprocess.run([executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    return int(json.loads(result.stdout)['streams'][0]['nb_read_frames'])


@pytest.mark.parametrize('frames', [1, 13, 125, 452, 14999, 15000])
def test_segment_time_grid_and_complete_coverage(frames):
    segments = plan_segments(frames, 15, [14.8, 30.1, 44.9])
    assert segments[0].start == 0 and segments[-1].end == frames
    assert sum(s.end-s.start for s in segments) == frames
    assert all(a.end == b.start for a, b in zip(segments, segments[1:]))
    assert all(0 <= s.context_start <= s.start < s.end <= s.context_end <= frames for s in segments)


@pytest.mark.skipif(not available(), reason='requires real FFmpeg and ffprobe')
def test_real_media_multisegment_timing_and_retry(tmp_path, monkeypatch):
    source, portrait = tmp_path/'source.mkv', tmp_path/'portrait.mp4'
    # 18.08 seconds = 452 frames: exercises the upstream floating-point floor bug.
    ff('-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=25:duration=18.08', '-f', 'lavfi', '-i',
       'sine=frequency=300:sample_rate=48000:duration=18.08', '-c:v', 'libx264', '-c:a', 'pcm_s16le', '-shortest', source)
    ff('-i', source, '-an', '-c:v', 'copy', portrait)
    real_run = pipeline.run
    model_calls = []
    def ai_double(command, **kwargs):
        if 'tbdub_batch.py' in str(command[1]):
            manifest = json.loads(Path(command[-1]).read_text())
            model_calls.append(len(manifest['segments']))
            for item in manifest['segments']:
                folder = Path(item['output_dir'])
                folder.mkdir(exist_ok=True)
                shutil.copyfile(item['video'], folder/f"{item['name']}_tbdub.mp4")
                (folder/'complete.json').write_text('{}')
            return ''
        return real_run(command, **kwargs)
    monkeypatch.setattr(pipeline, 'run', ai_double)
    monkeypatch.setattr(pipeline, 'model_environment', lambda name: (tmp_path, Path(sys.executable), os.environ.copy()))
    folder = tmp_path/'job'
    options = {'voice_conversion': False, 'keep_background': False, 'chunk_seconds': 5, 'fit_video': True}
    events = []
    outputs = pipeline.run_pipeline(folder, {'source': source, 'portrait': portrait}, options,
        lambda *event: events.append(event), lambda: False)
    assert model_calls[0] >= 3
    assert count_frames(folder/outputs['video']) == 452
    assert abs(probe(folder/outputs['video'])['duration']-18.08) < .05
    with wave.open(str(folder/outputs['audio'])) as audio:
        samples = audio.getnframes()
        assert abs(samples/audio.getframerate()-18.08) < .05
    report = json.loads((folder/'quality_report.json').read_text())
    assert report['full_decode'] == 'passed'
    assert report['frames'] == 452
    # Exercise cached audio/normalized clips and re-stitching on the same job.
    pipeline.run_pipeline(folder, {'source': source, 'portrait': portrait}, options, lambda *_: None, lambda: False)
    assert count_frames(folder/'final.mp4') == 452


@pytest.mark.skipif(not available(), reason='requires real FFmpeg and ffprobe')
def test_reject_short_portrait_without_looping(tmp_path):
    source, portrait = tmp_path/'source.mp4', tmp_path/'portrait.mp4'
    ff('-f', 'lavfi', '-i', 'color=size=64x64:rate=25:duration=2', '-f', 'lavfi', '-i',
       'sine=duration=2', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', source)
    ff('-f', 'lavfi', '-i', 'color=size=64x64:rate=25:duration=1', '-c:v', 'libx264', portrait)
    with pytest.raises(ValueError, match='不会循环'):
        pipeline.run_pipeline(tmp_path/'job', {'source': source, 'portrait': portrait},
            {'voice_conversion': False, 'fit_video': True}, lambda *_: None, lambda: False)
