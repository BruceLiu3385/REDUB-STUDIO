"""Local-only upload API and sequential job runner; no model loads at import time."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import mimetypes
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
from typing import Callable, Literal
import uuid
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parents[2]
MAX_UPLOAD = 512 * 1024 * 1024
ID_RE = re.compile(r"^[a-f0-9]{32}$")
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}
EXTENSIONS = {'source': VIDEO_EXTENSIONS, 'portrait': VIDEO_EXTENSIONS,
              'voice': VIDEO_EXTENSIONS | AUDIO_EXTENSIONS}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | {'.png', '.jpg', '.jpeg'}
TERMINAL = {'completed', 'failed', 'cancelled'}
VOICE_CODE_REF = '5de7a54aa4e5e2baadb0182dde554908b48b85c2'


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def default_data_dir() -> Path:
    return Path(os.environ.get('REDUB_DATA_DIR', str(WORKSPACE / '.redub-data'))).expanduser().resolve()


def executable(variable: str, name: str) -> str | None:
    configured = os.environ.get(variable)
    if configured:
        found = Path(configured).expanduser()
        return str(found.resolve()) if found.is_file() and os.access(found, os.X_OK) else None
    found = shutil.which(name)
    if found:
        return found
    model_env = os.environ.get('REDUB_TBDUB_ENV')
    if model_env:
        candidate = Path(model_env).expanduser() / 'bin' / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return None


def environment_python(variable: str) -> Path | None:
    value = os.environ.get(variable)
    if not value:
        return None
    path = Path(value).expanduser() / 'bin' / 'python'
    # Preserve the virtualenv invocation path; resolving its Python symlink would
    # bypass pyvenv.cfg and inspect packages in the base interpreter instead.
    return path.absolute() if path.is_file() and os.access(path, os.X_OK) else None


def module_present(python: Path | None, module: str, extra_path: Path | None = None) -> bool:
    if python is None:
        return False
    script = 'import importlib.util; print(int(importlib.util.find_spec(' + repr(module) + ') is not None))'
    env = os.environ.copy()
    if extra_path:
        env['PYTHONPATH'] = str(extra_path)
    try:
        result = subprocess.run([str(python), '-c', script], env=env,
                                capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and result.stdout.strip() == '1'
    except (OSError, subprocess.TimeoutExpired):
        return False


def upscale_ready(python: Path | None = None) -> bool:
    """Discover the local enhancement setup without importing torch or loading weights."""
    try:
        python = python or environment_python('REDUB_TBDUB_ENV')
        if python is None:
            return False
        model = Path(os.environ.get('REDUB_UPSCALE_MODEL',
                                   '/root/autodl-tmp/redub-upscale/RealESRGAN_x2plus.pth')).expanduser()
        return bool(model.is_file() and os.access(model, os.R_OK)
                    and all(module_present(python, name) for name in ('torch', 'cv2', 'numpy')))
    except OSError:
        # Optional root-owned model paths may be inaccessible to a regular user.
        return False


def source_revision(folder: Path | None, expected: str | None) -> dict:
    revision = None
    if folder and folder.is_dir():
        try:
            result = subprocess.run(['git', '-C', str(folder), 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                candidate = result.stdout.strip()
                if re.fullmatch(r'[a-f0-9]{40}', candidate):
                    revision = candidate
        except (OSError, subprocess.TimeoutExpired):
            pass
    expected = expected.strip() if expected else None
    return {'revision': revision, 'expected': expected,
            'verified': bool(expected and re.fullmatch(r'[a-f0-9]{40}', expected) and revision == expected)}


def capabilities() -> dict:
    tbdub = Path(os.environ['REDUB_TBDUB_ROOT']).expanduser() if os.environ.get('REDUB_TBDUB_ROOT') else None
    voice = Path(os.environ['REDUB_VOICE_ROOT']).expanduser() if os.environ.get('REDUB_VOICE_ROOT') else None
    tbdub_python = environment_python('REDUB_TBDUB_ENV')
    voice_python = environment_python('REDUB_VOICE_ENV')
    gpu_names = []
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired):
            pass
    model_files = ['src/TBDub/inference.py', 'checkpoints/config.json',
                   'checkpoints/tbdub_student.safetensors', 'checkpoints/Wan2.2_VAE.safetensors',
                   'checkpoints/null_prompt_emb.pt', 'checkpoints/face_landmarker.task',
                   'checkpoints/hubert-large-ll60k/pytorch_model.bin']
    tbdub_ref = os.environ.get('REDUB_TBDUB_CODE_REF')
    if not tbdub_ref and tbdub:
        try:
            tbdub_ref = (tbdub / 'logs/code_revision.txt').read_text().strip()
        except OSError:
            pass
    tbdub_source = source_revision(tbdub / 'src/TBDub' if tbdub else None, tbdub_ref)
    voice_source = source_revision(voice / 'src/chatterbox' if voice else None, VOICE_CODE_REF)
    tbdub_configured = bool(tbdub and tbdub_python and tbdub_source['verified']
                            and all((tbdub / file).is_file() for file in model_files))
    voice_ready = bool(voice and voice_python and voice_source['verified']
                       and (voice / 'src/chatterbox/src/chatterbox/vc.py').is_file()
                       and (voice / 'checkpoints/s3gen.safetensors').is_file()
                       and module_present(voice_python, 'chatterbox', voice / 'src/chatterbox/src'))
    separator_python = environment_python('REDUB_SEPARATOR_ENV')
    if not os.environ.get('REDUB_SEPARATOR_ENV'):
        separator_python = Path(sys.executable)
    return {'status': 'ok', 'gpu_ready': bool(tbdub_configured and gpu_names),
            'voice_ready': voice_ready, 'background_ready': module_present(separator_python, 'demucs'),
            'upscale_ready': upscale_ready(tbdub_python),
            'ffmpeg': bool(executable('REDUB_FFMPEG', 'ffmpeg')),
            'ffprobe': bool(executable('REDUB_FFPROBE', 'ffprobe')),
            'gpu_names': gpu_names, 'tbdub_configured': tbdub_configured,
            'source_revisions': {'tbdub': tbdub_source, 'voice': voice_source},
            'readiness_note': '仅检查本机工具、配置文件、固定源码版本和可见 GPU；未加载或验证模型推理。',
            'limits': {'max_duration': 600, 'portrait_max_duration': 660,
                       'voice_min_duration': 5, 'voice_max_duration': 60,
                       'chunk_seconds': 15, 'max_upload_bytes': MAX_UPLOAD}}


def positive_duration(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def stream_duration(stream: dict | None) -> float | None:
    """Same precedence as media.probe; frame counts use frame rate, not time_base."""
    if not stream:
        return None
    seconds = positive_duration(stream.get('duration'))
    if seconds is not None:
        return seconds
    try:
        seconds = positive_duration(int(stream['duration_ts']) * Fraction(stream['time_base']))
        if seconds is not None:
            return seconds
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    tags = stream.get('tags', {})
    tagged = tags.get('DURATION') or tags.get('duration')
    match = re.fullmatch(r'(\d+):(\d+):(\d+(?:\.\d+)?)', str(tagged or ''))
    if match:
        seconds = positive_duration(int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3]))
        if seconds is not None:
            return seconds
    if stream.get('codec_type') == 'video':
        for rate in ('avg_frame_rate', 'r_frame_rate'):
            try:
                seconds = positive_duration(int(stream['nb_frames']) / Fraction(stream[rate]))
                if seconds is not None:
                    return seconds
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass
    return None


def probe_media(path: Path, kind: str) -> dict:
    ffprobe = executable('REDUB_FFPROBE', 'ffprobe')
    if not ffprobe:
        raise HTTPException(503, '未找到 ffprobe；请安装并配置 REDUB_FFPROBE 后上传素材。')
    try:
        result = subprocess.run([ffprobe, '-v', 'error', '-protocol_whitelist', 'file,pipe',
                                 '-show_format', '-show_streams', '-of', 'json', str(path)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise ValueError('无法解码此媒体文件。')
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        video = next((item for item in streams if item.get('codec_type') == 'video'
                      and not item.get('disposition', {}).get('attached_pic')), None)
        audio = next((item for item in streams if item.get('codec_type') == 'audio'), None)
        has_audio = audio is not None
        durations = [stream_duration(item) for item in streams]
        duration = positive_duration(data.get('format', {}).get('duration')) or max(
            (item for item in durations if item), default=0)
        if duration <= 0:
            raise ValueError('媒体缺少有效时长。')
        audio_duration = (stream_duration(audio) or duration) if audio else 0
        video_duration = (stream_duration(video) or duration) if video else 0
        if audio and path.suffix.lower() == '.wav':
            try:
                with wave.open(str(path), 'rb') as pcm:
                    audio_duration = pcm.getnframes() / pcm.getframerate()
                if video is None:
                    duration = audio_duration
            except (wave.Error, EOFError, OSError):
                pass
        # Reject playlists and other containers that may refer to external files.
        allowed_formats = {'mov', 'mp4', 'm4a', '3gp', '3g2', 'mj2', 'matroska',
                           'webm', 'wav', 'mp3', 'aac', 'flac', 'ogg'}
        formats = set(data.get('format', {}).get('format_name', '').split(','))
        if not formats or not formats.issubset(allowed_formats):
            raise ValueError('不支持此媒体容器，请转为 MP4、WAV 或 MP3。')
        if kind in ('source', 'portrait') and video is None:
            raise ValueError('源视频和人物素材必须包含真实视频画面，不能上传静态图片。')
        if kind in ('source', 'voice') and not has_audio:
            raise ValueError('源视频和参考声音必须包含音轨。')
        if kind == 'voice' and not 5 <= audio_duration <= 60:
            raise ValueError('参考声音需为 5–60 秒，建议使用 10–30 秒清晰人声。')
        if kind == 'source' and audio_duration > 600:
            raise ValueError('源视频的口播音轨不能超过 600 秒。')
        if kind == 'portrait' and video_duration > 660:
            raise ValueError('人物视频不能超过 660 秒。')
        width, height = (int(video.get('width', 0)), int(video.get('height', 0))) if video else (0, 0)
        if video and (width <= 0 or height <= 0):
            raise ValueError('视频尺寸无效。')
        return {'duration': round(duration, 3), 'audio_duration': round(audio_duration, 6),
                'video_duration': round(video_duration, 6),
                'width': width, 'height': height, 'has_audio': has_audio}
    except subprocess.TimeoutExpired:
        raise HTTPException(422, '媒体探测超时，请使用普通 MP4、WAV 或 MP3 文件。')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc))


class JobOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    voice_conversion: StrictBool = True
    keep_background: StrictBool = False
    chunk_seconds: Literal[5, 10, 15] = 15
    fit_video: StrictBool = True
    enhance_video: StrictBool = False


class JobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    portrait_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    voice_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    options: JobOptions = Field(default_factory=JobOptions)


class EnhancementRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class JobManager:
    def __init__(self, root: Path, runner: Callable | None = None):
        self.root = root.resolve()
        self.assets_dir = self.root / 'assets'
        self.jobs_dir = self.root / 'jobs'
        self.jobs: dict[str, dict] = {}
        self.cancellations: dict[str, threading.Event] = {}
        self.mutex = threading.RLock()
        self.pending: queue.Queue[str | None] = queue.Queue()
        self.stopping = threading.Event()
        self.runner = runner
        self.thread: threading.Thread | None = None

    def start(self):
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        for folder in self.jobs_dir.iterdir():
            if not ID_RE.fullmatch(folder.name) or folder.is_symlink():
                continue
            try:
                job = json.loads((folder / 'job.json').read_text())
                if job['id'] != folder.name:
                    continue
                self.jobs[job['id']] = job
                self.cancellations[job['id']] = threading.Event()
                if job['status'] not in TERMINAL:
                    self.update(job['id'], status='failed', stage='interrupted',
                                message='服务已重启，可点击重试继续。', error='任务因服务重启中断。')
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.thread = threading.Thread(target=self.work, name='redub-worker', daemon=False)
        self.thread.start()

    def close(self):
        self.stopping.set()
        with self.mutex:
            for job_id, job in self.jobs.items():
                if job['status'] not in TERMINAL:
                    self.cancellations[job_id].set()
                    if job['status'] == 'queued':
                        self.update(job_id, status='cancelled', stage='cancelled', message='服务关闭，任务已取消。')
        self.pending.put(None)
        if self.thread:
            # Media cancellation permits up to 8s TERM cleanup; probes may take
            # up to 30s. Do not exit ahead of our non-daemon worker and orphan a GPU.
            self.thread.join(timeout=40)
            if self.thread.is_alive():
                raise RuntimeError('任务尚未完成退出清理；服务不会将仍在运行的子进程视为已释放。')

    def public(self, job: dict) -> dict:
        result = json.loads(json.dumps(job))
        folder = (self.jobs_dir / job['id']).resolve()
        # Audio becomes available before the slower lip-sync stage completes.
        # This marker can authorize only these two known, completed WAV files.
        try:
            marker = json.loads((folder / 'audio_ready.json').read_text())
            for key, filename in {'source_audio': 'source.wav', 'audio': 'converted.wav'}.items():
                path = (folder / filename).resolve()
                if marker.get(key) == filename and path.is_relative_to(folder) and path.is_file():
                    result['outputs'][key] = filename
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        result['download_urls'] = {key: f"/api/jobs/{job['id']}/download/{key}" for key in result['outputs']}
        return result

    def get(self, job_id: str) -> dict:
        with self.mutex:
            if job_id not in self.jobs:
                raise HTTPException(404, '任务不存在。')
            return self.public(self.jobs[job_id])

    def asset(self, asset_id: str, kind: str) -> tuple[dict, Path]:
        if not ID_RE.fullmatch(asset_id):
            raise HTTPException(404, '素材不存在。')
        folder = self.assets_dir / asset_id
        try:
            metadata = json.loads((folder / 'asset.json').read_text())
            path = (folder / metadata['file']).resolve()
            if metadata['id'] != asset_id or metadata['kind'] != kind:
                raise HTTPException(422, f'素材类型不匹配，期望 {kind}。')
            if not path.is_relative_to(self.assets_dir) or not path.is_file():
                raise HTTPException(404, '素材文件不存在。')
            return metadata, path
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(404, '素材不存在或元数据无效。')

    def update(self, job_id: str, **changes):
        with self.mutex:
            job = self.jobs[job_id]
            job.update(changes, updated_at=now())
            if 'message' in changes:
                job['events'].append({'time': now(), 'stage': job['stage'], 'message': job['message']})
                job['events'] = job['events'][-300:]
            write_json(self.jobs_dir / job_id / 'job.json', job)

    def submit(self, body: JobRequest) -> dict:
        source, _ = self.asset(body.source_id, 'source')
        self.asset(body.portrait_id, 'portrait')
        if body.options.voice_conversion and not body.voice_id:
            raise HTTPException(422, '开启音色转换时，必须上传参考声音。')
        if body.voice_id:
            self.asset(body.voice_id, 'voice')
        if body.options.enhance_video and not upscale_ready():
            raise HTTPException(503, 'AI 高清增强模型尚未就绪，请关闭此项或先完成模型配置。')
        job_id = uuid.uuid4().hex
        with self.mutex:
            folder = self.jobs_dir / job_id
            folder.mkdir()
            timestamp = now()
            self.jobs[job_id] = {'id': job_id, 'status': 'queued', 'stage': 'queued',
                'progress': 0, 'message': '等待处理。', 'error': None,
                'created_at': timestamp, 'updated_at': timestamp, 'outputs': {}, 'events': [],
                'source_name': source['name'],
                'assets': {'source': body.source_id, 'voice': body.voice_id, 'portrait': body.portrait_id},
                'options': body.options.model_dump()}
            self.cancellations[job_id] = threading.Event()
            self.update(job_id, message='等待处理。')
            self.pending.put(job_id)
            return self.public(self.jobs[job_id])

    def enhancement_source(self, job_id: str) -> Path:
        """Only a completed job's original, readable video can be enhanced."""
        with self.mutex:
            if not isinstance(job_id, str) or not ID_RE.fullmatch(job_id) or job_id not in self.jobs:
                raise HTTPException(404, '来源任务不存在。')
            job = self.jobs[job_id]
            if job['status'] != 'completed':
                raise HTTPException(409, '只能增强已完成任务的原版成片。')
            relative = job.get('outputs', {}).get('video')
            if not isinstance(relative, str) or not relative:
                raise HTTPException(422, '来源任务没有可增强的原版视频。')
            path = Path(relative)
            folder = self.jobs_dir / job_id
            resolved_folder = folder.resolve()
            resolved = (folder / path).resolve()
            if (folder.is_symlink() or resolved_folder.parent != self.jobs_dir.resolve()
                    or path.is_absolute() or '..' in path.parts
                    or path.suffix.lower() not in VIDEO_EXTENSIONS
                    or not resolved.is_relative_to(resolved_folder)
                    or not resolved.is_file() or not os.access(resolved, os.R_OK)):
                raise HTTPException(422, '来源任务的原版视频不存在或不可读取。')
            return resolved

    def enhance(self, source_id: str) -> dict:
        with self.mutex:
            self.enhancement_source(source_id)
            if not upscale_ready():
                raise HTTPException(503, 'AI 高清增强模型尚未就绪，请先完成模型配置。')
            source = self.jobs[source_id]
            job_id = uuid.uuid4().hex
            folder = self.jobs_dir / job_id
            folder.mkdir()
            timestamp = now()
            options = JobOptions.model_validate(source.get('options', {})).model_dump()
            options['enhance_video'] = True
            self.jobs[job_id] = {'id': job_id, 'status': 'queued', 'stage': 'queued',
                'progress': 0, 'message': '等待高清增强。', 'error': None,
                'created_at': timestamp, 'updated_at': timestamp, 'outputs': {}, 'events': [],
                'source_name': '高清增强 · ' + source.get('source_name', '口播成片'),
                'enhance_from': source_id,
                'assets': dict(source.get('assets', {})), 'options': options}
            self.cancellations[job_id] = threading.Event()
            self.update(job_id, message='等待高清增强，将直接处理现有成片。')
            self.pending.put(job_id)
            return self.public(self.jobs[job_id])

    def cancel(self, job_id: str) -> dict:
        with self.mutex:
            self.get(job_id)
            job = self.jobs[job_id]
            if job['status'] not in TERMINAL:
                self.cancellations[job_id].set()
                if job['status'] == 'queued':
                    self.update(job_id, status='cancelled', stage='cancelled', message='排队任务已取消。')
                else:
                    self.update(job_id, message='正在取消，请等待当前处理退出。')
            return self.public(job)

    def retry(self, job_id: str) -> dict:
        with self.mutex:
            self.get(job_id)
            job = self.jobs[job_id]
            if job['status'] not in {'failed', 'cancelled'}:
                raise HTTPException(409, '只有失败或已取消的任务可以重试。')
            if job.get('enhance_from'):
                self.enhancement_source(job['enhance_from'])
            else:
                for kind, asset_id in job['assets'].items():
                    if asset_id:
                        self.asset(asset_id, kind)
            if job.get('options', {}).get('enhance_video') and not upscale_ready():
                raise HTTPException(503, 'AI 高清增强模型尚未就绪，请先完成模型配置。')
            self.cancellations[job_id] = threading.Event()
            self.update(job_id, status='queued', stage='queued', progress=0,
                        outputs={}, error=None, message='已重新排队，将复用可用的中间结果。')
            self.pending.put(job_id)
            return self.public(job)

    def work(self):
        while not self.stopping.is_set():
            job_id = self.pending.get()
            try:
                if job_id is None:
                    return
                with self.mutex:
                    job = self.jobs[job_id]
                    if job['status'] != 'queued':
                        continue
                    cancelled = self.cancellations[job_id]
                    self.update(job_id, status='running', stage='starting', message='开始处理。')
                    asset_ids = dict(job['assets'])
                    options = dict(job['options'])
                    enhance_from = job.get('enhance_from')

                def report(stage, progress, message):
                    value = float(progress)
                    if not math.isfinite(value):
                        value = 0
                    self.update(job_id, stage=str(stage), progress=max(0, min(100, value)), message=str(message))

                folder = self.jobs_dir / job_id
                if enhance_from:
                    from .pipeline import run_enhancement
                    source_final = self.enhancement_source(enhance_from)
                    outputs = run_enhancement(folder, source_final, report, cancelled.is_set)
                else:
                    paths = {kind: self.asset(asset_id, kind)[1] for kind, asset_id in asset_ids.items() if asset_id}
                    runner = self.runner
                    if runner is None:
                        from .pipeline import run_pipeline
                        runner = run_pipeline
                    outputs = runner(folder, paths, options, report, cancelled.is_set)
                if cancelled.is_set():
                    self.update(job_id, status='cancelled', stage='cancelled', outputs={}, message='任务已取消。')
                    continue
                checked = {}
                if not isinstance(outputs, dict):
                    raise ValueError('处理流程没有返回有效输出。')
                for key, relative in outputs.items():
                    if not re.fullmatch(r'[a-zA-Z0-9_-]+', key) or not isinstance(relative, str):
                        raise ValueError('处理流程返回了无效文件名。')
                    path = Path(relative)
                    resolved = (folder / path).resolve()
                    allowed_file = resolved.suffix.lower() in MEDIA_EXTENSIONS or (
                        key == 'report' and relative == 'quality_report.json')
                    if (path.is_absolute() or '..' in path.parts or not resolved.is_relative_to(folder.resolve())
                            or not resolved.is_file() or not allowed_file):
                        raise ValueError('处理流程输出文件不存在或超出允许目录。')
                    checked[key] = path.as_posix()
                if not checked:
                    raise ValueError('处理流程没有生成可下载文件。')
                self.update(job_id, status='completed', stage='completed', progress=100,
                            outputs=checked, error=None, message='处理完成。')
            except Exception as exc:
                if job_id is not None:
                    cancelled = self.cancellations[job_id].is_set()
                    self.update(job_id, status='cancelled' if cancelled else 'failed',
                                stage='cancelled' if cancelled else 'failed', outputs={},
                                error=None if cancelled else str(exc),
                                message='任务已取消。' if cancelled else '处理失败，可查看错误并重试。')
            finally:
                self.pending.task_done()


class UploadSizeLimit:
    """Bound multipart request bytes before Starlette can spool unlimited data."""
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/api/assets' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        headers = dict(scope['headers'])
        try:
            too_large = int(headers.get(b'content-length', b'0')) > self.limit
        except ValueError:
            too_large = True
        if too_large:
            return await JSONResponse({'detail': '上传文件超过大小限制。'}, status_code=413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > self.limit:
                raise HTTPException(413, '上传文件超过大小限制。')
            return message

        await self.app(scope, limited_receive, send)


def create_app(data_dir: Path | None = None, runner: Callable | None = None) -> FastAPI:
    manager = JobManager(data_dir or default_data_dir(), runner)

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        yield
        manager.close()

    app = FastAPI(title='Redub Studio', lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(UploadSizeLimit, limit=MAX_UPLOAD + 2 * 1024 * 1024)

    @app.get('/api/health')
    def health():
        return capabilities()

    @app.post('/api/assets', status_code=201)
    async def upload_asset(file: UploadFile = File(...), kind: str = Form(...)):
        if kind not in EXTENSIONS:
            raise HTTPException(422, '素材类型必须是 source、voice 或 portrait。')
        name = (file.filename or '').replace('\\', '/').split('/')[-1][:200]
        suffix = Path(name).suffix.lower()
        if suffix not in EXTENSIONS[kind]:
            raise HTTPException(422, '文件扩展名不受支持；人物素材必须上传视频。')
        asset_id = uuid.uuid4().hex
        folder = manager.assets_dir / asset_id
        folder.mkdir()
        path = folder / ('media' + suffix)
        size = 0
        try:
            with path.open('xb') as target:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, '单个文件不能超过 512 MiB。')
                    target.write(block)
            if not size:
                raise HTTPException(422, '文件不能为空。')
            media = await asyncio.to_thread(probe_media, path, kind)
            metadata = {'id': asset_id, 'kind': kind, 'name': name, 'bytes': size, **media}
            write_json(folder / 'asset.json', {**metadata, 'file': path.name})
            return metadata
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await file.close()

    @app.post('/api/jobs', status_code=201)
    def create_job(body: JobRequest):
        return manager.submit(body)

    @app.get('/api/jobs')
    def list_jobs():
        with manager.mutex:
            return [manager.public(job) for job in sorted(manager.jobs.values(),
                    key=lambda item: item['created_at'], reverse=True)[:100]]

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: str):
        return manager.get(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        return manager.cancel(job_id)

    @app.post('/api/jobs/{job_id}/retry')
    def retry_job(job_id: str):
        return manager.retry(job_id)

    @app.post('/api/jobs/{job_id}/enhance', status_code=201)
    def enhance_job(job_id: str, body: EnhancementRequest | None = None):
        return manager.enhance(job_id)

    @app.get('/api/jobs/{job_id}/download/{name}')
    def download(job_id: str, name: str):
        job = manager.get(job_id)
        outputs = job['outputs']
        relative = outputs.get(name)
        if relative is None and name in outputs.values():
            relative = name
        if relative is None:
            raise HTTPException(404, '输出文件不存在。')
        folder = (manager.jobs_dir / job_id).resolve()
        path = (folder / relative).resolve()
        allowed_file = path.suffix.lower() in MEDIA_EXTENSIONS or (
            relative == 'quality_report.json' and outputs.get('report') == relative)
        if not path.is_relative_to(folder) or not path.is_file() or not allowed_file:
            raise HTTPException(404, '输出文件不存在。')
        return FileResponse(path, filename=path.name, media_type=mimetypes.guess_type(path.name)[0])

    @app.get('/')
    def index():
        path = PACKAGE / 'static' / 'index.html'
        if not path.is_file():
            return JSONResponse({'detail': '界面文件尚未安装，API 已可用。'}, status_code=503)
        return FileResponse(path)

    if (PACKAGE / 'static').is_dir():
        app.mount('/static', StaticFiles(directory=PACKAGE / 'static', follow_symlink=False), name='static')
    return app


app = create_app()


def main():
    import uvicorn
    # Deliberately bind only to loopback. Use an authenticated SSH tunnel remotely.
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('REDUB_PORT', '7860')))


if __name__ == '__main__':
    main()
