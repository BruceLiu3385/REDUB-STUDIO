"""Media utilities. Commands are argument lists, never shell strings."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import wave
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, asdict
from fractions import Fraction


class CancelledError(RuntimeError):
    pass


def executable(name: str) -> str:
    configured = os.environ.get(f"REDUB_{name.upper()}")
    candidates = [configured, shutil.which(name)]
    env = os.environ.get("REDUB_TBDUB_ENV")
    if env:
        candidates.append(str(Path(env) / "bin" / name))
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(f"缺少 {name}，请安装或设置 REDUB_{name.upper()}")


def run(command, *, log: Path, cancelled=lambda: False, cwd=None, env=None, timeout=10800, heartbeat=None, output=None):
    """Cancellation kills the whole process group, including FFmpeg/model children."""
    log.parent.mkdir(parents=True, exist_ok=True)
    if cancelled():
        raise CancelledError("任务已取消")
    with ExitStack() as stack:
        stream = stack.enter_context(log.open("ab"))
        stdout = stack.enter_context(Path(output).open("wb")) if output else stream
        process = subprocess.Popen(
            [str(x) for x in command], stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stream if output else subprocess.STDOUT, cwd=cwd, env=env,
            start_new_session=True,
        )
        start = time.monotonic()
        try:
            while process.poll() is None:
                if heartbeat:
                    heartbeat()
                if cancelled():
                    raise CancelledError("任务已取消")
                if time.monotonic() - start > timeout:
                    raise RuntimeError("处理超时；已保留完成片段，可稍后继续处理")
                time.sleep(0.25)
        finally:
            # Also reap the child if a progress callback or unexpected exception
            # fails. A failed web job must not leave a GPU subprocess running.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=8)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if process.returncode:
            with log.open("rb") as reader:
                reader.seek(max(0, log.stat().st_size - 5000))
                tail = reader.read().decode("utf-8", "replace")
            raise RuntimeError(f"处理失败（退出码 {process.returncode}），详情：\n{tail}")
    with log.open("rb") as reader:
        reader.seek(max(0, log.stat().st_size - 20000))
        return reader.read().decode("utf-8", "replace")


def choose_seams(items: list[dict], cancelled=lambda: False) -> list[dict]:
    """Choose the closest matching frame within shared context, without mouth cross-fades.

    Both clips represent the same original global frames. Low-res luma difference
    ranks candidate cut points; this is a heuristic, not a naturalness guarantee.
    """
    result = [dict(item) for item in items]
    for previous, current in zip(result, result[1:]):
        if cancelled():
            raise CancelledError("任务已取消")
        nominal = current["start"]
        first = max(current["context_start"], nominal - 7, previous["start"] + 1)
        last = min(previous["context_end"], nominal + 8, current["end"] - 1)
        if last <= first:
            continue
        def frames_for(item):
            path = Path(item["output_dir"]) / f"{item['name']}_tbdub.mp4"
            local = first - item["context_start"]
            filter_graph = f"trim=start_frame={local}:end_frame={local+last-first},setpts=PTS-STARTPTS,scale=64:64,format=gray"
            with tempfile.TemporaryDirectory(prefix="redub-seam-") as workspace:
                folder = Path(workspace)
                raw = folder / "frames.raw"
                run([executable("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path),
                    "-vf", filter_graph, "-an", "-f", "rawvideo", "-"],
                    log=folder / "frames.log", output=raw, timeout=60, cancelled=cancelled)
                return raw.read_bytes()
        left, right = frames_for(previous), frames_for(current)
        count = min(len(left), len(right)) // 4096
        if not count:
            continue
        # A slight preference for the pause-aligned nominal point avoids wandering
        # far from pauses merely because one candidate has a static background.
        scores = []
        for index in range(count):
            a, b = left[index*4096:(index+1)*4096], right[index*4096:(index+1)*4096]
            difference = sum(abs(x-y) for x, y in zip(a, b)) / 4096
            scores.append(difference + .015 * abs(first + index - nominal))
        seam = first + min(range(count), key=lambda index: scores[index])
        previous["end"], current["start"] = seam, seam
    return result


def probe(path: Path) -> dict:
    result = subprocess.run(
        [executable("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    data = json.loads(result.stdout)
    video = next((x for x in data.get("streams", []) if x.get("codec_type") == "video" and not x.get("disposition", {}).get("attached_pic")), None)
    audio = next((x for x in data.get("streams", []) if x.get("codec_type") == "audio"), None)
    def positive(value):
        try:
            number = float(value)
            return number if math.isfinite(number) and number > 0 else None
        except (TypeError, ValueError):
            return None

    def stream_duration(stream):
        if not stream:
            return None
        seconds = positive(stream.get("duration"))
        if seconds is not None:
            return seconds
        try:
            seconds = positive(int(stream["duration_ts"]) * Fraction(stream["time_base"]))
            if seconds is not None:
                return seconds
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        # Matroska/WebM commonly store individual stream lengths in tags rather
        # than stream.duration. The container may include a longer other track.
        tags = stream.get("tags", {})
        tagged = tags.get("DURATION") or tags.get("duration")
        match = re.fullmatch(r"(\d+):(\d+):(\d+(?:\.\d+)?)", str(tagged or ""))
        if match:
            return positive(int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3]))
        return None

    durations = [stream_duration(x) for x in data.get("streams", [])]
    duration = positive(data.get("format", {}).get("duration")) or max((x for x in durations if x), default=0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("媒体没有有效时长")
    audio_duration = stream_duration(audio) or duration if audio else 0
    audio_samples, audio_sample_rate = None, int(audio.get("sample_rate") or 0) if audio else 0
    # PCM WAV extraction is our authoritative speech timeline. Counting samples
    # avoids relying on rounded container duration or another stream's length.
    if audio and Path(path).suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as pcm:
                audio_samples, audio_sample_rate = pcm.getnframes(), pcm.getframerate()
            audio_duration = audio_samples / audio_sample_rate
            if not video:
                duration = audio_duration
        except (wave.Error, EOFError):
            # Float WAV is not supported by all Python wave versions; ffprobe
            # remains available for intermediate model output in that format.
            pass
    return {"duration": duration, "width": int(video.get("width", 0)) if video else 0,
            "height": int(video.get("height", 0)) if video else 0,
            "has_audio": audio is not None, "has_video": video is not None,
            "audio_duration": audio_duration,
            "audio_samples": audio_samples, "audio_sample_rate": audio_sample_rate,
            "video_duration": (stream_duration(video) or duration) if video else 0,
            "streams": data.get("streams", [])}


def count_video_frames(path: Path, cancelled=lambda: False) -> int:
    """Decode and count the actual first video stream, not container duration."""
    process = subprocess.Popen(
        [executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    started = time.monotonic()
    try:
        while True:
            if cancelled():
                raise CancelledError("任务已取消")
            if time.monotonic() - started > 180:
                raise RuntimeError("视频计帧超时")
            try:
                stdout, stderr = process.communicate(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise RuntimeError(f"视频计帧失败：{stderr[-1000:]}")
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
    streams = json.loads(stdout).get("streams", [])
    if not streams or not str(streams[0].get("nb_read_frames", "")).isdigit():
        raise ValueError("无法确认实际视频帧数")
    count = int(streams[0]["nb_read_frames"])
    if count < 1:
        raise ValueError("视频没有可解码画面")
    return count


def ffmpeg(args, log: Path, cancelled=lambda: False):
    return run([executable("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "warning", "-y", *args], log=log, cancelled=cancelled)


@dataclass(frozen=True)
class Segment:
    index: int
    start: int
    end: int
    context_start: int
    context_end: int

    def dictionary(self):
        return asdict(self)


def plan_segments(total_frames: int, chunk_seconds=15, silences=(), fps=25, overlap_frames=12):
    """Cut on the global frame grid, near pauses; context never duplicates output frames."""
    if total_frames < 1 or not 5 <= chunk_seconds <= 30:
        raise ValueError("片长应为5–30秒，且视频须至少一帧")
    target = int(chunk_seconds * fps)
    silence_frames = [round(t * fps) for t in silences]
    boundaries = [0]
    while total_frames - boundaries[-1] > target + 2 * fps:
        desired = boundaries[-1] + target
        candidates = [f for f in silence_frames if abs(f - desired) <= 2 * fps and f - boundaries[-1] >= 5 * fps]
        cut = min(candidates, key=lambda f: abs(f - desired)) if candidates else desired
        boundaries.append(cut)
    boundaries.append(total_frames)
    return [Segment(i, a, b, max(0, a - overlap_frames), min(total_frames, b + overlap_frames))
            for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))]


def silence_centres(path, log, cancelled=lambda: False):
    text = run([executable("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(path), "-af",
                "silencedetect=noise=-34dB:d=0.18", "-f", "null", "-"], log=log, cancelled=cancelled)
    starts = re.findall(r"silence_start: ([\d.]+)", text)
    ends = re.findall(r"silence_end: ([\d.]+)", text)
    return [(float(a) + float(b)) / 2 for a, b in zip(starts, ends) if float(b) >= float(a)]


def save_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)
