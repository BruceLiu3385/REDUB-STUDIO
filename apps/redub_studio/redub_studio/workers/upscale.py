"""Stream the application's 25 fps output through conservative Real-ESRGAN x2.

Run in the existing CUDA/PyTorch environment. Only an explicitly supplied local
RealESRGAN_x2plus checkpoint is loaded; no model or package is downloaded.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


FPS = 25
STRENGTH = 0.7
TILE_PAD = 32
TIMESTAMP_TOLERANCE = Fraction(1, 10000)


def atomic_json(path: Path | None, data: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def tail_log(handle) -> str:
    handle.flush()
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - 8000))
    return handle.read().decode("utf-8", errors="replace").strip()


def stop_process(process) -> None:
    if process is None:
        return
    for pipe in (process.stdin, process.stdout):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def probe(ffprobe: str, path: Path) -> dict:
    result = subprocess.run([
        ffprobe, "-v", "error", "-count_packets", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def video_stream(info: dict) -> dict:
    videos = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("Expected exactly one video stream")
    video = videos[0]
    if Fraction(video.get("r_frame_rate", "0/1")) != FPS or Fraction(video.get("avg_frame_rate", "0/1")) != FPS:
        raise ValueError("Only the application's 25 fps CFR video is supported")
    if video.get("sample_aspect_ratio", "1:1") not in ("1:1", "N/A", "0:1"):
        raise ValueError("Input must have square pixels")
    rotation = video.get("tags", {}).get("rotate", 0)
    rotations = [rotation] + [item.get("rotation", 0) for item in video.get("side_data_list", [])]
    if any(float(value) % 360 for value in rotations):
        raise ValueError("Normalize the video's rotation before enhancement")
    if int(video.get("width", 0)) < 2 or int(video.get("height", 0)) < 2:
        raise ValueError("Input frame dimensions must be at least 2 pixels")
    return video


def count_cfr_frames(ffprobe: str, path: Path) -> int:
    """Decode timestamps in display order, checking every interval without a list."""
    process = None
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen([
                ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "default=noprint_wrappers=1", str(path),
            ], stdout=subprocess.PIPE, stderr=errors, text=True)
            frames = 0
            for line in process.stdout:
                # Some FFprobe versions append H.264 SEI text directly to CSV
                # timestamps. Keyed output keeps frame timing distinct from SEI.
                if not line.startswith('best_effort_timestamp_time='):
                    continue
                timestamp = line.strip().partition('=')[2]
                try:
                    pts = Fraction(timestamp)
                except (ValueError, ZeroDivisionError) as exc:
                    raise ValueError(f"Missing or invalid frame timestamp: {timestamp!r}") from exc
                if abs(pts - Fraction(frames, FPS)) > TIMESTAMP_TOLERANCE:
                    raise ValueError(f"Video is not zero-based 25 fps CFR at frame {frames}: {timestamp}")
                frames += 1
            code = process.wait(timeout=30)
            if code or tail_log(errors):
                raise RuntimeError(f"ffprobe frame validation failed ({code}): {tail_log(errors)}")
            if frames == 0:
                raise ValueError("Input has no decoded video frames")
            return frames
        finally:
            stop_process(process)


def audio_signature(ffmpeg: str, path: Path, info: dict) -> list[dict]:
    """Compare copied compressed packets, including count and codec parameters."""
    streams = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        return []
    result = subprocess.run([
        ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-map", "0:a", "-c:a", "copy", "-f", "streamhash", "-hash", "sha256", "-",
    ], capture_output=True, text=True, check=True)
    hashes = [line.strip().rsplit("=", 1)[-1] for line in result.stdout.splitlines()
              if line.strip() and not line.startswith("#")]
    if len(hashes) != len(streams):
        raise RuntimeError("Could not hash all compressed audio streams")
    signatures = []
    for stream, payload_hash in zip(streams, hashes):
        if len(payload_hash) != 64:
            raise RuntimeError("Invalid compressed audio SHA256 response")
        signatures.append({
            "codec": stream.get("codec_name"), "sample_rate": stream.get("sample_rate"),
            "channels": stream.get("channels"), "channel_layout": stream.get("channel_layout"),
            "packets": int(stream["nb_read_packets"]), "sha256": payload_hash,
        })
    return signatures


def output_size(width: int, height: int) -> tuple[int, int]:
    """Up to x2, fitting either orientation inside 1080 x 1920; never downscale."""
    scale = max(1.0, min(2.0, 1920 / max(width, height), 1080 / min(width, height)))
    if scale == 1:
        # 4:2:0 needs even dimensions; odd high-resolution frames gain one border pixel.
        return width + width % 2, height + height % 2
    return max(width + width % 2, int(width * scale) // 2 * 2), max(height + height % 2, int(height * scale) // 2 * 2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Upscaler:
    def __init__(self, model_path: Path, tile: int):
        import cv2
        import numpy as np
        import torch
        try:
            from .rrdbnet import RRDBNet
        except ImportError:
            from rrdbnet import RRDBNet
        self.cv2, self.np, self.torch = cv2, np, torch
        self.tile = tile
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("Real-ESRGAN enhancement requires an available CUDA GPU")
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("params_ema"), dict):
            raise ValueError("Expected the official RealESRGAN_x2plus checkpoint with params_ema")
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
        self.model.load_state_dict(checkpoint["params_ema"], strict=True)
        del checkpoint
        self.model = self.model.eval().to(device="cuda", dtype=torch.float16)
        torch.cuda.reset_peak_memory_stats()

    def enhance(self, rgb, width: int, height: int):
        cv2, np, torch = self.cv2, self.np, self.torch
        source_h, source_w = rgb.shape[:2]
        padded = cv2.copyMakeBorder(rgb, 0, source_h % 2, 0, source_w % 2, cv2.BORDER_REFLECT_101)
        padded_h, padded_w = padded.shape[:2]
        enhanced = np.empty((padded_h * 2, padded_w * 2, 3), dtype=np.uint8)
        with torch.inference_mode():
            for top in range(0, padded_h, self.tile):
                bottom = min(top + self.tile, padded_h)
                for left in range(0, padded_w, self.tile):
                    right = min(left + self.tile, padded_w)
                    y0, y1 = max(0, top - TILE_PAD), min(padded_h, bottom + TILE_PAD)
                    x0, x1 = max(0, left - TILE_PAD), min(padded_w, right + TILE_PAD)
                    # All bounds are even, retaining global pixel-unshuffle alignment.
                    tile = np.ascontiguousarray(padded[y0:y1, x0:x1].transpose(2, 0, 1))
                    tensor = torch.from_numpy(tile).unsqueeze(0).to(device="cuda", dtype=torch.float16).div_(255)
                    prediction = self.model(tensor)
                    if not torch.isfinite(prediction).all().item():
                        raise RuntimeError("Model returned non-finite pixels")
                    core = prediction[0, :, (top-y0)*2:(bottom-y0)*2, (left-x0)*2:(right-x0)*2]
                    pixels = core.permute(1, 2, 0).float().clamp_(0, 1).mul_(255).round_().byte().cpu().numpy()
                    enhanced[top*2:bottom*2, left*2:right*2] = pixels
                    del prediction, core, tensor, pixels
        enhanced = enhanced[:source_h*2, :source_w*2]
        if width == source_w + source_w % 2 and height == source_h + source_h % 2:
            # Preserve source geometry at 1x; add a replicated border for odd sizes.
            ai = cv2.resize(enhanced, (source_w, source_h), interpolation=cv2.INTER_AREA)
            ai = cv2.copyMakeBorder(ai, 0, source_h % 2, 0, source_w % 2, cv2.BORDER_REPLICATE)
            reference = cv2.copyMakeBorder(rgb, 0, source_h % 2, 0, source_w % 2, cv2.BORDER_REPLICATE)
        else:
            ai = cv2.resize(enhanced, (width, height), interpolation=cv2.INTER_AREA)
            reference = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return cv2.addWeighted(ai, STRENGTH, reference, 1 - STRENGTH, 0)


def read_frame(pipe, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = pipe.readinto(view[offset:])
        if not count:
            if offset:
                raise RuntimeError(f"Decoder returned a partial frame: {offset}/{size} bytes")
            return b""
        offset += count
    return data


def stream_video(cli, upscaler, partial: Path, frames: int, width: int, height: int,
                 source_width: int, source_height: int, progress) -> int:
    decoder = encoder = None
    with tempfile.TemporaryFile() as decode_errors, tempfile.TemporaryFile() as encode_errors:
        try:
            decoder = subprocess.Popen([
                cli.ffmpeg, "-nostdin", "-v", "error", "-xerror", "-noautorotate", "-threads", "4",
                "-i", str(cli.input), "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0",
                "-threads", "4", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ], stdout=subprocess.PIPE, stderr=decode_errors)
            encoder = subprocess.Popen([
                cli.ffmpeg, "-nostdin", "-v", "error", "-xerror", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
                "-r", str(FPS), "-i", "pipe:0", "-i", str(cli.input),
                "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "-1",
                "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-threads", "4",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(partial),
            ], stdin=subprocess.PIPE, stderr=encode_errors)
            completed, last_progress = 0, 0.0
            while True:
                raw = read_frame(decoder.stdout, source_width * source_height * 3)
                if not raw:
                    break
                if completed >= frames:
                    raise RuntimeError("Decoder produced more frames than validated input")
                source = upscaler.np.frombuffer(raw, dtype=upscaler.np.uint8).reshape(source_height, source_width, 3)
                enhanced = upscaler.enhance(source, width, height)
                try:
                    encoder.stdin.write(enhanced.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    raise RuntimeError(f"Video encoder stopped: {tail_log(encode_errors)}") from exc
                completed += 1
                now = time.monotonic()
                if now - last_progress >= 1 or completed == frames:
                    progress("enhancing", completed)
                    last_progress = now
            encoder.stdin.close()
            if decoder.wait(timeout=60):
                raise RuntimeError(f"Video decoder failed: {tail_log(decode_errors)}")
            if encoder.wait(timeout=300):
                raise RuntimeError(f"Video encoder failed: {tail_log(encode_errors)}")
            if completed != frames:
                raise RuntimeError(f"Decoded frame count differs: {completed} != {frames}")
            return completed
        finally:
            stop_process(decoder)
            stop_process(encoder)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tile", type=int, default=512)
    cli = parser.parse_args(argv)
    if cli.tile < 32 or cli.tile > 1024 or cli.tile % 2:
        parser.error("--tile must be an even number between 32 and 1024")
    paths = [path.resolve() for path in (cli.input, cli.output, cli.model, cli.progress, cli.report) if path is not None]
    if len(paths) != len(set(paths)):
        parser.error("Input, output, model, progress and report paths must be different")
    return cli


def main(argv=None):
    cli = parse_args(argv)
    def terminate(signum, frame):
        # The parent cancels the process group with SIGTERM. Converting it to
        # an exception allows pipe cleanup and removal of the partial MP4.
        raise KeyboardInterrupt('Enhancement cancelled')
    signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    partial = cli.output.with_name(f".{cli.output.stem}.{os.getpid()}.partial.mp4")
    report = {"status": "running", "source": str(cli.input), "output": str(cli.output),
              "model": "RealESRGAN_x2plus", "model_path": str(cli.model), "strength": STRENGTH,
              "fps": FPS, "frames": 0, "total_frames": 0, "tile": cli.tile, "tile_pad": TILE_PAD,
              "face_enhance": False, "temporal_model": False}

    def progress(stage, frames=0):
        report.update(stage=stage, frames=frames, elapsed_seconds=round(time.monotonic()-started, 3))
        atomic_json(cli.progress, report)

    try:
        progress("validating_input")
        if not cli.input.is_file() or not cli.model.is_file():
            raise FileNotFoundError("Input video and the local RealESRGAN_x2plus model must exist")
        if cli.output.suffix.lower() != ".mp4":
            raise ValueError("Output must be an .mp4 file")
        source_info = probe(cli.ffprobe, cli.input)
        source = video_stream(source_info)
        source_width, source_height = int(source["width"]), int(source["height"])
        width, height = output_size(source_width, source_height)
        frames = count_cfr_frames(cli.ffprobe, cli.input)
        source_audio = audio_signature(cli.ffmpeg, cli.input, source_info)
        report.update(source_width=source_width, source_height=source_height, width=width, height=height,
                      total_frames=frames, audio=source_audio, model_sha256=file_sha256(cli.model))
        if source_width % 2 or source_height % 2:
            report["dimension_note"] = "Even output dimensions are required by yuv420p; aspect ratio is rounded within one pixel."
        if max(source_width, source_height) >= 1920 or min(source_width, source_height) >= 1080:
            report["resolution_note"] = "Original resolution retained; enhancement at 1x, no downscaling."
        progress("loading_model")
        upscaler = Upscaler(cli.model, cli.tile)
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        stream_video(cli, upscaler, partial, frames, width, height, source_width, source_height, progress)
        progress("validating_output", frames)
        output_info = probe(cli.ffprobe, partial)
        output_video = video_stream(output_info)
        if (int(output_video["width"]), int(output_video["height"])) != (width, height):
            raise RuntimeError("Encoded output dimensions differ from the requested size")
        # libx264 emits one video packet per raw frame. Counting MP4 packets and
        # checking its frame metadata avoids a second full decode here; the
        # application's outer pipeline performs the final full-decode check.
        output_frames = int(output_video["nb_read_packets"])
        if output_frames != frames:
            raise RuntimeError(f"Output frame count differs: {output_frames} != {frames}")
        declared_frames = output_video.get("nb_frames", str(output_frames))
        if declared_frames != "N/A" and int(declared_frames) != frames:
            raise RuntimeError("Output container frame count differs from encoded packets")
        if audio_signature(cli.ffmpeg, partial, output_info) != source_audio:
            raise RuntimeError("Copied audio packets differ from the input; output was not committed")
        report.update(status="complete", stage="complete", frames=frames,
                      elapsed_seconds=round(time.monotonic()-started, 3),
                      peak_vram_mb=round(upscaler.torch.cuda.max_memory_allocated()/1024**2, 1),
                      audio_verified=True, cfr_verified=True)
        os.replace(partial, cli.output)
        atomic_json(cli.report, report)
        atomic_json(cli.progress, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=round(time.monotonic()-started, 3))
        atomic_json(cli.report, report)
        atomic_json(cli.progress, report)
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
