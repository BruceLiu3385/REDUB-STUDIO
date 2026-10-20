"""Isolated, genuine speech-to-speech conversion with a fixed global timeline.

Run in REDUB_VOICE_ENV, never the web server's or TBDub's Python environment.
This adapter changes timbre; preservation of emotion and articulation needs review.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SAMPLE_RATE = 24_000
TOKEN_SAMPLES = 960  # S3 tokenizer: one 40-ms token; 24-kHz output timeline.
CONTEXT_SAMPLES = 9_600  # 0.4 s on either side, including virtual edge context.
MAX_WINDOW_SAMPLES = 480_000  # 20 s including both context regions.
CROSSFADE_HALF_SAMPLES = 240  # +/-10 ms at the same absolute sample positions.
CHECKPOINT_SIZE = 1_056_484_620
CHECKPOINT_SHA256 = "2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e"
CHECKPOINT_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    end: int
    read_start: int
    read_end: int
    pad_left: int
    pad_right: int

    @property
    def input_samples(self):
        return self.read_end - self.read_start + self.pad_left + self.pad_right


def plan_windows(total_samples: int) -> list[Window]:
    """Balanced cores on a 40-ms grid; no tiny final chunk and no lost samples."""
    if total_samples <= 0:
        raise ValueError("源音频为空")
    core_limit = MAX_WINDOW_SAMPLES - 2 * CONTEXT_SAMPLES
    count = math.ceil(total_samples / core_limit)
    token_count = math.ceil(total_samples / TOKEN_SAMPLES)
    boundaries = [token_count * i // count * TOKEN_SAMPLES for i in range(count)]
    boundaries.append(total_samples)
    windows = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if not 0 <= start < end <= total_samples:
            raise ValueError("无法建立有效的转换时间轴")
        left, right = start - CONTEXT_SAMPLES, end + CONTEXT_SAMPLES
        windows.append(Window(index, start, end, max(0, left), min(total_samples, right),
                              max(0, -left), max(0, right-total_samples)))
    if any(w.input_samples > MAX_WINDOW_SAMPLES for w in windows):
        raise ValueError("转换片段超出20秒窗口")
    return windows


def verify_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoints" / "s3gen.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少真实音色转换模型：{checkpoint}。请先运行独立声音环境安装脚本。")
    if checkpoint.stat().st_size != CHECKPOINT_SIZE:
        raise RuntimeError("Chatterbox主权重大小不符，可能下载未完成；不会使用原音频冒充转换结果")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("Chatterbox主权重SHA-256校验失败，请重新核对官方固定版本")
    return checkpoint


def align_prediction(converted, input_samples, tolerance_samples):
    """Compensate trailing token rounding without moving the prediction's origin."""
    import numpy as np
    delta = len(converted)-input_samples
    if abs(delta) > tolerance_samples:
        raise ValueError(f"token边缘时长偏差{delta/SAMPLE_RATE:+.3f}秒，超过容差")
    if delta < 0:
        converted = np.pad(converted, (0, -delta))
    elif delta > 0:
        converted = converted[:input_samples]
    return converted, delta


def retain_core(converted, input_samples, core_samples, tolerance_samples):
    """Retain the fixed-time centre; also used by single-window regression tests."""
    converted, delta = align_prediction(converted, input_samples, tolerance_samples)
    retained = converted[CONTEXT_SAMPLES:CONTEXT_SAMPLES+core_samples]
    if len(retained) != core_samples:
        raise ValueError("保留区样本数不符，拒绝拼接")
    return retained, delta


def plan_crossfades(windows: list[Window]) -> list[int]:
    """Half-width per boundary; neighbouring fades never overlap on tiny cores."""
    widths = []
    for previous, current in zip(windows, windows[1:]):
        if previous.end != current.start:
            raise ValueError("转换片段不在连续的绝对时间轴上")
        widths.append(min(CROSSFADE_HALF_SAMPLES, CONTEXT_SAMPLES,
                          (previous.end-previous.start)//2,
                          (current.end-current.start)//2))
    return widths


def blend_window(prediction, window: Window, left_half: int, right_half: int, pending=None):
    """Emit the next timeline block and hold only a small following overlap.

    Each pending tuple contains (absolute_start, samples). At boundary B, both
    predictions represent [B-H, B+H). Blend those same timestamps; never overlap
    or concatenate samples from different moments to shorten the recording.
    """
    import numpy as np
    core = window.end-window.start
    if (min(left_half, right_half) < 0 or max(left_half, right_half) > CONTEXT_SAMPLES
            or left_half+right_half > core):
        raise ValueError("交叉淡化窗口超出片段或上下文范围")
    if len(prediction) != core+2*CONTEXT_SAMPLES:
        raise ValueError("交叉淡化需要完整且已校正长度的预测上下文")
    record = None
    middle = prediction[CONTEXT_SAMPLES+left_half:CONTEXT_SAMPLES+core-right_half]
    if left_half:
        start, end = window.start-left_half, window.start+left_half
        if pending is None or pending[0] != start or len(pending[1]) != end-start:
            raise ValueError("相邻预测的交叉淡化时间位置不一致")
        previous = pending[1]
        current = prediction[CONTEXT_SAMPLES-left_half:CONTEXT_SAMPLES+left_half]
        phase = np.linspace(0.0, np.pi, 2*left_half, dtype=np.float32)
        weight = (1.0-np.cos(phase))*0.5
        # Difference form preserves identical predictions exactly, including DC.
        mixed = previous+(current-previous)*weight
        emitted = np.concatenate((mixed, middle))
        record = {"boundary_sample": window.start, "start_sample": start,
                  "end_sample": end, "samples": end-start,
                  "seconds": (end-start)/SAMPLE_RATE}
    else:
        if pending is not None:
            raise ValueError("发现未消费的交叉淡化上下文")
        emitted = middle
    following = None
    if right_half:
        following = (window.end-right_half,
                     prediction[CONTEXT_SAMPLES+core-right_half:
                                CONTEXT_SAMPLES+core+right_half].copy())
    return emitted, following, record


def ffmpeg_command() -> str:
    explicit = os.environ.get("REDUB_FFMPEG")
    if explicit:
        if Path(explicit).is_file() and os.access(explicit, os.X_OK):
            return explicit
        raise RuntimeError("REDUB_FFMPEG 指定的程序不可执行")
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError("声音转换需要ffmpeg进行流式重采样，请配置REDUB_FFMPEG") from exc


def normalize_wav(source: Path, destination: Path, *, limit_seconds=None):
    command = [ffmpeg_command(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-protocol_whitelist", "file,pipe", "-i", str(source)]
    if limit_seconds is not None:
        command += ["-t", str(limit_seconds)]
    command += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_f32le", str(destination)]
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="原配音转为本人参考音色，保留全局时间轴")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=["chatterbox"], default="chatterbox")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-tolerance", type=float, default=0.2,
                        help="仅允许token边缘补裁的最大秒数，必须介于0.15和0.30之间")
    return parser.parse_args()


def main():
    args = parse_args()
    def cancelled(_signum, _frame):
        raise InterruptedError("音色转换已取消")
    signal.signal(signal.SIGTERM, cancelled)
    root = args.root.expanduser().resolve()
    source = args.source.expanduser().resolve(strict=True)
    reference = args.reference.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if not source.is_file() or not reference.is_file():
        raise ValueError("源配音和本人参考声音必须是文件")
    if output in (source, reference):
        raise ValueError("输出不能覆盖源音频或参考声音")
    if not math.isfinite(args.edge_tolerance) or not 0.15 <= args.edge_tolerance <= 0.30:
        raise ValueError("边缘误差容许值必须在0.15至0.30秒之间")
    checkpoint = verify_checkpoint(root)
    code = root / "src" / "chatterbox" / "src"
    if not (code / "chatterbox" / "vc.py").is_file():
        raise RuntimeError("缺少已安装并固定版本的Chatterbox源码，请运行独立声音安装脚本")
    # Runtime must not fetch another model or depend on implicit network fallback.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(root / "cache" / "huggingface"))
    os.environ["PYTHONNOUSERSITE"] = "1"
    sys.path.insert(0, str(code))

    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.vc import ChatterboxVC

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("本转换后端需要已验证可用的CUDA显卡")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"backend": args.backend, "sample_rate": SAMPLE_RATE,
              "checkpoint_revision": CHECKPOINT_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
              "seed": args.seed, "context_seconds_each_side": 0.4,
              "max_model_window_seconds": 20, "edge_tolerance_seconds": args.edge_tolerance,
              "retiming": False, "segments": [],
              "note": "保持精确总时间轴不代表情绪、F0或每个音素完全不变，请预听中文与接缝。"}
    with tempfile.TemporaryDirectory(prefix=".voice-work-", dir=output.parent) as workspace:
        work = Path(workspace)
        original_info = sf.info(source)
        if not 0.5 <= original_info.duration <= 600.05:
            raise ValueError("首版音色转换支持0.5秒至10分钟源配音")
        reference_info = sf.info(reference)
        if not 3 <= reference_info.duration <= 60.05:
            raise ValueError("本人参考声音需为3至60秒，建议5至10秒干净单人语音")
        source24 = source
        if original_info.samplerate != SAMPLE_RATE or original_info.channels != 1:
            source24 = work / "source24.wav"
            normalize_wav(source, source24)
        reference24 = work / "reference24.wav"
        normalize_wav(reference, reference24, limit_seconds=10)
        normalized_info = sf.info(source24)
        expected_samples = round(original_info.frames * SAMPLE_RATE / original_info.samplerate)
        if abs(normalized_info.frames - expected_samples) > 2:
            raise RuntimeError("重采样前后时长不一致，已停止以防时间轴漂移")
        total_samples = normalized_info.frames
        windows = plan_windows(total_samples)
        fade_halves = plan_crossfades(windows)
        crossfade_records = []
        report.update(source_duration_seconds=original_info.duration,
                      source_sample_rate=original_info.samplerate,
                      output_samples=total_samples, segment_count=len(windows),
                      reference_used_seconds=sf.info(reference24).duration)
        print(f"[voice] Loading one Chatterbox VC model; {len(windows)} windows, target condition once", flush=True)
        model = ChatterboxVC.from_local(checkpoint.parent, device=args.device)
        if model.sr != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected model output sample rate: {model.sr}")
        with torch.inference_mode():
            model.set_target_voice(str(reference24))
        partial_audio = work / "converted.float.wav"
        chunk_path = work / "context.wav"
        maximum_peak = 0.0
        pending_overlap = None
        emitted_samples = 0
        with sf.SoundFile(source24) as reader, sf.SoundFile(partial_audio, "w", samplerate=SAMPLE_RATE,
                channels=1, format="WAV", subtype="FLOAT") as writer:
            for window in windows:
                reader.seek(window.read_start)
                samples = reader.read(window.read_end-window.read_start, dtype="float32", always_2d=False)
                if len(samples) != window.read_end-window.read_start or not np.isfinite(samples).all():
                    raise RuntimeError(f"源音频第{window.index+1}段读取不完整或包含非有限值")
                model_input = np.pad(samples, (window.pad_left, window.pad_right))
                sf.write(chunk_path, model_input, SAMPLE_RATE, subtype="FLOAT")
                torch.manual_seed(args.seed + window.index)
                torch.cuda.manual_seed_all(args.seed + window.index)
                segment_started = time.monotonic()
                # No target_voice_path here: reuse the one precomputed target condition.
                converted = model.generate(audio=str(chunk_path)).detach().cpu().numpy().reshape(-1)
                if not len(converted) or not np.isfinite(converted).all():
                    raise RuntimeError(f"模型第{window.index+1}段输出为空或包含非有限值")
                try:
                    prediction, delta = align_prediction(converted, len(model_input),
                                                         round(args.edge_tolerance*SAMPLE_RATE))
                    left_half = fade_halves[window.index-1] if window.index else 0
                    right_half = fade_halves[window.index] if window.index < len(fade_halves) else 0
                    retained, pending_overlap, blend = blend_window(
                        prediction, window, left_half, right_half, pending_overlap)
                except ValueError as exc:
                    raise RuntimeError(f"第{window.index+1}段：{exc}；"
                                       "未拉伸音频，也未替换成原配音") from exc
                # Only the trailing context may be padded/trimmed for token rounding.
                # Positive/negative length errors never accumulate into the next core.
                maximum_peak = max(maximum_peak, float(np.max(np.abs(retained))))
                writer.write(retained)
                emitted_samples += len(retained)
                if blend is not None:
                    crossfade_records.append(blend)
                entry = asdict(window)
                entry.update(edge_delta_samples=delta,
                             emitted_samples=len(retained),
                             crossfade_before_samples=blend["samples"] if blend else 0,
                             elapsed_seconds=round(time.monotonic()-segment_started, 3))
                report["segments"].append(entry)
                print(f"[voice] {window.index+1}/{len(windows)}: "
                      f"{window.start/SAMPLE_RATE:.3f}–{window.end/SAMPLE_RATE:.3f}s, "
                      f"token edge delta {delta/SAMPLE_RATE:+.4f}s", flush=True)
        if pending_overlap is not None or emitted_samples != total_samples:
            raise RuntimeError("交叉淡化时间轴不完整，未发布输出")
        report["crossfade"] = {
            "method": "raised_cosine_at_identical_absolute_sample_positions",
            "requested_seconds": 2*CROSSFADE_HALF_SAMPLES/SAMPLE_RATE,
            "count": len(crossfade_records), "windows": crossfade_records,
        }
        if sf.info(partial_audio).frames != total_samples:
            raise RuntimeError("音色转换总样本数不匹配，未发布输出")
        if maximum_peak < 1e-8:
            raise RuntimeError("模型没有生成可用语音，未以原声音替代")
        # One constant gain for the whole recording avoids clipping without changing
        # relative dynamics or independently normalizing individual chunks.
        gain = min(1.0, 0.99/maximum_peak)
        final_audio = work / "final.wav"
        with sf.SoundFile(partial_audio) as reader, sf.SoundFile(final_audio, "w", samplerate=SAMPLE_RATE,
                channels=1, format="WAV", subtype="FLOAT") as writer:
            while True:
                block = reader.read(240_000, dtype="float32")
                if not len(block):
                    break
                writer.write(block*gain)
        final_info = sf.info(final_audio)
        if final_info.frames != total_samples or final_info.samplerate != SAMPLE_RATE:
            raise RuntimeError("最终声音文件验证失败")
        report.update(global_gain=gain, unscaled_peak=maximum_peak,
                      elapsed_seconds=round(time.monotonic()-started, 3), success=True)
        report_file = work / "report.json"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        final_audio.replace(output)
        report_file.replace(output.with_suffix(".voice.json"))
    print(f"[voice] completed: {total_samples} samples @ {SAMPLE_RATE}Hz; output={output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[voice] ERROR: {error}", file=sys.stderr, flush=True)
        raise
