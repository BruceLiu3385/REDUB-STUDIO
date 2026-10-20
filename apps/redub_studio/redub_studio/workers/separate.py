"""Offline Demucs adapter; FFmpeg/SoundFile audio I/O avoids TorchCodec.

Install/prefetch with deploy/redub_studio/02_install_separator.sh first.
Only the installer may retrieve the explicitly pinned official checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile

MODEL_FILE = "955717e8-8726e21a.th"
MODEL_SHA_PREFIX = "8726e21a"
SAMPLE_RATE = 44100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    def interrupted(_signum, _frame):
        raise InterruptedError("分离任务已取消。")

    signal.signal(signal.SIGTERM, interrupted)
    root = Path(os.environ.get("REDUB_SEPARATOR_ROOT", "/root/autodl-tmp/redub-separator")).expanduser().resolve()
    os.environ["TORCH_HOME"] = str(root / "model-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    checkpoint = root / "model-cache" / "hub" / "checkpoints" / MODEL_FILE
    receipt_path = root / "logs" / "runtime-check.json"
    if not checkpoint.is_file() or not receipt_path.is_file():
        raise RuntimeError("缺少已预取的 htdemucs 权重或校验记录；先运行 02_install_separator.sh。")
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("prefetch_completed") or receipt.get("model") != "htdemucs":
        raise RuntimeError("htdemucs 预取记录不完整；重新运行 02_install_separator.sh。")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if not actual.startswith(MODEL_SHA_PREFIX) or actual != receipt.get("sha256"):
        raise RuntimeError("htdemucs 缓存校验失败；拒绝自动下载或使用未校验权重。")

    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    def refuse_download(*_args, **_kwargs):
        raise RuntimeError("分离任务只允许离线运行；请在安装阶段完成官方模型预取。")

    # load_state_dict_from_url uses this function on a cache miss.
    torch.hub.download_url_to_file = refuse_download
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("分离环境看不到 CUDA；请选择正确实例或显式用 --device cpu 测试。")
    source = args.source.expanduser().resolve(strict=True)
    if source.suffix.lower() != ".wav":
        raise ValueError("分离适配器只接受流水线提取的本地 WAV。")
    source_info = sf.info(str(source))
    if source_info.duration < 0.5 or source_info.duration > 600.1:
        raise ValueError("分离源音频需为 0.5–600 秒 WAV。")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configured = os.environ.get("REDUB_FFMPEG")
    ffmpeg = configured or shutil.which("ffmpeg")
    if not ffmpeg or not Path(ffmpeg).is_file() or not os.access(ffmpeg, os.X_OK):
        raise RuntimeError("缺少 FFmpeg；请配置 REDUB_FFMPEG。")
    with tempfile.TemporaryDirectory(prefix=".separate-", dir=output_dir) as temporary:
        temp = Path(temporary)
        normalized = temp / "source-44100.wav"
        subprocess.run([
            ffmpeg, "-nostdin", "-y", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-i", str(source), "-map", "0:a:0", "-vn", "-ar", str(SAMPLE_RATE),
            "-ac", "2", "-c:a", "pcm_f32le", str(normalized),
        ], check=True, timeout=180)
        samples, sample_rate = sf.read(str(normalized), dtype="float32", always_2d=True)
        if sample_rate != SAMPLE_RATE or samples.shape[1] != 2 or not np.isfinite(samples).all():
            raise ValueError("FFmpeg 归一化音频格式或采样值无效。")
        original_frames = len(samples)
        if original_frames == 0:
            raise ValueError("源音频为空。")
        model = get_model("htdemucs")
        if model.samplerate != SAMPLE_RATE or model.audio_channels != 2:
            raise RuntimeError("htdemucs 模型配置与 44.1kHz 双声道预期不符。")
        model.eval()
        audio = torch.from_numpy(samples.T.copy())
        mean = audio.mean(dim=0).mean()
        scale = audio.mean(dim=0).std()
        with torch.inference_mode():
            if float(audio.abs().max()) <= 1e-8:
                vocal = torch.zeros_like(audio)
                silence = True
            else:
                if float(scale) <= 1e-8:
                    raise ValueError("源音频近似恒定直流信号，无法进行有效人声分离。")
                normalized_audio = (audio - mean) / scale
                separated = apply_model(
                    model, normalized_audio[None], device=device, shifts=1,
                    split=True, overlap=0.25, progress=True, num_workers=0,
                )[0]
                vocal = separated[model.sources.index("vocals")].cpu() * scale + mean
                silence = False
        if vocal.shape != audio.shape or not torch.isfinite(vocal).all().item():
            raise RuntimeError("分离结果采样数/通道数不符；拒绝拉伸或隐式补齐。")
        # The residual is sample-aligned and sums back to the exact input mix.
        background = audio - vocal
        for name, tensor in (("vocals", vocal), ("background", background)):
            target = temp / f"{name}.wav"
            sf.write(str(target), tensor.T.numpy(), SAMPLE_RATE, subtype="FLOAT")
            info = sf.info(str(target))
            if info.frames != original_frames or info.samplerate != SAMPLE_RATE or info.channels != 2:
                raise RuntimeError("分离文件写出校验失败。")
        record = {
            "model": "htdemucs", "checkpoint_sha256": actual, "sample_rate": SAMPLE_RATE,
            "frames": original_frames, "duration": original_frames / SAMPLE_RATE,
            "source_duration": source_info.duration, "device": str(device),
            "silent_input": silence, "retiming": False,
            "background_method": "normalized_source_minus_predicted_vocals",
            "audio_io": "ffmpeg_and_soundfile", "quality_listened": False,
        }
        (temp / "separation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        for name in ("vocals.wav", "background.wav", "separation.json"):
            (temp / name).replace(output_dir / name)
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
