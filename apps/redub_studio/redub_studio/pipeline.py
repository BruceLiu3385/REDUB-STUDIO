"""Real media pipeline. Missing model configuration is an error, never a demo result."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import sys
from fractions import Fraction

from .media import CancelledError, ffmpeg, probe, run, plan_segments, silence_centres, save_json, executable, choose_seams, count_video_frames


def model_environment(prefix: str):
    value = os.environ.get(f"REDUB_{prefix}_ENV")
    root = os.environ.get(f"REDUB_{prefix}_ROOT")
    if not value or not root:
        raise RuntimeError(f"尚未配置 {prefix} 模型。请先完成云端模型联调。")
    python = Path(value) / "bin/python"
    if not python.is_file():
        raise RuntimeError(f"模型环境不存在：{python}")
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONNOUSERSITE="1")
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return Path(root), python, environment


def valid_media(path, duration=None, video=False, expected_frames=None, cancelled=lambda: False):
    try:
        info = probe(path)
        if not (info["has_video"] if video else info["has_audio"]):
            return False
        actual_duration = info["video_duration"] if video else info["audio_duration"]
        tolerance = .041 if video else .12
        if not video and info.get("audio_samples") is not None:
            tolerance = 1 / info["audio_sample_rate"] + 1e-7
        if duration is not None and abs(actual_duration - duration) > tolerance:
            return False
        if expected_frames is not None:
            stream = next(item for item in info["streams"] if item.get("codec_type") == "video"
                          and not item.get("disposition", {}).get("attached_pic"))
            if Fraction(stream.get("avg_frame_rate") or "0/1") != 25:
                return False
            if count_video_frames(path, cancelled=cancelled) != expected_frames:
                return False
        return True
    except CancelledError:
        raise
    except Exception:
        return False


def enhance_video_file(job_dir: Path, source: Path, report, cancelled,
                       *, progress_start=96, progress_end=98) -> tuple[Path, dict]:
    """Run super-resolution after the lip-sync subprocess has released its GPU."""
    model = Path(os.environ.get('REDUB_UPSCALE_MODEL',
                                '/root/autodl-tmp/redub-upscale/RealESRGAN_x2plus.pth'))
    if not model.is_file():
        raise RuntimeError('AI 高清模型尚未安装，请关闭高清增强或联系部署人员。')
    _, python, environment = model_environment('TBDUB')
    source_info = probe(source)
    frames = count_video_frames(source, cancelled=cancelled)
    duration = frames / 25
    target = job_dir / 'final_hd.mp4'
    progress_file, report_file = job_dir / 'upscale-progress.json', job_dir / 'upscale-report.json'
    marker_file = job_dir / 'upscale-complete.json'
    signature = {'source_path': str(source.resolve()), 'model_path': str(model.resolve()),
                 'source_size': source.stat().st_size, 'source_mtime_ns': source.stat().st_mtime_ns,
                 'model_size': model.stat().st_size, 'model_mtime_ns': model.stat().st_mtime_ns,
                 'profile': 'x2plus-1080p-blend070-v1'}
    cached = False
    try:
        cached_report = json.loads(report_file.read_text())
        cached = (isinstance(cached_report, dict) and json.loads(marker_file.read_text()) == signature
                  and valid_media(target, duration, video=True, expected_frames=frames, cancelled=cancelled))
    except (OSError, ValueError):
        pass
    if not cached:
        marker_file.unlink(missing_ok=True)
        progress_file.unlink(missing_ok=True)
        report('upscale', progress_start, 'AI 高清增强：保留原比例、声音和口型时间轴')
        last = [-1]
        def heartbeat():
            try:
                data = json.loads(progress_file.read_text())
                completed = min(frames, max(0, int(data['frames'])))
                if completed != last[0]:
                    last[0] = completed
                    report('upscale', progress_start + (progress_end-progress_start) * completed / frames,
                           f'AI 高清增强：{completed}/{frames}帧')
            except (OSError, ValueError, KeyError, TypeError):
                pass
        run([python, Path(__file__).parent / 'workers/upscale.py', '--input', source,
             '--output', target, '--model', model, '--ffmpeg', executable('ffmpeg'),
             '--ffprobe', executable('ffprobe'), '--progress', progress_file, '--report', report_file],
            log=job_dir / 'logs/upscale.log', env=environment, cancelled=cancelled,
            heartbeat=heartbeat)
    enhanced = probe(target)
    audio_tolerance = 1024 / max(1, enhanced['audio_sample_rate']) + .002
    if (not valid_media(target, duration, video=True, expected_frames=frames, cancelled=cancelled)
            or not enhanced['has_audio']
            or abs(enhanced['audio_duration'] - source_info['audio_duration']) > audio_tolerance):
        raise RuntimeError('高清增强输出的帧数、时长或音轨不一致；原版成片已保留。')
    report('upscale', 99, '检查高清成片完整性')
    ffmpeg(['-v', 'error', '-xerror', '-i', str(target), '-f', 'null', '-'],
           job_dir / 'logs/upscale-verify.log', cancelled)
    enhancement = json.loads(report_file.read_text())
    if not isinstance(enhancement, dict):
        raise RuntimeError('高清增强报告格式错误；原版成片已保留。')
    enhancement.update(width=enhanced['width'], height=enhanced['height'], frames=frames,
                       fps=25, duration=duration, full_decode='passed', audio_duration_checked=True)
    save_json(marker_file, signature)
    return target, enhancement


def run_enhancement(job_dir: Path, source_video: Path, report, cancelled) -> dict:
    """Enhance an existing finished video without regenerating voice or lip sync."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'logs').mkdir(exist_ok=True)
    original = job_dir / 'final.mp4'
    source_marker = job_dir / 'original-source.json'
    signature = {'path': str(source_video.resolve()), 'size': source_video.stat().st_size,
                 'mtime_ns': source_video.stat().st_mtime_ns}
    same_original = False
    try:
        same_original = original.is_file() and json.loads(source_marker.read_text()) == signature
    except (OSError, ValueError):
        pass
    if not same_original:
        partial = job_dir / 'original.partial.mp4'
        # The manager supplies a validated finished video from another job.
        shutil.copyfile(source_video, partial)
        if cancelled():
            raise CancelledError('任务已取消')
        partial.replace(original)
        save_json(source_marker, signature)
    enhanced, quality = enhance_video_file(job_dir, original, report, cancelled,
                                         progress_start=2, progress_end=95)
    quality['note'] = 'AI 超分会估计细节，请对比原版的脸部、牙齿、口型和连续帧。'
    save_json(job_dir / 'quality_report.json', quality)
    report('completed', 100, '高清增强完成，可对比原版与 AI 高清版')
    return {'video': original.name, 'video_hd': enhanced.name, 'report': 'quality_report.json'}


def run_pipeline(job_dir: Path, assets: dict, options: dict, report, cancelled) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    logs = job_dir / "logs"
    logs.mkdir(exist_ok=True)
    source, portrait = Path(assets["source"]), Path(assets["portrait"])
    report("extract", 2, "检查媒体并提取原配音")
    source_info, portrait_info = probe(source), probe(portrait)
    if not source_info["has_audio"] or not source_info["has_video"]:
        raise ValueError("原文案视频必须同时包含画面和声音")
    if not portrait_info["has_video"]:
        raise ValueError("人物素材必须为动作视频")
    source_audio = job_dir / "source.wav"
    extraction_marker = job_dir / "source_audio_complete.json"
    if not extraction_marker.is_file() or not valid_media(source_audio):
        partial_audio = job_dir / "source.partial.wav"
        ffmpeg(["-i", str(source), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(partial_audio)], logs / "extract.log", cancelled)
        extracted = probe(partial_audio)
        if not extracted.get("audio_samples") or extracted["audio_sample_rate"] != 48000:
            raise RuntimeError("原配音未完整提取为可计数的PCM音频")
        partial_audio.replace(source_audio)
        save_json(extraction_marker, {"samples": extracted["audio_samples"], "sample_rate": 48000})
    # The extracted PCM samples, rather than a video's rounded container length,
    # define all audio cuts and the global 25 fps output frame grid.
    duration = probe(source_audio)["audio_duration"]
    if not .5 <= duration <= 600:
        raise ValueError("首版支持0.5秒至10分钟配音")
    frames = math.ceil(duration * 25 - 1e-7)
    working_duration = frames / 25
    portrait_duration = portrait_info["video_duration"]
    ratio = duration / portrait_duration
    if ratio > 1.03 + 1e-9 or (ratio > 1 + 1e-6 and not options.get("fit_video", True)):
        raise ValueError(f"本人视频{portrait_duration:.2f}秒，配音{duration:.2f}秒；请补充足够时长的视频，系统不会循环动作。")
    if cancelled():
        raise CancelledError("任务已取消")
    vocal = source_audio
    background = None
    if options.get("keep_background", False):
        report("separate", 8, "分离人声与伴奏；原伴奏会在最后回混")
        separator_env = os.environ.get("REDUB_SEPARATOR_ENV")
        if not separator_env or not (Path(separator_env) / "bin/python").is_file():
            raise RuntimeError("未配置背景音乐分离模块；请配置 REDUB_SEPARATOR_ENV 或使用无背景音乐的原视频。")
        split_dir = job_dir / "separated"
        vocal = split_dir / "vocals.wav"
        background = split_dir / "background.wav"
        if not valid_media(vocal, duration) or not valid_media(background, duration):
            run([Path(separator_env) / "bin/python", Path(__file__).parent / "workers/separate.py",
                "--source", source_audio, "--output-dir", split_dir], log=logs / "separate.log", cancelled=cancelled)
    converted = job_dir / "converted.wav"
    if not valid_media(converted, duration):
        raw_converted = job_dir / "voice_raw.wav"
        if options.get("voice_conversion", True):
            report("voice", 15, "转换为本人音色；保留原文案的节奏，效果需预听确认")
            root, python, environment = model_environment("VOICE")
            reference = Path(assets["voice"])
            reference_wav = job_dir / "reference.wav"
            ffmpeg(["-i", str(reference), "-map", "0:a:0", "-t", "30", "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(reference_wav)], logs / "reference.log", cancelled)
            # Separate process releases voice GPU memory before loading TBDub.
            run([python, Path(__file__).parent / "workers/voice_convert.py", "--root", root,
                 "--source", vocal, "--reference", reference_wav, "--output", raw_converted],
                log=logs / "voice.log", env=environment, cancelled=cancelled)
            voice_duration = probe(raw_converted)["duration"]
            tolerance = max(.15, min(.5, duration * .005))
            if abs(voice_duration - duration) > tolerance:
                raise RuntimeError(f"转换音频时长偏差过大（{voice_duration:.3f} vs {duration:.3f}秒），为避免音画漂移已停止。")
        else:
            report("voice", 20, "使用原配音进行口型测试")
            raw_converted = vocal
        # Only compensate codec/token-grid rounding; never silently retime speech.
        ffmpeg(["-i", str(raw_converted), "-af", f"apad,atrim=duration={duration:.9f}", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(converted)], logs / "align_voice.log", cancelled)
    save_json(job_dir / "audio_ready.json", {"source_audio": "source.wav", "audio": "converted.wav"})
    report("prepare", 28, "统一人物视频与配音时间轴，寻找停顿切点")
    normalized = job_dir / "portrait_aligned.mp4"
    if not valid_media(normalized, working_duration, video=True, expected_frames=frames, cancelled=cancelled):
        setpts = f"setpts={ratio:.12f}*(PTS-STARTPTS)," if ratio > 1 + 1e-6 else "setpts=PTS-STARTPTS,"
        # At most one still end frame covers rounding up to the next 25 fps tick;
        # this does not loop the action video or stretch beyond the 3% gate.
        filters = setpts + "fps=25,tpad=stop_mode=clone:stop_duration=0.04,scale=w='min(1920,iw)':h='min(1920,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1"
        ffmpeg(["-i", str(portrait), "-t", f"{working_duration:.9f}", "-an", "-vf", filters,
                "-frames:v", str(frames), "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", str(normalized)], logs / "portrait.log", cancelled)
        if not valid_media(normalized, working_duration, video=True, expected_frames=frames, cancelled=cancelled):
            raise RuntimeError("人物视频未正确对齐到目标帧数，已停止以避免后续循环补帧")
    pauses = silence_centres(converted, logs / "silences.log", cancelled)
    segments = plan_segments(frames, float(options.get("chunk_seconds", 15)), pauses)
    clips_dir = job_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    manifest = {"fps": 25, "source_duration": duration, "working_duration": working_duration, "segments": []}
    for segment in segments:
        prefix = clips_dir / f"segment_{segment.index:04d}"
        item = segment.dictionary()
        item.update(video=str(prefix.with_suffix(".mp4")), audio=str(prefix.with_suffix(".wav")),
                    output_dir=str(clips_dir / f"generated_{segment.index:04d}"), name=f"segment_{segment.index:04d}")
        count = segment.context_end - segment.context_start
        clip_duration = count / 25
        if not valid_media(Path(item["video"]), clip_duration, video=True, expected_frames=count, cancelled=cancelled):
            ffmpeg(["-i", str(normalized), "-vf", f"trim=start_frame={segment.context_start}:end_frame={segment.context_end},setpts=PTS-STARTPTS",
                    "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", item["video"]], logs / "split_video.log", cancelled)
            if not valid_media(Path(item["video"]), clip_duration, video=True, expected_frames=count, cancelled=cancelled):
                raise RuntimeError(f"第{segment.index+1}段人物输入帧数不完整")
        if not valid_media(Path(item["audio"]), clip_duration):
            ffmpeg(["-i", str(converted), "-af", f"apad,atrim=start={segment.context_start/25:.9f}:end={segment.context_end/25:.9f},asetpts=PTS-STARTPTS",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", item["audio"]], logs / "split_audio.log", cancelled)
        manifest["segments"].append(item)
    manifest_path = job_dir / "segments.json"
    save_json(manifest_path, manifest)
    report("lipsync", 35, f"开始生成{len(segments)}段口型；模型在本批任务内只加载一次")
    root, python, environment = model_environment("TBDUB")
    last_count = [-1]
    def update_progress():
        count = sum((Path(item["output_dir"]) / "complete.json").is_file() for item in manifest["segments"])
        if count != last_count[0]:
            last_count[0] = count
            report("lipsync", 35 + 48 * count / len(segments), f"口型生成：{count}/{len(segments)}段完成")
    run([python, Path(__file__).parent / "workers/tbdub_batch.py", "--root", root, "--manifest", manifest_path],
        log=logs / "tbdub.log", env=environment, cancelled=cancelled, heartbeat=update_progress)
    report("stitch", 85, "裁去重叠区，逐帧拼接并统一封装音轨")
    for item in manifest["segments"]:
        expected = (item["context_end"] - item["context_start"]) / 25
        generated = Path(item["output_dir"]) / f"{item['name']}_tbdub.mp4"
        if not valid_media(generated, expected, video=True, expected_frames=item["context_end"] - item["context_start"], cancelled=cancelled):
            raise RuntimeError(f"第{item['index']+1}段输出不完整，可继续处理")
    adjusted = choose_seams(manifest["segments"], cancelled)
    save_json(job_dir / "seams.json", adjusted)
    trimmed = []
    for item in adjusted:
        generated = Path(item["output_dir"]) / f"{item['name']}_tbdub.mp4"
        retained = clips_dir / f"retained_{item['index']:04d}.mp4"
        # Re-trim on retry too: selected seam positions can change after regeneration.
        left, right = item["start"] - item["context_start"], item["end"] - item["context_start"]
        core = (item["end"] - item["start"]) / 25
        ffmpeg(["-i", str(generated), "-vf", f"trim=start_frame={left}:end_frame={right},setpts=PTS-STARTPTS",
                    "-an", "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", str(retained)], logs / "trim.log", cancelled)
        if not valid_media(retained, core, video=True, expected_frames=item["end"] - item["start"], cancelled=cancelled):
            raise RuntimeError(f"第{item['index']+1}段保留画面帧数不符，未拼接成片")
        trimmed.append(retained)
    # Paths are generated names within this job; no user filenames enter concat syntax.
    concat = clips_dir / "concat.txt"
    concat.write_text("".join(f"file '{path.name}'\n" for path in trimmed))
    track = converted
    if background:
        track = job_dir / "mixed.wav"
        ffmpeg(["-i", str(converted), "-i", str(background), "-filter_complex",
                "[0:a][1:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=0[a]",
                "-map", "[a]", "-t", f"{duration:.9f}", "-ar", "48000", "-c:a", "pcm_s16le", str(track)], logs / "mix.log", cancelled)
    temporary = job_dir / "final.partial.mp4"
    ffmpeg(["-f", "concat", "-safe", "1", "-i", str(concat), "-i", str(track), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temporary)], logs / "compose.log", cancelled)
    report("verify", 95, "检查完整视频、声音与片段边界")
    final_info = probe(temporary)
    audio_tolerance = 1024 / max(1, final_info["audio_sample_rate"]) + .002
    if (not valid_media(temporary, working_duration, video=True, expected_frames=frames, cancelled=cancelled)
            or not final_info["has_audio"] or abs(final_info["audio_duration"] - duration) > audio_tolerance):
        raise RuntimeError("最终视频时长或音轨检查失败，未标记完成")
    ffmpeg(["-v", "error", "-xerror", "-i", str(temporary), "-f", "null", "-"], logs / "verify.log", cancelled)
    final = job_dir / "final.mp4"
    temporary.replace(final)
    quality = {"duration": duration, "frames": frames, "fps": 25,
        "segments": len(segments), "seam_times": [item["start"]/25 for item in adjusted[1:]],
        "portrait_speed_adjusted": ratio > 1 + 1e-6, "audio_duration_checked": True,
        "width": final_info['width'], "height": final_info['height'],
        "note": "技术检查不等于自然度合格；请预听音色并查看列出的接缝。", "full_decode": "passed"}
    outputs = {"video": final.name, "audio": converted.name, "source_audio": source_audio.name,
               "report": "quality_report.json"}
    if options.get('enhance_video', False):
        enhanced, enhancement = enhance_video_file(job_dir, final, report, cancelled)
        quality['enhancement'] = enhancement
        outputs['video_hd'] = enhanced.name
    save_json(job_dir / "quality_report.json", quality)
    report("completed", 100, "生成完成，请预览音色、口型和接缝后下载")
    return outputs
