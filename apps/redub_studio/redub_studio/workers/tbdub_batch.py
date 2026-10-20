"""Run inside the isolated TBDub environment; one model load per batch."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    cli = parser.parse_args()
    repo = cli.root / "src/TBDub"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import inference
    from inference import parse_args, build_pipeline, preprocess_inputs, run_inference
    import torch
    manifest = json.loads(cli.manifest.read_text())
    pipeline = embedding = None
    checkpoint = cli.root / "checkpoints"
    # Our segment WAVs have an exact integer number of 25fps frames. Upstream's
    # float->int duration conversion can lose one frame (e.g. 452 becomes 451).
    # Override only in this wrapper; never patch the installed upstream source.
    def exact_audio_length(path):
        with wave.open(str(path)) as audio:
            samples, rate = audio.getnframes(), audio.getframerate()
        return (samples * 25 + rate // 2) // rate
    inference.get_audio_length = exact_audio_length
    for item in manifest["segments"]:
        output_dir = Path(item["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{item['name']}_tbdub.mp4"
        done = output_dir / "complete.json"
        signature = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        expected_frames = item["context_end"] - item["context_start"]
        if target.is_file() and done.is_file():
            try:
                receipt = json.loads(done.read_text())
                result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=nb_read_frames", "-of", "json", str(target)], capture_output=True, text=True, check=True)
                count = int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])
                if receipt.get("signature") == signature and count == expected_frames:
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"], check=True)
                    print(f"[batch] resume: {item['name']}", flush=True)
                    continue
            except (ValueError, KeyError, IndexError, OSError, subprocess.CalledProcessError):
                pass
            done.unlink(missing_ok=True)
        args = parse_args(["--video", item["video"], "--audio", item["audio"],
            "--checkpoint-dir", str(checkpoint), "--config", str(checkpoint / "config.json"),
            "--inference-mode", "student", "--num-student-steps", "2", "--sigma-shift", "1.0",
            "--motion-from-latents", "--seed", "42", "--no-cpu-offload",
            "--output-dir", str(output_dir), "--sample-name", item["name"],
            "--preprocess-report", str(output_dir / "face_detection.json")])
        if pipeline is None:
            pipeline, embedding = build_pipeline(args)
        sample = preprocess_inputs(item["video"], item["audio"], args)
        run_inference(pipeline, embedding, sample, item["name"], args)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"], check=True)
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(target)], capture_output=True, text=True, check=True)
        actual_frames = int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])
        if actual_frames != expected_frames:
            raise RuntimeError(f"Output frame count differs: {actual_frames} != {expected_frames}")
        complete = done.with_suffix(".tmp")
        complete.write_text(json.dumps({"index": item["index"], "output": str(target), "signature": signature, "frames": actual_frames}))
        complete.replace(done)
        print(f"[batch] completed {item['index']+1}/{len(manifest['segments'])}", flush=True)
        del sample
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
