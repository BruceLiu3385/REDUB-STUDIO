#!/usr/bin/env python3
"""Run a real GPU workflow against a local REDUB-STUDIO HTTP service.

Supply your own test media. The receipt, quality report and downloaded MP4s stay
on this computer and may contain private media; they should not be committed.
This tool never starts/stops a cloud instance or installs model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit

import httpx


TERMINAL = {"completed", "failed", "cancelled"}


def validate_endpoint(value: str) -> str:
    """Allow HTTP loopback only, with no credentials, path, query or fragment."""
    try:
        endpoint = urlsplit(value)
        valid = (endpoint.scheme == "http"
                 and endpoint.hostname in ("127.0.0.1", "localhost", "::1")
                 and endpoint.username is None and endpoint.password is None
                 and endpoint.port is not None and 1024 <= endpoint.port <= 65535
                 and endpoint.path in ("", "/") and not endpoint.query and not endpoint.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Smoke tests require a loopback HTTP endpoint with an explicit port (1024-65535).")
    return value.rstrip("/")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument("--voice", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:7860")
    parser.add_argument("--receipt", type=Path, required=True, help="Local JSON receipt; also saves .mp4, .hd.mp4 and .report.json")
    parser.add_argument("--timeout", type=float, default=1200, help="Job wait deadline in seconds")
    parser.add_argument("--keep-background", action="store_true")
    parser.add_argument("--no-voice-conversion", action="store_true")
    parser.add_argument("--enhance-video", action="store_true")
    parser.add_argument("--chunk-seconds", type=int, choices=(5, 10, 15), default=15)
    args = parser.parse_args(argv)
    try:
        args.endpoint = validate_endpoint(args.endpoint)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if not args.no_voice_conversion and args.voice is None:
        parser.error("--voice is required unless --no-voice-conversion is supplied")
    if args.receipt.suffix.lower() != ".json":
        parser.error("--receipt must be a .json file")
    inputs = {path.resolve() for path in (args.source, args.portrait, args.voice) if path is not None}
    for path in (args.source, args.portrait, args.voice):
        if path is not None and not path.is_file():
            parser.error("All supplied test media must be existing regular files")
    outputs = (args.receipt, args.receipt.with_suffix(".mp4"),
               args.receipt.with_suffix(".hd.mp4"), args.receipt.with_suffix(".report.json"))
    if any(path.resolve() in inputs for path in outputs):
        parser.error("Receipt and downloaded results must not overwrite input media")
    return args


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download_url(job: dict, key: str) -> str:
    identifier = job.get("id", "")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
        raise RuntimeError("Job response contains an invalid identifier")
    expected = f"/api/jobs/{identifier}/download/{key}"
    if job.get("download_urls", {}).get(key) != expected:
        raise RuntimeError(f"Job is missing the expected {key} download endpoint")
    return expected


def download_video(client, job: dict, key: str, destination: Path) -> dict:
    """Stream HTTP video to a temporary sibling; commit only plausible MP4s."""
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest, size, prefix = hashlib.sha256(), 0, b""
        with client.stream("GET", download_url(job, key)) as response:
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";", 1)[0] != "video/mp4":
                raise RuntimeError("Video download did not return video/mp4")
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".partial",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_bytes():
                    prefix = (prefix + chunk)[:32]
                    size += len(chunk)
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if size < 1000 or prefix[4:8] != b"ftyp":
            raise RuntimeError("Video download is too small or lacks an MP4 file-type header")
        os.replace(temporary, destination)
        return {"bytes": size, "sha256": digest.hexdigest(), "file": str(destination)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_report(report: dict, *, enhanced: bool = False) -> None:
    if not isinstance(report, dict):
        raise RuntimeError("Quality report must be a JSON object")
    if report.get("full_decode") != "passed" or report.get("audio_duration_checked") is not True:
        raise RuntimeError("Quality report does not confirm full decode and audio duration checks")
    for field in ("frames", "width", "height"):
        if type(report.get(field)) is not int or report[field] <= 0:
            raise RuntimeError(f"Quality report has invalid {field}")
    if report.get("fps") != 25:
        raise RuntimeError("Quality report does not confirm the expected 25 fps output")
    if enhanced:
        enhancement = report.get("enhancement")
        validate_report(enhancement)
        if (enhancement.get("status") != "complete"
                or enhancement.get("frames") != report["frames"]
                or enhancement.get("audio_verified") is not True
                or enhancement.get("cfr_verified") is not True):
            raise RuntimeError("Enhancement report does not confirm frame and audio preservation")


def run(args) -> dict:
    started = time.monotonic()
    receipt = {"status": "running", "note": "Actual HTTP/model test; perceptual quality still requires review."}
    job = None
    with httpx.Client(base_url=args.endpoint, timeout=120, trust_env=False, follow_redirects=False) as client:
        try:
            health = client.get("/api/health").raise_for_status().json()
            receipt["health"] = health
            required = ["gpu_ready", "ffmpeg", "ffprobe"]
            if not args.no_voice_conversion:
                required.append("voice_ready")
            if args.keep_background:
                required.append("background_ready")
            if args.enhance_video:
                required.append("upscale_ready")
            missing = [field for field in required if not health.get(field)]
            if missing:
                raise RuntimeError("Model environment is not ready: " + ", ".join(missing))
            assets = {}
            for kind in ("source", "portrait", "voice"):
                path = getattr(args, kind)
                if path is None:
                    continue
                with path.open("rb") as stream:
                    response = client.post("/api/assets", data={"kind": kind}, files={"file": (path.name, stream)})
                assets[kind] = response.raise_for_status().json()
            receipt["assets"] = assets
            payload = {f"{kind}_id": asset["id"] for kind, asset in assets.items()}
            payload["options"] = {"voice_conversion": not args.no_voice_conversion,
                                  "keep_background": args.keep_background, "enhance_video": args.enhance_video,
                                  "chunk_seconds": args.chunk_seconds, "fit_video": True}
            job = client.post("/api/jobs", json=payload).raise_for_status().json()
            # Validate even before using the identifier to construct polling URLs.
            identifier = job.get("id", "")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
                raise RuntimeError("Job response contains an invalid identifier")
            deadline, last = time.monotonic() + args.timeout, None
            while job["status"] not in TERMINAL:
                key = tuple(job.get(field) for field in ("status", "stage", "message"))
                if key != last:
                    print(json.dumps({field: job.get(field) for field in ("id", "status", "stage", "progress", "message")}, ensure_ascii=False), flush=True)
                    last = key
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Workflow wait deadline exceeded")
                time.sleep(min(2, remaining))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Workflow wait deadline exceeded")
                job = client.get(f"/api/jobs/{identifier}", timeout=min(120, remaining)).raise_for_status().json()
            receipt["job"] = job
            if job["status"] != "completed":
                raise RuntimeError("Workflow ended without completion; inspect the receipt")
            report = client.get(download_url(job, "report")).raise_for_status().json()
            receipt["quality_report"] = report
            validate_report(report, enhanced=args.enhance_video)
            write_json(args.receipt.with_suffix(".report.json"), report)
            downloads = {"video": download_video(client, job, "video", args.receipt.with_suffix(".mp4"))}
            if args.enhance_video:
                downloads["video_hd"] = download_video(client, job, "video_hd", args.receipt.with_suffix(".hd.mp4"))
            receipt.update(status="passed", downloads=downloads)
        except (Exception, KeyboardInterrupt) as exc:
            receipt.update(status="failed", error_type=type(exc).__name__)
            if job is not None:
                receipt["job"] = job
                if job.get("status") not in TERMINAL:
                    try:
                        identifier = job.get("id", "")
                        if isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
                            cancellation = client.post(f"/api/jobs/{identifier}/cancel", timeout=15).raise_for_status().json()
                            receipt["cancellation"] = cancellation
                    except Exception as cancellation_error:
                        receipt["cancellation_error_type"] = type(cancellation_error).__name__
            raise
        finally:
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_json(args.receipt, receipt)
    return receipt


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        receipt = run(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Smoke test failed ({type(exc).__name__}); inspect the local receipt and job status.", flush=True)
        return 1
    print(json.dumps({"status": receipt["status"], "receipt": str(args.receipt),
                      "elapsed_seconds": receipt["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
