"""Local tool contract tests using an in-process HTTP transport, no GPU/network."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("smoke_web", ROOT / "tools/smoke_web.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "source.mp4"
        self.portrait = self.folder / "portrait.mp4"
        self.source.write_bytes(b"supplied fixture")
        self.portrait.write_bytes(b"supplied fixture")
        self.receipt = self.folder / "receipt.json"

    def arguments(self, *extra):
        return smoke.parse_args(["--source", str(self.source), "--portrait", str(self.portrait),
                                 "--receipt", str(self.receipt), "--no-voice-conversion", *extra])

    def report(self, enhanced=False):
        report = {"full_decode": "passed", "audio_duration_checked": True,
                  "frames": 25, "width": 512, "height": 512, "fps": 25}
        if enhanced:
            report["enhancement"] = dict(report, status="complete", audio_verified=True, cfr_verified=True)
        return report

    def test_loopback_only_and_no_credentials(self):
        for endpoint in ("http://127.0.0.1:7860", "http://localhost:7860/", "http://[::1]:7860"):
            self.assertEqual(smoke.validate_endpoint(endpoint), endpoint.rstrip("/"))
        for endpoint in ("https://localhost:7860", "http://example.org:7860", "http://localhost",
                         "http://user:secret@localhost:7860", "http://localhost:7860/api",
                         "http://localhost:7860/?token=secret", "http://localhost:7860/#secret",
                         "http://localhost:80", "http://localhost:invalid"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                smoke.validate_endpoint(endpoint)

    def test_enhancement_and_other_options_are_explicit(self):
        args = self.arguments("--enhance-video", "--keep-background", "--chunk-seconds", "5")
        self.assertTrue(args.enhance_video)
        self.assertTrue(args.keep_background)
        self.assertTrue(args.no_voice_conversion)
        self.assertEqual(args.chunk_seconds, 5)
        self.assertFalse(self.arguments().enhance_video)

    def test_receipt_cannot_overwrite_media(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.arguments("--receipt", str(self.source.with_suffix(".json")))

    def test_reports_must_confirm_decode_and_enhancement(self):
        smoke.validate_report(self.report())
        smoke.validate_report(self.report(True), enhanced=True)
        for report in (None, {}, dict(self.report(), full_decode="failed"),
                       dict(self.report(), frames=0), dict(self.report(), fps=30),
                       dict(self.report(), enhancement={})):
            with self.subTest(report=report), self.assertRaises(RuntimeError):
                smoke.validate_report(report, enhanced=True)
        report = self.report(True)
        report["enhancement"]["frames"] = 24
        with self.assertRaises(RuntimeError):
            smoke.validate_report(report, enhanced=True)

    def test_download_urls_must_stay_on_expected_service(self):
        for url in ("http://example.org/video", "//example.org/video", "/unrelated"):
            with self.assertRaises(RuntimeError):
                smoke.download_url({"id": "fixture", "download_urls": {"video": url}}, "video")

    def test_full_tool_downloads_original_hd_and_report(self):
        calls = []
        mp4 = b"\0\0\0\x18ftypisom" + b"x" * 1200
        def handler(request):
            calls.append((request.method, request.url.path))
            if request.url.path == "/api/health":
                return httpx.Response(200, json={field: True for field in
                    ("gpu_ready", "ffmpeg", "ffprobe", "upscale_ready")})
            if request.url.path == "/api/assets":
                return httpx.Response(201, json={"id": "fixture-asset"})
            if request.url.path == "/api/jobs":
                options = json.loads(request.content)["options"]
                self.assertTrue(options["enhance_video"])
                self.assertFalse(options["voice_conversion"])
                return httpx.Response(201, json={"id": "fixture", "status": "completed", "download_urls": {
                    key: f"/api/jobs/fixture/download/{key}" for key in ("video", "video_hd", "report")}})
            if request.url.path.endswith("/report"):
                return httpx.Response(200, json=self.report(True))
            return httpx.Response(200, content=mp4, headers={"content-type": "video/mp4"})
        client = httpx.Client(base_url="http://127.0.0.1:7860", transport=httpx.MockTransport(handler))
        with patch.object(smoke.httpx, "Client", return_value=client):
            receipt = smoke.run(self.arguments("--enhance-video"))
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(self.receipt.with_suffix(".mp4").read_bytes(), mp4)
        self.assertEqual(self.receipt.with_suffix(".hd.mp4").read_bytes(), mp4)
        self.assertTrue(self.receipt.with_suffix(".report.json").is_file())
        self.assertIn(("GET", "/api/jobs/fixture/download/video_hd"), calls)

    def test_timeout_cancels_job_and_writes_receipt(self):
        calls = []
        def handler(request):
            calls.append((request.method, request.url.path))
            if request.url.path == "/api/health":
                return httpx.Response(200, json={"gpu_ready": True, "ffmpeg": True, "ffprobe": True})
            if request.url.path == "/api/assets":
                return httpx.Response(201, json={"id": "fixture-asset"})
            if request.url.path.endswith("/cancel"):
                return httpx.Response(200, json={"id": "fixture", "status": "cancelled"})
            return httpx.Response(201, json={"id": "fixture", "status": "running"})
        client = httpx.Client(base_url="http://127.0.0.1:7860", transport=httpx.MockTransport(handler))
        with patch.object(smoke.httpx, "Client", return_value=client), \
             patch.object(smoke.time, "monotonic", side_effect=range(100, 200)), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                smoke.run(self.arguments("--timeout", "0.1"))
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["cancellation"]["status"], "cancelled")
        self.assertIn(("POST", "/api/jobs/fixture/cancel"), calls)

    def test_invalid_video_does_not_replace_existing_download(self):
        destination = self.receipt.with_suffix(".mp4")
        destination.write_bytes(b"existing review copy")
        with httpx.Client(base_url="http://127.0.0.1:7860", transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"not a video", headers={"content-type": "video/mp4"}))) as client:
            with self.assertRaises(RuntimeError):
                smoke.download_video(client, {"id": "fixture", "download_urls": {
                    "video": "/api/jobs/fixture/download/video"}}, "video", destination)
        self.assertEqual(destination.read_bytes(), b"existing review copy")
        self.assertFalse(list(self.folder.glob("*.partial")))


if __name__ == "__main__":
    unittest.main()
