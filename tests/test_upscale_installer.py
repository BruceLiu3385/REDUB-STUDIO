"""Offline installer integrity/commit tests; no GPU or network is used."""

import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("upscale_installer", ROOT / "deploy/redub_studio/07_install_upscale.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    payload = b"offline checkpoint fixture"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.model = self.folder / installer.MODEL_NAME
        for name, value in (("EXPECTED_SIZE", len(self.payload)),
                            ("EXPECTED_SHA256", hashlib.sha256(self.payload).hexdigest())):
            replacement = patch.object(installer, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def opener(self, request, *, timeout):
        self.assertEqual(request.full_url, installer.MODEL_URL)
        self.assertGreater(timeout, 0)
        return io.BytesIO(self.payload)

    def test_download_verified_before_atomic_replace(self):
        real_replace = os.replace
        def checked_replace(source, destination):
            source = Path(source)
            self.assertEqual(source.parent, self.model.parent)
            self.assertFalse(self.model.exists())
            installer.verify_checkpoint(source)
            real_replace(source, destination)
        with patch.object(installer.os, "replace", side_effect=checked_replace) as replace:
            result = installer.install_model(self.model, opener=self.opener)
        self.assertEqual(result, "installed")
        self.assertEqual(self.model.read_bytes(), self.payload)
        replace.assert_called_once()
        self.assertEqual(list(self.folder.iterdir()), [self.model])

    def test_existing_valid_checkpoint_never_downloaded_even_with_force(self):
        self.model.write_bytes(self.payload)
        def forbidden(*args, **kwargs):
            self.fail("A valid existing model must not trigger a network request")
        for force in (False, True):
            self.assertEqual(installer.install_model(self.model, force=force, opener=forbidden), "verified_existing")

    def test_size_and_sha256_rejected_separately(self):
        self.model.write_bytes(self.payload[:-1])
        with self.assertRaisesRegex(installer.InstallError, "size"):
            installer.verify_checkpoint(self.model)
        self.model.write_bytes(b"x" * len(self.payload))
        with self.assertRaisesRegex(installer.InstallError, "SHA-256"):
            installer.verify_checkpoint(self.model)

    def test_existing_invalid_model_requires_force(self):
        self.model.write_bytes(b"old invalid model")
        with self.assertRaisesRegex(installer.InstallError, "--force"):
            installer.install_model(self.model, opener=lambda *a, **k: self.fail("No request expected"))
        self.assertEqual(self.model.read_bytes(), b"old invalid model")

    def test_force_preserves_original_until_verified_replace(self):
        self.model.write_bytes(b"old invalid model")
        real_replace = os.replace
        def checked_replace(source, destination):
            self.assertEqual(self.model.read_bytes(), b"old invalid model")
            installer.verify_checkpoint(Path(source))
            real_replace(source, destination)
        with patch.object(installer.os, "replace", side_effect=checked_replace):
            installer.install_model(self.model, force=True, opener=self.opener)
        self.assertEqual(self.model.read_bytes(), self.payload)

    def test_failed_download_never_replaces_existing_file(self):
        for payload in (self.payload[:-1], b"x" * len(self.payload), self.payload + b"extra"):
            with self.subTest(payload=payload):
                self.model.write_bytes(b"old invalid model")
                with self.assertRaises(installer.InstallError):
                    installer.install_model(self.model, force=True,
                                            opener=lambda *a, **k: io.BytesIO(payload))
                self.assertEqual(self.model.read_bytes(), b"old invalid model")
                self.assertEqual(list(self.folder.iterdir()), [self.model])

    def test_interrupted_download_removes_partial_file(self):
        class Interrupted(io.BytesIO):
            def read(self, count=-1):
                raise OSError("connection interrupted")
        with self.assertRaises(installer.InstallError):
            installer.install_model(self.model, opener=lambda *a, **k: Interrupted())
        self.assertFalse(self.model.exists())
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_failed_atomic_replace_keeps_existing_model(self):
        self.model.write_bytes(b"old invalid model")
        with patch.object(installer.os, "replace", side_effect=OSError("denied")):
            with self.assertRaises(installer.InstallError):
                installer.install_model(self.model, force=True, opener=self.opener)
        self.assertEqual(self.model.read_bytes(), b"old invalid model")
        self.assertEqual(list(self.folder.iterdir()), [self.model])

    def test_directory_is_not_replaced(self):
        self.model.mkdir()
        with self.assertRaisesRegex(installer.InstallError, "regular file"):
            installer.install_model(self.model, force=True, opener=self.opener)
        self.assertTrue(self.model.is_dir())

    def test_cli_precedence_and_environment(self):
        environment = self.folder / "environment.pth"
        explicit = self.folder / "explicit.pth"
        directory = self.folder / "custom"
        with patch.dict(os.environ, {"REDUB_UPSCALE_MODEL": str(environment)}):
            self.assertEqual(installer.parse_args([]).model, environment)
            self.assertEqual(installer.parse_args(["--model", str(explicit)]).model, explicit)
            self.assertEqual(installer.parse_args(["--directory", str(directory)]).model,
                             directory / installer.MODEL_NAME)
        with patch.dict(os.environ, {"REDUB_UPSCALE_MODEL": ""}):
            self.assertEqual(installer.parse_args([]).model, installer.DEFAULT_MODEL)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            installer.parse_args(["--model", str(explicit), "--directory", str(directory)])

    def test_invalid_timeouts_do_not_download(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(installer.InstallError):
                installer.install_model(self.model, timeout=timeout, opener=self.opener)

    def test_proxy_credentials_not_echoed(self):
        diagnostic = io.StringIO()
        with patch.object(installer, "urlopen", side_effect=URLError("http://private-user:private-password@proxy.example")):
            with contextlib.redirect_stderr(diagnostic):
                result = installer.main(["--model", str(self.model)])
        self.assertEqual(result, 1)
        self.assertNotIn("private-user", diagnostic.getvalue())
        self.assertNotIn("private-password", diagnostic.getvalue())
        self.assertNotIn("proxy.example", diagnostic.getvalue())


class OfficialPinTests(unittest.TestCase):
    def test_official_checkpoint_is_pinned(self):
        self.assertEqual(installer.MODEL_URL, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
        self.assertEqual(installer.EXPECTED_SIZE, 67061725)
        self.assertEqual(installer.EXPECTED_SHA256, "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb")


if __name__ == "__main__":
    unittest.main()
