#!/usr/bin/env python3
"""Install the pinned official Real-ESRGAN x2 checkpoint, without GPU packages.

Existing models are verified too. A different or damaged file is preserved unless
--force is supplied; even then it is replaced only after download verification.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_NAME = "RealESRGAN_x2plus.pth"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
EXPECTED_SIZE = 67061725
EXPECTED_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
DEFAULT_MODEL = Path("/root/autodl-tmp/redub-upscale") / MODEL_NAME


class InstallError(Exception):
    """A diagnostic safe to print, without connection or proxy details."""


def verify_checkpoint(path: Path) -> None:
    """Reject anything other than the exact pinned, regular checkpoint file."""
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise InstallError("The model destination must be a regular file, not a symlink.")
        if info.st_size != EXPECTED_SIZE:
            raise InstallError("Checkpoint size does not match the pinned official model.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != EXPECTED_SHA256:
            raise InstallError("Checkpoint SHA-256 does not match the pinned official model.")
    except OSError as exc:
        raise InstallError("Could not read the checkpoint for verification.") from exc


def install_model(model: Path, *, force: bool = False, timeout: float = 120,
                  opener=None) -> str:
    """Return 'verified_existing' or 'installed'; commit only verified bytes."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise InstallError("Download timeout must be a finite positive number.")
    model = model.expanduser()
    # lexists also notices broken symlinks. Never replace links or directories.
    if os.path.lexists(model):
        if not stat.S_ISREG(model.lstat().st_mode):
            raise InstallError("The model destination must be a regular file, not a symlink.")
        try:
            verify_checkpoint(model)
            return "verified_existing"
        except InstallError as exc:
            if not force:
                raise InstallError("Existing checkpoint failed verification; use --force to replace it.") from exc
    temporary = None
    try:
        model.parent.mkdir(parents=True, exist_ok=True)
        request = Request(MODEL_URL, headers={"User-Agent": "REDUB-STUDIO-model-installer/1"})
        with tempfile.NamedTemporaryFile(mode="wb", dir=model.parent,
                                         prefix=f".{MODEL_NAME}.", suffix=".partial",
                                         delete=False) as target:
            temporary = Path(target.name)
            with (opener or urlopen)(request, timeout=timeout) as response:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > EXPECTED_SIZE:
                        raise InstallError("Download exceeds the pinned checkpoint size.")
                    target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        verify_checkpoint(temporary)
        os.replace(temporary, model)
        return "installed"
    except InstallError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        # urllib exceptions may contain authenticated proxy URLs. Do not echo them.
        raise InstallError("Checkpoint download or installation failed; existing model preserved.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--model", type=Path, help="Full checkpoint file path; overrides REDUB_UPSCALE_MODEL")
    destination.add_argument("--directory", type=Path, help="Directory for RealESRGAN_x2plus.pth; overrides the environment")
    parser.add_argument("--force", action="store_true", help="Replace an existing invalid model after verification")
    parser.add_argument("--timeout", type=float, default=120, help="Download socket timeout in seconds (default: 120)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if args.directory is not None:
        args.model = args.directory / MODEL_NAME
    elif args.model is None:
        args.model = Path(os.environ.get("REDUB_UPSCALE_MODEL") or DEFAULT_MODEL)
    args.model = args.model.expanduser()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = install_model(args.model, force=args.force, timeout=args.timeout)
    except InstallError as exc:
        print(f"Model installation failed: {exc}", file=sys.stderr)
        return 1
    except OSError:
        # Includes destination inspection/cleanup failures outside the download.
        print("Model installation failed: could not access the model destination.", file=sys.stderr)
        return 1
    print(f"RealESRGAN_x2plus: {result}; official size and SHA-256 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
