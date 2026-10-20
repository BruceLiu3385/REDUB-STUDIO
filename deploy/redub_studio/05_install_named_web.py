#!/usr/bin/env python3
"""Experimental named Jupyter service, retained for reference.

This deployment experiment did not provide a working named-service route on the
tested AutoDL setup. Use 03_start_web.sh / 06_start_private_web.py and the numeric
authenticated /proxy/<port>/ route instead. It is not part of normal installation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile


# The generic file is also read when AutoDL supplies an explicit --config path.
DEFAULT_CONFIG = "/root/.jupyter/jupyter_config.json"
PROJECT_ROOT = Path(os.environ.get("REDUB_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
START_SCRIPT = str(PROJECT_ROOT / "deploy" / "redub_studio" / "03_start_web.sh")
SERVER_NAME = "redub"
SERVER_CONFIG = {
    "command": ["/bin/bash", START_SCRIPT],
    "environment": {"REDUB_PORT": "{port}"},
    "timeout": 120,
    "absolute_url": False,
    "launcher_entry": {"title": "口播换声工作台"},
}


class ConfigError(Exception):
    """A safe diagnostic that contains no existing configuration values."""


def reject_symlinks(path: Path) -> None:
    for candidate in (*reversed(path.parents), path):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ConfigError("Refusing a symlink in the configuration path.")


def read_existing(path: Path) -> bytes | None:
    reject_symlinks(path)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ConfigError("The configuration must be a regular file.")
        return stream.read()


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError("Existing JSON contains a duplicate key; resolve it first.")
        result[key] = value
    return result


def reject_constant(_value: str) -> None:
    raise ConfigError("Existing JSON contains a nonstandard numeric constant.")


def merged_config(raw: bytes | None) -> tuple[dict, bool]:
    try:
        config = json.loads(
            raw.decode("utf-8") if raw is not None else "{}",
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ConfigError("Existing configuration is not UTF-8; no changes made.") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Existing JSON is invalid at line {exc.lineno}, column {exc.colno}; no changes made."
        ) from exc
    if not isinstance(config, dict):
        raise ConfigError("Existing JSON must contain an object.")
    proxy = config.setdefault("ServerProxy", {})
    if not isinstance(proxy, dict):
        raise ConfigError("Existing ServerProxy configuration must be an object.")
    servers = proxy.setdefault("servers", {})
    if not isinstance(servers, dict):
        raise ConfigError("Existing ServerProxy.servers configuration must be an object.")
    unchanged = servers.get(SERVER_NAME) == SERVER_CONFIG
    # Only this named service is managed here; retain every other configuration key.
    servers[SERVER_NAME] = SERVER_CONFIG
    return config, unchanged


def backup_original(path: Path, raw: bytes) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.backup-{timestamp}-{secrets.token_hex(4)}")
    descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def install(path: Path, original: bytes | None, config: dict) -> Path | None:
    reject_symlinks(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    reject_symlinks(path)
    if read_existing(path) != original:
        raise ConfigError("Configuration changed during preview; rerun before installing.")
    backup = backup_original(path, original) if original is not None else None
    content = (json.dumps(config, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if read_existing(path) != original:
            raise ConfigError("Configuration changed during installation; no replacement made.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experimental named service (not a validated AutoDL route); prefer the numeric proxy route.",
        epilog="No service restart, GPU power action, network request or authentication change.",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="verified root-level Jupyter config JSON (default: %(default)s)",
    )
    parser.add_argument("--install", action="store_true", help="apply the previewed merge")
    args = parser.parse_args(argv)
    if not Path(args.config).expanduser().is_absolute():
        parser.error("--config must be an absolute path")
    path = Path(os.path.abspath(os.path.expanduser(args.config)))
    try:
        original = read_existing(path)
        config, unchanged = merged_config(original)
        backup = None
        if args.install and not unchanged:
            backup = install(path, original, config)
        # Deliberately never print config, original, or the previous redub entry.
        print(json.dumps({
            "mode": "install" if args.install else "preview",
            "config_file": str(path),
            "action": "unchanged" if unchanged else ("update" if original is not None else "create"),
            "desired_server": {SERVER_NAME: SERVER_CONFIG},
            "backup_file": str(backup) if backup else None,
            "services_restarted": False,
            "gpu_power_action": False,
        }, ensure_ascii=False, indent=2))
        print("No restart or authentication change. Use the existing Jupyter login.")
        print("After Jupyter next starts with this config: <jupyter-base>/redub/.")
        if not args.install:
            print("Preview only; rerun with --install to write the configuration.")
        return 0
    except ConfigError as exc:
        print(f"Configuration refused: {exc}", file=sys.stderr)
    except OSError as exc:
        # Avoid dumping file contents or a traceback containing caller locals.
        print(f"Configuration operation failed: {exc.strerror or type(exc).__name__}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
