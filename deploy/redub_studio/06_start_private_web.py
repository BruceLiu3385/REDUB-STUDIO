#!/usr/bin/env python3
"""Reuse or start this private Web process, without touching Jupyter authentication."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request


class WebStartError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _absolute(path: str | Path) -> Path:
    # Keep the venv Python invocation path; resolving its symlink bypasses the venv.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _private_fd(path: Path, *, truncate: bool = False) -> int:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if truncate else os.O_APPEND)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WebStartError("启动锁或日志不是普通文件，请检查工作台数据目录。")
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _selected_process_environment(path: Path) -> dict[str, str]:
    # Inspect only these two REDUB fields, never decode or copy unrelated values.
    selected = {}
    prefixes = {b"REDUB_DATA_DIR=": "REDUB_DATA_DIR", b"REDUB_PORT=": "REDUB_PORT"}
    pending = b""
    with path.open("rb") as stream:
        while block := stream.read(4096):
            pieces = (pending + block).split(b"\0")
            pending = pieces.pop()
            for piece in pieces:
                for prefix, name in prefixes.items():
                    if piece.startswith(prefix):
                        selected[name] = os.fsdecode(piece[len(prefix):])
            if len(selected) == len(prefixes):
                break
    return selected


def _find_webs(web_python: Path, data_dir: Path, proc_root: Path) -> list[tuple[int, int]]:
    matches = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / "cmdline").read_bytes().split(b"\0")
            if command and command[-1] == b"":
                command.pop()
            if command != [os.fsencode(web_python), b"-m", b"redub_studio.server"]:
                continue
            environment = _selected_process_environment(directory / "environ")
            configured_data = environment.get("REDUB_DATA_DIR")
            if not configured_data:
                raise WebStartError("检测到工作台进程，但无法确认其数据目录；为避免重复启动已停止。")
            if _absolute(configured_data).resolve() != data_dir:
                continue
            try:
                port = int(environment["REDUB_PORT"])
                if not 1024 <= port <= 65535:
                    raise ValueError()
            except (KeyError, ValueError):
                raise WebStartError("检测到本工作台进程，但无法确认端口；为避免重复启动已停止。") from None
            matches.append((int(directory.name), port))
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process exited during inspection.
        except PermissionError:
            raise WebStartError("无法检查现有进程；为避免重复启动工作台，请联系部署人员检查权限。") from None
    return matches


def _process_start(pid: int, proc_root: Path) -> str:
    # Linux field 22; split after comm because process names can contain spaces.
    return (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def _pending_launcher(marker: Path, script: Path, proc_root: Path) -> tuple[int, int] | None:
    if not marker.exists():
        return None
    if marker.is_symlink():
        raise WebStartError("启动状态文件异常，请联系部署人员检查。")
    try:
        record = json.loads(marker.read_text())
        pid, port = int(record["pid"]), int(record["port"])
        if not 1024 <= port <= 65535:
            raise ValueError()
        if _process_start(pid, proc_root) != record["start"]:
            marker.unlink()
            return None
        command = (proc_root / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if command == [b"/bin/bash", os.fsencode(script)]:
            return pid, port
        # A live process in another state is not safe to duplicate.
        raise WebStartError("上一次启动进程尚未完成切换，请稍后重新运行此单元格。")
    except (FileNotFoundError, ProcessLookupError):
        marker.unlink(missing_ok=True)
        return None
    except (ValueError, TypeError, KeyError, IndexError):
        raise WebStartError("启动状态文件无法确认，请联系部署人员检查；未启动第二个进程。") from None


def _healthy(port: int, timeout: float) -> bool:
    # Loopback only: no proxy environment, redirects, credentials or Jupyter request.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/health")
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.loads(response.read(65537))
            return (
                response.status == 200 and isinstance(payload, dict)
                and payload.get("status") == "ok"
                and all(isinstance(payload.get(key), bool)
                        for key in ("gpu_ready", "voice_ready", "ffmpeg", "ffprobe"))
                and isinstance(payload.get("limits"), dict)
            )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _child_environment(web_env: Path, data_dir: Path, port: int) -> dict[str, str]:
    # Explicit allowlist: do not forward the notebook's full environment or tokens.
    allowed = (
        "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH", "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
        "REDUB_PROJECT_ROOT", "REDUB_WEB_ROOT", "REDUB_TBDUB_ROOT", "REDUB_TBDUB_ENV",
        "REDUB_VOICE_ROOT", "REDUB_VOICE_ENV", "REDUB_SEPARATOR_ROOT", "REDUB_SEPARATOR_ENV",
        "REDUB_FFMPEG", "REDUB_FFPROBE", "REDUB_TBDUB_CODE_REF", "REDUB_UPSCALE_MODEL",
    )
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    environment.update(REDUB_WEB_ENV=str(web_env), REDUB_DATA_DIR=str(data_dir),
                       REDUB_PORT=str(port), PYTHONUNBUFFERED="1")
    return environment


def ensure_web(*, web_env: str | Path | None = None, data_dir: str | Path | None = None,
               start_script: str | Path | None = None, port: int = 7860,
               timeout: float = 30, proc_root: Path = Path("/proc")) -> int:
    """Return a verified local port. Existing unready processes are never duplicated."""
    if not 0 < timeout <= 30 or not 1024 <= port <= 65535:
        raise WebStartError("启动等待时间或端口设置无效。")
    deadline = time.monotonic() + timeout
    web_root = Path(os.environ.get("REDUB_WEB_ROOT", "/root/autodl-tmp/redub-web"))
    web_env = _absolute(web_env or os.environ.get("REDUB_WEB_ENV", str(web_root / "env")))
    data_dir = _absolute(data_dir or os.environ.get("REDUB_DATA_DIR", "/root/autodl-tmp/redub-data")).resolve()
    web_python = web_env / "bin/python"
    script = _absolute(start_script or Path(__file__).with_name("03_start_web.sh"))
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path, log_path = data_dir / ".web-start.lock", data_dir / "web-start.log"
    marker = data_dir / ".web-starting.json"
    lock_fd = _private_fd(lock_path)
    launched = None
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise WebStartError("另一个启动操作尚未结束，请稍后重新运行此单元格。") from None
                time.sleep(0.1)
        if time.monotonic() >= deadline:
            raise WebStartError("启动等待时间已用完，请重新运行此单元格。")
        matches = _find_webs(web_python, data_dir, proc_root)
        if len(matches) > 1:
            raise WebStartError("检测到多个本工作台进程，请联系部署人员停止多余进程后再试。")
        if matches:
            selected_pid, selected_port = matches[0]
        elif pending := _pending_launcher(marker, script, proc_root):
            selected_pid, selected_port = pending
        else:
            if not web_python.is_file() or not os.access(web_python, os.X_OK) or not script.is_file():
                raise WebStartError("工作台程序或运行环境尚未安装完整，请联系部署人员检查。")
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    # Match Uvicorn's listener semantics: recently closed HTTP
                    # connections in TIME_WAIT must not block a service restart.
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind(("127.0.0.1", port))
            except OSError:
                raise WebStartError(f"端口 {port} 已被其他程序占用，未启动新的工作台。") from None
            if deadline - time.monotonic() < 5:
                raise WebStartError("剩余启动等待时间不足，请重新运行此单元格；未启动新进程。")
            log_fd = _private_fd(log_path)
            try:
                launched = subprocess.Popen(
                    ["/bin/bash", str(script)], stdin=subprocess.DEVNULL,
                    stdout=log_fd, stderr=subprocess.STDOUT, start_new_session=True,
                    close_fds=True, env=_child_environment(web_env, data_dir, port),
                )
            finally:
                os.close(log_fd)
            selected_pid, selected_port = launched.pid, port
            try:
                record = {"pid": selected_pid, "port": selected_port,
                          "start": _process_start(selected_pid, proc_root)}
                with os.fdopen(_private_fd(marker, truncate=True), "w") as stream:
                    json.dump(record, stream)
            except (FileNotFoundError, ProcessLookupError):
                raise WebStartError(f"工作台启动进程已退出，请联系部署人员查看 {log_path}。") from None
        while time.monotonic() < deadline:
            if launched is not None and launched.poll() is not None:
                raise WebStartError(f"工作台未能启动，请联系部署人员查看 {log_path}。")
            remaining = deadline - time.monotonic()
            if _healthy(selected_port, min(5, max(0.01, remaining))):
                current = _find_webs(web_python, data_dir, proc_root)
                if current == [(selected_pid, selected_port)]:
                    marker.unlink(missing_ok=True)
                    return selected_port
                raise WebStartError("工作台进程在连接时发生变化，请重新运行此单元格；未再启动第二个进程。")
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        raise WebStartError(
            f"30 秒内工作台尚未就绪，可能仍在启动。请稍后重新运行此单元格；不会重复启动。日志：{log_path}"
        )
    except OSError:
        raise WebStartError("工作台启动文件或进程检查失败，请联系部署人员检查数据目录与权限。") from None
    finally:
        os.close(lock_fd)


def main() -> int:
    try:
        print(ensure_web())
        return 0
    except WebStartError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
