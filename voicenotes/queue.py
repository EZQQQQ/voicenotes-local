from __future__ import annotations

from datetime import datetime
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

from .config import AppConfig, Paths
from .state import atomic_write_json, atomic_write_text, read_json


_LOCK_HANDLES: dict[Path, int] = {}


def enqueue_session(paths: Paths, session: Path) -> Path:
    queue_dir = paths.run / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    item = queue_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{time.monotonic_ns()}_{uuid.uuid4().hex}_{session.name}.json"
    atomic_write_json(item, {"session_path": str(session), "enqueued_at": datetime.now().isoformat(timespec="seconds")})
    return item


def _lock_path(paths: Paths) -> Path:
    return paths.run / "pipeline.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_pipeline_lock(paths: Paths) -> bool:
    paths.run.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(paths)
    if lock in _LOCK_HANDLES:
        return False
    while lock.exists():
        try:
            fd = os.open(lock, os.O_RDWR)
        except FileNotFoundError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        try:
            try:
                payload = json.load(os.fdopen(os.dup(fd), "r", encoding="utf-8"))
                if _pid_alive(int(payload.get("pid", 0))):
                    return False
            except (OSError, ValueError, json.JSONDecodeError):
                return False
            try:
                if os.path.samestat(os.stat(lock), os.fstat(fd)):
                    lock.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    payload = json.dumps({"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")})
    fd, temp_name = tempfile.mkstemp(prefix=".pipeline.", suffix=".tmp", dir=paths.run)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.link(temp_name, lock)
    except FileExistsError:
        return False
    finally:
        Path(temp_name).unlink(missing_ok=True)
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    _LOCK_HANDLES[lock] = fd
    return True


def release_pipeline_lock(paths: Paths) -> None:
    lock = _lock_path(paths)
    fd = _LOCK_HANDLES.pop(lock, None)
    if fd is None:
        return
    try:
        lock.unlink(missing_ok=True)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _worker_log(paths: Paths, message: str) -> None:
    path = paths.run / "worker.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")


def drain_queue(config: AppConfig, paths: Paths) -> None:
    if not acquire_pipeline_lock(paths):
        return

    _worker_log(paths, "worker started")
    try:
        queue_dir = paths.run / "queue"
        for item in sorted(queue_dir.glob("*.json")):
            try:
                session = Path(str(read_json(item)["session_path"]))
                _worker_log(paths, f"processing {item.name}")
                from . import pipeline

                pipeline.process_session(session, config, paths)
            except Exception as error:
                atomic_write_text(session / "error.log", f"{error}\n")
                _worker_log(paths, f"failed {item.name}: {error}")
            else:
                item.unlink(missing_ok=True)
                _worker_log(paths, f"completed {item.name}")
    finally:
        release_pipeline_lock(paths)
        _worker_log(paths, "worker drained queue")


def try_spawn_worker(paths: Paths) -> bool:
    if _lock_path(paths).exists():
        return False
    worker_log = paths.run / "worker.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    handle = worker_log.open("ab")
    subprocess.Popen([sys.executable, "-m", "voicenotes", "worker"], stdin=subprocess.DEVNULL, stdout=handle, stderr=handle)
    handle.close()
    return True
