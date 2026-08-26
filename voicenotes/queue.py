from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

from .config import Paths
from .state import atomic_write_json, read_json


def enqueue_session(paths: Paths, session: Path) -> Path:
    queue_dir = paths.run / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    item = queue_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{session.name}.json"
    atomic_write_json(item, {"session_path": str(session), "enqueued_at": datetime.now().isoformat(timespec="seconds")})
    return item


def _lock_path(paths: Paths) -> Path:
    return paths.run / "pipeline.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_pipeline_lock(paths: Paths) -> bool:
    paths.run.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(paths)
    if lock.exists():
        try:
            payload = read_json(lock)
            if _pid_alive(int(payload.get("pid", 0))):
                return False
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")}, handle)
    return True


def release_pipeline_lock(paths: Paths) -> None:
    _lock_path(paths).unlink(missing_ok=True)


def try_spawn_worker(paths: Paths) -> bool:
    if _lock_path(paths).exists():
        return False
    worker_log = paths.run / "worker.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    handle = worker_log.open("ab")
    subprocess.Popen([sys.executable, "-m", "voicenotes", "worker"], stdin=subprocess.DEVNULL, stdout=handle, stderr=handle)
    handle.close()
    return True
