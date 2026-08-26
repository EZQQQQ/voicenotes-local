from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import os
import signal
import subprocess
import time

from .config import AppConfig, Paths
from .state import atomic_write_json, read_json


DEVICE_RE = re.compile(r"\[(\d+)\]\s+(.+)$")


def list_audio_devices(ffmpeg: str = "ffmpeg") -> list[str]:
    result = subprocess.run(
        [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        check=False,
    )
    devices: list[str] = []
    in_audio = False
    for line in result.stderr.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if in_audio and "AVFoundation video devices:" in line:
            break
        if in_audio:
            match = DEVICE_RE.search(line)
            if match:
                devices.append(match.group(2).strip())
    return devices


def resolve_audio_device(configured: str, devices: list[str]) -> int:
    if configured == "default":
        if not devices:
            raise ValueError("No avfoundation audio devices found")
        return 0
    for index, name in enumerate(devices):
        if name == configured:
            return index
    available = "\n".join(f"- {name}" for name in devices)
    raise ValueError(f"Configured audio device not found: {configured}\nAvailable devices:\n{available}")


def create_session_dir(output_root: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    session = output_root / stamp
    session.mkdir(parents=True, exist_ok=False)
    return session


def is_live_ffmpeg(pid: int) -> bool:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True, check=False)
    return result.returncode == 0 and Path(result.stdout.strip()).name == "ffmpeg"


def _active_recording_path(paths: Paths) -> Path:
    return paths.run / "current-recording.json"


def _clear_stale_recording_state(paths: Paths) -> None:
    state_path = _active_recording_path(paths)
    if state_path.exists():
        state_path.unlink()


def _wait_pid_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def repair_wav(session: Path) -> bool:
    audio = session / "audio.wav"
    interrupted = session / "audio.interrupted.wav"
    if not audio.exists():
        return False
    audio.replace(interrupted)
    result = subprocess.run(["ffmpeg", "-y", "-i", str(interrupted), "-c", "copy", str(audio)], check=False)
    return result.returncode == 0


def stop_recording(paths: Paths) -> Path:
    state_path = _active_recording_path(paths)
    if not state_path.exists():
        raise RuntimeError("No active recording")
    recording = read_json(state_path)
    pid = int(recording.get("pid", 0))
    session = Path(str(recording.get("session_path", "")))
    if not pid or not is_live_ffmpeg(pid):
        raise RuntimeError("stale recording state")

    interrupted = False
    os.kill(pid, signal.SIGINT)
    if not _wait_pid_exit(pid, 10):
        os.kill(pid, signal.SIGTERM)
        if not _wait_pid_exit(pid, 3):
            os.kill(pid, signal.SIGKILL)
            interrupted = True

    if interrupted:
        if not _wait_pid_exit(pid, 1):
            atomic_write_json(session / "session.json", {**recording, "recording_interrupted": True, "wav_repair_succeeded": False, "stop_failed": True})
            raise RuntimeError("recording process did not exit after SIGKILL")
        repaired = repair_wav(session)
        atomic_write_json(session / "session.json", {**recording, "recording_interrupted": True, "wav_repair_succeeded": repaired})

    state_path.unlink(missing_ok=True)
    from .queue import enqueue_session, try_spawn_worker

    enqueue_session(paths, session)
    try_spawn_worker(paths)
    return session


def start_recording(config: AppConfig, paths: Paths) -> Path:
    state_path = _active_recording_path(paths)
    if state_path.exists():
        state = read_json(state_path)
        pid = int(state.get("pid", 0))
        if pid and is_live_ffmpeg(pid):
            raise RuntimeError(f"Recording already active: {state.get('session_path')}")
        _clear_stale_recording_state(paths)
    devices = list_audio_devices()
    device_index = resolve_audio_device(config.audio_device, devices)
    resolved_name = devices[device_index]
    session = create_session_dir(config.output_root)
    session.mkdir(parents=True, exist_ok=True)
    ffmpeg_log = session / "ffmpeg.log"
    log_handle = ffmpeg_log.open("ab")
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{device_index}", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(session / "audio.wav")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=log_handle,
    )
    log_handle.close()
    atomic_write_json(
        state_path,
        {
            "session_path": str(session),
            "pid": process.pid,
            "resolved_device": resolved_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ffmpeg_log_path": str(ffmpeg_log),
        },
    )
    return session


def record_test(config: AppConfig, paths: Paths, duration_seconds: int = 10) -> Path:
    devices = list_audio_devices()
    device_index = resolve_audio_device(config.audio_device, devices)
    session = config.output_root / f"record-test_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    session.mkdir(parents=True, exist_ok=False)
    with (session / "ffmpeg.log").open("ab") as log_handle:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{device_index}", "-t", str(duration_seconds), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(session / "audio.wav")],
            stderr=log_handle,
            check=True,
        )
    return session
