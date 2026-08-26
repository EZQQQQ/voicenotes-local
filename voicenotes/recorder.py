from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import subprocess

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
    with (session / "ffmpeg.log").open("ab"):
        subprocess.run(
            ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{device_index}", "-t", str(duration_seconds), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(session / "audio.wav")],
            check=True,
        )
    return session
