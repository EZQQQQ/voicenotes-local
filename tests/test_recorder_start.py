from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from voicenotes.config import AppConfig, Paths
from voicenotes.recorder import create_session_dir, list_audio_devices, record_test, resolve_audio_device, start_recording
from voicenotes.state import read_json


def config(tmp_path, audio_device="default"):
    return AppConfig(
        output_root=tmp_path / "VoiceNotes",
        hotkey_mods=["cmd"],
        hotkey_key="`",
        audio_device=audio_device,
        ollama_model="qwen2.5:14b",
        auto_open=True,
    )


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_create_session_dir_uses_second_precision(tmp_path):
    session = create_session_dir(tmp_path, datetime(2026, 8, 27, 14, 30, 12))

    assert session == tmp_path / "2026-08-27_143012"
    assert session.is_dir()


def test_list_audio_devices_parses_avfoundation_stderr(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stderr="\n".join(
                [
                    "[AVFoundation indev @ 0x1] AVFoundation audio devices:",
                    "[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone",
                    "[AVFoundation indev @ 0x1] [1] External Mic",
                ]
            )
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    assert list_audio_devices() == ["MacBook Pro Microphone", "External Mic"]


def test_resolve_audio_device_default_is_index_zero():
    assert resolve_audio_device("default", ["Built-in", "USB"]) == 0


def test_resolve_audio_device_requires_exact_name():
    assert resolve_audio_device("USB Mic", ["Built-in", "USB Mic"]) == 1
    with pytest.raises(ValueError, match="Configured audio device not found"):
        resolve_audio_device("USB", ["Built-in", "USB Mic"])


def test_start_recording_writes_state_and_ffmpeg_log(tmp_path, monkeypatch):
    launched = {}

    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])
    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: False)
    monkeypatch.setattr("voicenotes.recorder.create_session_dir", lambda output_root, now=None: output_root / "2026-08-27_143012")

    class FakePopen:
        pid = 4321

        def __init__(self, args, stderr, stdout, stdin):
            launched["args"] = args
            launched["stderr_name"] = Path(stderr.name).name

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr("voicenotes.recorder.play_start_sound", lambda: None)

    session = start_recording(config(tmp_path), paths(tmp_path))

    assert session == tmp_path / "VoiceNotes" / "2026-08-27_143012"
    assert launched["args"][-1] == str(session / "audio.wav")
    assert launched["stderr_name"] == "ffmpeg.log"
    state = read_json(tmp_path / "run" / "current-recording.json")
    assert state["session_path"] == str(session)
    assert state["pid"] == 4321
    assert state["resolved_device"] == "MacBook Pro Microphone"


def test_record_test_uses_shared_device_resolution_without_state_file(tmp_path, monkeypatch):
    launched = {}
    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])

    def fake_run(args, stderr, check):
        launched["args"] = args
        launched["stderr_name"] = Path(stderr.name).name
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    session = record_test(config(tmp_path), paths(tmp_path), duration_seconds=3)

    assert "-t" in launched["args"]
    assert "3" in launched["args"]
    assert launched["stderr_name"] == "ffmpeg.log"
    assert not (tmp_path / "run" / "current-recording.json").exists()
    assert session.name.startswith("record-test_")
