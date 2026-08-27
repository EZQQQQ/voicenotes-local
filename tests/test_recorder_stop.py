import signal

import pytest

from voicenotes.config import Paths
from voicenotes.recorder import stop_recording
from voicenotes.state import atomic_write_json, read_json


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_stop_sends_sigint_clears_state_and_enqueues(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 4096)
    atomic_write_json(p.run / "current-recording.json", {"session_path": str(session), "pid": 555, "resolved_device": "Mic"})
    sent = []
    enqueued = []

    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: True)
    monkeypatch.setattr("os.kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr("voicenotes.recorder._wait_pid_exit", lambda pid, timeout: True)
    monkeypatch.setattr("voicenotes.queue.enqueue_session", lambda paths, session_path: enqueued.append(session_path) or (p.run / "queue" / "item.json"))
    monkeypatch.setattr("voicenotes.queue.try_spawn_worker", lambda paths: True)

    stopped = stop_recording(p)

    assert stopped == session
    assert sent == [(555, signal.SIGINT)]
    assert enqueued == [session]
    assert not (p.run / "current-recording.json").exists()


def test_stop_escalates_and_repairs_after_sigkill(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 4096)
    atomic_write_json(p.run / "current-recording.json", {"session_path": str(session), "pid": 777, "resolved_device": "Mic"})
    sent = []
    waits = iter([False, False])

    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: True)
    monkeypatch.setattr("os.kill", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr("voicenotes.recorder._wait_pid_exit", lambda pid, timeout: next(waits, True))
    monkeypatch.setattr("voicenotes.recorder.repair_wav", lambda session_path: True)
    monkeypatch.setattr("voicenotes.queue.enqueue_session", lambda paths, session_path: p.run / "queue" / "item.json")
    monkeypatch.setattr("voicenotes.queue.try_spawn_worker", lambda paths: True)

    stop_recording(p)

    assert sent == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    session_state = read_json(session / "session.json")
    assert session_state["recording_interrupted"] is True
    assert session_state["wav_repair_succeeded"] is True


def test_stop_waits_for_exit_after_sigkill_before_repair(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    atomic_write_json(p.run / "current-recording.json", {"session_path": str(session), "pid": 777})
    events = []
    waits = iter([False, False, True])

    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: True)
    monkeypatch.setattr("os.kill", lambda pid, sig: events.append(("signal", sig)))
    monkeypatch.setattr("voicenotes.recorder._wait_pid_exit", lambda pid, timeout: events.append(("wait", timeout)) or next(waits))
    monkeypatch.setattr("voicenotes.recorder.repair_wav", lambda session_path: events.append(("repair", session_path)) or True)
    monkeypatch.setattr("voicenotes.queue.enqueue_session", lambda paths, session_path: p.run / "queue" / "item.json")
    monkeypatch.setattr("voicenotes.queue.try_spawn_worker", lambda paths: True)

    stop_recording(p)

    assert events == [
        ("signal", signal.SIGINT),
        ("wait", 10),
        ("signal", signal.SIGTERM),
        ("wait", 3),
        ("signal", signal.SIGKILL),
        ("wait", 1),
        ("repair", session),
    ]


def test_stop_refuses_recycled_pid(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    atomic_write_json(p.run / "current-recording.json", {"session_path": str(session), "pid": 888})
    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: False)

    with pytest.raises(RuntimeError, match="stale recording state"):
        stop_recording(p)

    assert not (p.run / "current-recording.json").exists()
    assert "stale recording state" in (p.run / "last-error.txt").read_text(encoding="utf-8")
