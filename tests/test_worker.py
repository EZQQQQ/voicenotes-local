from pathlib import Path

from voicenotes.config import AppConfig, Paths
from voicenotes.queue import drain_queue, enqueue_session
from voicenotes.state import notify, play_start_sound


def config(tmp_path):
    return AppConfig(tmp_path / "VoiceNotes", ["cmd"], "`", "default", "qwen2.5:14b", True)


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_drain_queue_processes_items_sequentially_and_removes_them(tmp_path, monkeypatch):
    p = paths(tmp_path)
    first = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    second = tmp_path / "VoiceNotes" / "2026-08-27_143045"
    first.mkdir(parents=True)
    second.mkdir()
    enqueue_session(p, first)
    enqueue_session(p, second)
    processed = []

    monkeypatch.setattr("voicenotes.pipeline.process_session", lambda session, cfg, paths: processed.append(session))

    drain_queue(config(tmp_path), p)

    assert processed == [first, second]
    assert list((p.run / "queue").glob("*.json")) == []


def test_drain_queue_releases_lock_after_failure(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    enqueue_session(p, session)

    def fail(*args):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr("voicenotes.pipeline.process_session", fail)

    drain_queue(config(tmp_path), p)

    assert not (p.run / "pipeline.lock").exists()
    assert (session / "error.log").exists()


def test_notify_uses_osascript(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda args, check=False: calls.append(args))

    notify("Note ready", "session")

    assert calls[0][:3] == ["osascript", "-e", 'display notification "session" with title "Note ready"']


def test_play_start_sound_uses_afplay(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda args, check=False: calls.append(args))

    play_start_sound()

    assert calls[0][0] == "afplay"
