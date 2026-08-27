from pathlib import Path

import pytest

from voicenotes.config import AppConfig, Paths
from voicenotes.queue import drain_queue, enqueue_session
from voicenotes.state import notify, play_start_sound


def config(tmp_path):
    return AppConfig(tmp_path / "VoiceNotes", ["cmd"], "`", "default", "qwen2.5:14b", True)


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


@pytest.fixture(autouse=True)
def prevent_worker_spawn(monkeypatch):
    monkeypatch.setattr("voicenotes.queue.try_spawn_worker", lambda paths: False)


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


def test_drain_queue_processes_item_enqueued_while_worker_holds_lock(tmp_path, monkeypatch):
    p = paths(tmp_path)
    first = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    second = tmp_path / "VoiceNotes" / "2026-08-27_143045"
    first.mkdir(parents=True)
    second.mkdir()
    enqueue_session(p, first)
    processed = []

    def process(session, cfg, worker_paths):
        processed.append(session)
        if session == first:
            enqueue_session(worker_paths, second)

    monkeypatch.setattr("voicenotes.pipeline.process_session", process)

    drain_queue(config(tmp_path), p)

    assert processed == [first, second]
    assert list((p.run / "queue").glob("*.json")) == []


def test_drain_queue_releases_lock_after_failure(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    item = enqueue_session(p, session)

    def fail(*args):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr("voicenotes.pipeline.process_session", fail)

    drain_queue(config(tmp_path), p)

    assert not (p.run / "pipeline.lock").exists()
    assert (session / "error.log").exists()
    assert item.exists()


def test_drain_queue_moves_malformed_item_aside_and_processes_valid_items(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    malformed = p.run / "queue" / "broken.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    enqueue_session(p, session)
    processed = []
    monkeypatch.setattr("voicenotes.pipeline.process_session", lambda path, cfg, worker_paths: processed.append(path))

    drain_queue(config(tmp_path), p)

    assert processed == [session]
    assert not malformed.exists()
    assert (p.run / "queue" / "malformed" / malformed.name).exists()
    assert "Malformed queue item" in (p.run / "last-error.txt").read_text(encoding="utf-8")


def test_drain_queue_worker_log_contains_only_lifecycle_entries(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    enqueue_session(p, session)
    monkeypatch.setattr("voicenotes.pipeline.process_session", lambda session, cfg, paths: None)

    drain_queue(config(tmp_path), p)

    messages = [line.partition(" ")[2] for line in (p.run / "worker.log").read_text(encoding="utf-8").splitlines()]
    assert messages == ["worker started", "worker drained queue"]


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
