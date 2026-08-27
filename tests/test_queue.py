import json
import threading
from datetime import datetime

import pytest

from voicenotes.config import Paths
from voicenotes.queue import acquire_pipeline_lock, enqueue_session, release_pipeline_lock, try_spawn_worker


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


@pytest.fixture(autouse=True)
def prevent_worker_spawn(monkeypatch):
    monkeypatch.setattr("voicenotes.queue.try_spawn_worker", lambda paths: False)


def test_enqueue_session_writes_json_item(tmp_path):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)

    item = enqueue_session(p, session)

    payload = json.loads(item.read_text(encoding="utf-8"))
    assert payload["session_path"] == str(session)
    assert item.parent == p.run / "queue"
    assert item.name.endswith("_2026-08-27_143012.json")


def test_enqueue_session_does_not_overwrite_same_second_item(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 8, 27, 14, 30, 12)

    monkeypatch.setattr("voicenotes.queue.datetime", FixedDatetime)

    first = enqueue_session(p, session)
    second = enqueue_session(p, session)

    assert first != second
    assert sorted((p.run / "queue").glob("*.json")) == sorted([first, second])


def test_pipeline_lock_is_atomic(tmp_path):
    p = paths(tmp_path)

    assert acquire_pipeline_lock(p) is True
    assert acquire_pipeline_lock(p) is False
    release_pipeline_lock(p)
    assert acquire_pipeline_lock(p) is True


def test_pipeline_lock_keeps_partially_initialized_lock(tmp_path):
    p = paths(tmp_path)
    p.run.mkdir(parents=True)
    lock = p.run / "pipeline.lock"
    lock.write_text('{"pid":', encoding="utf-8")

    assert acquire_pipeline_lock(p) is False
    assert lock.read_text(encoding="utf-8") == '{"pid":'


def test_concurrent_acquisition_keeps_partially_initialized_lock(tmp_path):
    p = paths(tmp_path)
    p.run.mkdir(parents=True)
    lock = p.run / "pipeline.lock"
    lock.write_text('{"pid":', encoding="utf-8")
    barrier = threading.Barrier(3)
    results = []

    def acquire():
        barrier.wait()
        results.append(acquire_pipeline_lock(p))

    first = threading.Thread(target=acquire)
    second = threading.Thread(target=acquire)
    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()

    assert results == [False, False]
    assert lock.read_text(encoding="utf-8") == '{"pid":'


def test_try_spawn_worker_noops_when_lock_held(tmp_path, monkeypatch):
    p = paths(tmp_path)
    acquire_pipeline_lock(p)
    launched = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **kwargs: launched.append(args))

    assert try_spawn_worker(p) is False
    assert launched == []
