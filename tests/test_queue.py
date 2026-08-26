import json

from voicenotes.config import Paths
from voicenotes.queue import acquire_pipeline_lock, enqueue_session, release_pipeline_lock, try_spawn_worker


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_enqueue_session_writes_json_item(tmp_path):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)

    item = enqueue_session(p, session)

    payload = json.loads(item.read_text(encoding="utf-8"))
    assert payload["session_path"] == str(session)
    assert item.parent == p.run / "queue"
    assert item.name.endswith("_2026-08-27_143012.json")


def test_pipeline_lock_is_atomic(tmp_path):
    p = paths(tmp_path)

    assert acquire_pipeline_lock(p) is True
    assert acquire_pipeline_lock(p) is False
    release_pipeline_lock(p)
    assert acquire_pipeline_lock(p) is True


def test_try_spawn_worker_noops_when_lock_held(tmp_path, monkeypatch):
    p = paths(tmp_path)
    acquire_pipeline_lock(p)
    launched = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **kwargs: launched.append(args))

    assert try_spawn_worker(p) is False
    assert launched == []
