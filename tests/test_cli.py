import json

from voicenotes.cli import main


def test_config_json_command_outputs_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / ".voicenotes"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        "\n".join(
            [
                f'output_root = "{tmp_path / "VoiceNotes"}"',
                'audio_device = "default"',
                'ollama_model = "qwen2.5:14b"',
                "auto_open = true",
                "[hotkey]",
                'mods = ["cmd"]',
                'key = "`"',
            ]
        ),
        encoding="utf-8",
    )

    code = main(["config", "--json"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["hotkey"] == {"mods": ["cmd"], "key": "`"}


def test_unknown_command_returns_internal_error(capsys):
    code = main(["not-a-command"])

    assert code == 2
    assert "usage:" in capsys.readouterr().err


def test_stop_command_prints_stopped_session(monkeypatch, capsys):
    monkeypatch.setattr("voicenotes.cli.stop_recording", lambda paths: paths.output_root / "2026-08-27_143012")

    assert main(["stop"]) == 0
    assert capsys.readouterr().out.strip().endswith("2026-08-27_143012")


def test_toggle_stops_when_recording_state_is_live(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    run = tmp_path / ".voicenotes" / "run"
    run.mkdir(parents=True)
    (run / "current-recording.json").write_text('{"pid": 123}', encoding="utf-8")
    monkeypatch.setattr("voicenotes.cli.has_live_recording", lambda paths: True)
    monkeypatch.setattr("voicenotes.cli.stop_recording", lambda paths: paths.output_root / "2026-08-27_143012")

    assert main(["toggle"]) == 0
    assert capsys.readouterr().out.strip().endswith("2026-08-27_143012")


def test_toggle_starts_when_recording_state_is_stale(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    run = tmp_path / ".voicenotes" / "run"
    run.mkdir(parents=True)
    (run / "current-recording.json").write_text('{"pid": 123}', encoding="utf-8")
    monkeypatch.setattr("voicenotes.cli.has_live_recording", lambda paths: False)
    monkeypatch.setattr("voicenotes.cli.start_recording", lambda config, paths: paths.output_root / "2026-08-27_143045")
    monkeypatch.setattr("voicenotes.cli.load_config", lambda: object())

    assert main(["toggle"]) == 0
    assert capsys.readouterr().out.strip().endswith("2026-08-27_143045")


def test_toggle_starts_when_recording_state_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("voicenotes.cli.start_recording", lambda config, paths: paths.output_root / "2026-08-27_143012")
    monkeypatch.setattr("voicenotes.cli.load_config", lambda: object())

    assert main(["toggle"]) == 0
    assert capsys.readouterr().out.strip().endswith("2026-08-27_143012")


def test_process_command_runs_pipeline_for_session(tmp_path, monkeypatch, capsys):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    processed = []
    monkeypatch.setattr("voicenotes.cli.load_config", lambda: object())
    run = tmp_path / "run"
    (run / "queue").mkdir(parents=True)
    worker_paths = type("WorkerPaths", (), {"run": run})()
    monkeypatch.setattr("voicenotes.cli.default_paths", lambda: worker_paths)
    monkeypatch.setattr("voicenotes.cli.acquire_pipeline_lock", lambda paths: True)
    released = []
    monkeypatch.setattr("voicenotes.cli.release_pipeline_lock", lambda paths: released.append(paths))
    monkeypatch.setattr("voicenotes.cli.try_spawn_worker", lambda paths: False)
    monkeypatch.setattr("voicenotes.cli.process_session", lambda path, config, paths: processed.append((path, config, paths)))

    assert main(["process", str(session)]) == 0
    assert processed[0][0] == session
    assert released == [worker_paths]
    assert capsys.readouterr().out.strip() == str(session)


def test_retry_command_returns_one_for_pipeline_failure(tmp_path, monkeypatch, capsys):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    monkeypatch.setattr("voicenotes.cli.load_config", lambda: object())
    run = tmp_path / "run"
    (run / "queue").mkdir(parents=True)
    worker_paths = type("WorkerPaths", (), {"run": run})()
    monkeypatch.setattr("voicenotes.cli.default_paths", lambda: worker_paths)
    monkeypatch.setattr("voicenotes.cli.acquire_pipeline_lock", lambda paths: True)
    monkeypatch.setattr("voicenotes.cli.release_pipeline_lock", lambda paths: None)
    monkeypatch.setattr("voicenotes.cli.try_spawn_worker", lambda paths: False)
    monkeypatch.setattr("voicenotes.cli.retry_session", lambda path, config, paths: (_ for _ in ()).throw(RuntimeError("pipeline failed")))

    assert main(["retry", str(session)]) == 1
    assert "pipeline failed" in capsys.readouterr().err


def test_manual_pipeline_command_returns_one_when_worker_is_active(tmp_path, monkeypatch, capsys):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    monkeypatch.setattr("voicenotes.cli.default_paths", lambda: object())
    monkeypatch.setattr("voicenotes.cli.acquire_pipeline_lock", lambda paths: False)
    monkeypatch.setattr("voicenotes.cli.process_session", lambda *args: (_ for _ in ()).throw(AssertionError("must not run")))

    assert main(["process", str(session)]) == 1
    assert "pipeline is already active" in capsys.readouterr().err


def test_manual_pipeline_command_spawns_worker_for_items_queued_while_locked(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    run = tmp_path / "run"
    queue = run / "queue"
    queue.mkdir(parents=True)
    worker_paths = type("WorkerPaths", (), {"run": run})()
    spawned = []

    monkeypatch.setattr("voicenotes.cli.load_config", lambda: object())
    monkeypatch.setattr("voicenotes.cli.default_paths", lambda: worker_paths)
    monkeypatch.setattr("voicenotes.cli.acquire_pipeline_lock", lambda paths: True)
    monkeypatch.setattr("voicenotes.cli.release_pipeline_lock", lambda paths: None)
    monkeypatch.setattr("voicenotes.cli.try_spawn_worker", lambda paths: spawned.append(paths) or True)

    def process(path, config, paths):
        (queue / "queued.json").write_text('{"session_path": "/tmp/queued"}', encoding="utf-8")

    monkeypatch.setattr("voicenotes.cli.process_session", process)

    assert main(["process", str(session)]) == 0
    assert spawned == [worker_paths]
