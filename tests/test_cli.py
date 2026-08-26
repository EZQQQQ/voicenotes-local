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


def test_toggle_stops_when_recording_state_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    run = tmp_path / ".voicenotes" / "run"
    run.mkdir(parents=True)
    (run / "current-recording.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("voicenotes.cli.stop_recording", lambda paths: paths.output_root / "2026-08-27_143012")

    assert main(["toggle"]) == 0
    assert capsys.readouterr().out.strip().endswith("2026-08-27_143012")


def test_toggle_starts_when_recording_state_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("voicenotes.cli.start_recording", lambda config, paths: paths.output_root / "2026-08-27_143012")
    monkeypatch.setattr("voicenotes.cli.load_config", lambda: object())

    assert main(["toggle"]) == 0
    assert capsys.readouterr().out.strip().endswith("2026-08-27_143012")
