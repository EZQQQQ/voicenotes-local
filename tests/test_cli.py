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
