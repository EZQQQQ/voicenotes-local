from pathlib import Path

from voicenotes.config import AppConfig, config_as_dict, default_paths, load_config


def test_default_paths_use_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = default_paths()

    assert paths.app == tmp_path / ".voicenotes" / "app"
    assert paths.run == tmp_path / ".voicenotes" / "run"
    assert paths.config == tmp_path / ".voicenotes" / "config.toml"
    assert paths.models == tmp_path / ".voicenotes" / "models"
    assert paths.output_root == tmp_path / "VoiceNotes"


def test_load_config_reads_exact_five_user_fields(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_root = "~/VoiceNotes"',
                'audio_device = "default"',
                'ollama_model = "qwen2.5:14b"',
                "auto_open = true",
                "",
                "[hotkey]",
                'mods = ["cmd"]',
                'key = "`"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == AppConfig(
        output_root=Path("~/VoiceNotes").expanduser(),
        hotkey_mods=["cmd"],
        hotkey_key="`",
        audio_device="default",
        ollama_model="qwen2.5:14b",
        auto_open=True,
    )


def test_config_as_dict_is_hammerspoon_friendly(tmp_path):
    config = AppConfig(
        output_root=tmp_path / "VoiceNotes",
        hotkey_mods=["cmd"],
        hotkey_key="`",
        audio_device="default",
        ollama_model="qwen2.5:14b",
        auto_open=True,
    )

    assert config_as_dict(config) == {
        "output_root": str(tmp_path / "VoiceNotes"),
        "hotkey": {"mods": ["cmd"], "key": "`"},
        "audio_device": "default",
        "ollama_model": "qwen2.5:14b",
        "auto_open": True,
    }
