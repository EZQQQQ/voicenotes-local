from pathlib import Path


def test_hammerspoon_config_uses_cli_for_config_status_and_toggle():
    lua = Path("hammerspoon/voicenotes.lua").read_text(encoding="utf-8")

    assert "voicenotes config --json" in lua
    assert "voicenotes status --json" in lua
    assert "voicenotes toggle" in lua
    assert "menubar:setMenu" in lua
    assert "Start Recording" in lua
    assert "Stop Recording" in lua
    assert "Open VoiceNotes Folder" in lua
    assert "Quit" in lua
    assert "hotkey:disable()" in lua
    assert 'hs.application.get("Hammerspoon")' in lua
    assert ":kill()" in lua
    assert "hs.pathwatcher.new" in lua
    assert "mkdir -p ~/.voicenotes/run" in lua
    assert "~/.voicenotes/run" in lua
    assert "toml" not in lua.lower()


def test_hammerspoon_config_has_no_recording_or_ffmpeg_logic():
    lua = Path("hammerspoon/voicenotes.lua").read_text(encoding="utf-8")

    assert "ffmpeg" not in lua
    assert "avfoundation" not in lua
    assert "current-recording.json" not in lua
