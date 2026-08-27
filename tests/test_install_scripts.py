from pathlib import Path


def test_install_script_has_required_idempotent_behaviors():
    script = Path("install.sh").read_text(encoding="utf-8")

    assert "uname -m" in script
    assert "arm64" in script
    assert "VOICENOTES_REPO_URL" in script
    assert "$HOME/.voicenotes/app" in script
    assert "$HOME/.voicenotes/venv" in script
    assert "$CONFIG_DIR/run" in script
    assert '"$BREW_PREFIX/bin/python3.11" -m venv "$VENV_DIR"' in script
    assert '"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"' in script
    assert "ollama pull" in script
    assert "download-whisper-model" in script
    assert 'require("voicenotes")' in script
    assert "hammerspoon://reload" in script
    assert "NONINTERACTIVE" in script
    assert "</dev/tty" in script
    assert "NONINTERACTIVE=1" in script
    assert '"$WRAPPER" config --json' in script
    assert 'ollama pull "$OLLAMA_MODEL"' in script


def test_install_script_uses_explicit_homebrew_binary_after_bootstrap():
    script = Path("install.sh").read_text(encoding="utf-8")

    assert 'BREW="$BREW_PREFIX/bin/brew"' in script
    assert '"$BREW" install git ffmpeg python@3.11' in script
    assert '"$BREW" install --cask hammerspoon ollama' in script


def test_uninstall_script_leaves_user_data_and_shared_deps():
    script = Path("uninstall.sh").read_text(encoding="utf-8")

    assert "$HOME/VoiceNotes" in script
    assert "not remove" in script
    assert "ollama rm qwen2.5:14b" in script
    assert "brew uninstall" not in script
    assert "ollama rm qwen2.5:14b" in script
    assert 'require("voicenotes")' in script
