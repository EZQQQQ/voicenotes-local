from __future__ import annotations

from pathlib import Path
import platform
import shutil

import voicenotes.ollama as ollama
import voicenotes.recorder as recorder

from .config import AppConfig, Paths
from .pipeline import whisper_model_dir


def _check(label: str, check) -> bool:
    try:
        check()
    except Exception as error:
        print(f"FAIL {label}: {error}")
        return False
    print(f"PASS {label}")
    return True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_doctor(config: AppConfig, paths: Paths) -> int:
    checks = [
        _check("Apple Silicon", lambda: _require(platform.machine() == "arm64", "Apple Silicon is required")),
        _check("brew", lambda: _require(shutil.which("brew") is not None, "brew not found")),
        _check("ffmpeg", lambda: _require(shutil.which("ffmpeg") is not None, "ffmpeg not found")),
        _check("ollama", lambda: _require(shutil.which("ollama") is not None, "ollama not found")),
        _check("git", lambda: _require(shutil.which("git") is not None, "git not found")),
        _check("python3", lambda: _require(shutil.which("python3") is not None, "python3 not found")),
        _check("Whisper model", lambda: _require(whisper_model_dir(paths).exists(), f"missing {whisper_model_dir(paths)}")),
        _check("Ollama model", lambda: ollama.ensure_model_available(config.ollama_model)),
    ]

    hammerspoon = Path.home() / ".hammerspoon"
    checks.append(_check("Hammerspoon module", lambda: _require((hammerspoon / "voicenotes.lua").exists(), "voicenotes.lua not found")))

    def hammerspoon_init() -> None:
        content = (hammerspoon / "init.lua").read_text(encoding="utf-8")
        _require(content.count('require("voicenotes")') == 1, 'init.lua must contain exactly one require("voicenotes")')

    checks.append(_check("Hammerspoon init", hammerspoon_init))
    checks.append(_check("Audio devices", lambda: _require(bool(recorder.list_audio_devices()), "no audio devices found")))

    try:
        recorder.record_test(config, paths, duration_seconds=2)
    except Exception as error:
        print(f"FAIL Record test: {error}")
        print("Check System Settings > Privacy & Security > Microphone.")
        checks.append(False)
    else:
        print("PASS Record test")
        checks.append(True)

    return 0 if all(checks) else 1
