from __future__ import annotations

from pathlib import Path
import importlib
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


def _require_runtime_dependencies() -> None:
    importlib.import_module("huggingface_hub")
    importlib.import_module("mlx_whisper")


def _require_homebrew_python_311() -> None:
    python = shutil.which("python3.11")
    _require(python is not None, "python3.11 not found")
    resolved = Path(python).resolve()
    _require(str(resolved).startswith("/opt/homebrew/"), f"Homebrew python3.11 required, found {resolved}")


def _require_model_files(paths: Paths) -> None:
    model_dir = whisper_model_dir(paths)
    has_config = (model_dir / "config.json").is_file()
    has_weights = any(model_dir.glob("*.safetensors"))
    _require(model_dir.is_dir() and has_config and has_weights, f"incomplete or missing {model_dir}")


def run_doctor(config: AppConfig, paths: Paths) -> int:
    checks = [
        _check("Apple Silicon", lambda: _require(platform.machine() == "arm64", "Apple Silicon is required")),
        _check("brew", lambda: _require(shutil.which("brew") is not None, "brew not found")),
        _check("ffmpeg", lambda: _require(shutil.which("ffmpeg") is not None, "ffmpeg not found")),
        _check("ollama", lambda: _require(shutil.which("ollama") is not None, "ollama not found")),
        _check("git", lambda: _require(shutil.which("git") is not None, "git not found")),
        _check("Homebrew python3.11", _require_homebrew_python_311),
        _check("Python dependencies", _require_runtime_dependencies),
        _check("Whisper model", lambda: _require_model_files(paths)),
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
