from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


@dataclass(frozen=True)
class Paths:
    app: Path
    run: Path
    config: Path
    models: Path
    output_root: Path


@dataclass(frozen=True)
class AppConfig:
    output_root: Path
    hotkey_mods: list[str]
    hotkey_key: str
    audio_device: str
    ollama_model: str
    auto_open: bool


def default_paths(home: Path | None = None) -> Paths:
    root = home or Path(os.environ["HOME"])
    base = root / ".voicenotes"
    return Paths(
        app=base / "app",
        run=base / "run",
        config=base / "config.toml",
        models=base / "models",
        output_root=root / "VoiceNotes",
    )


def load_config(path: Path | None = None) -> AppConfig:
    paths = default_paths()
    config_path = path or paths.config
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig(
        output_root=Path(raw["output_root"]).expanduser(),
        hotkey_mods=list(raw["hotkey"]["mods"]),
        hotkey_key=str(raw["hotkey"]["key"]),
        audio_device=str(raw["audio_device"]),
        ollama_model=str(raw["ollama_model"]),
        auto_open=bool(raw["auto_open"]),
    )


def config_as_dict(config: AppConfig) -> dict[str, object]:
    return {
        "output_root": str(config.output_root),
        "hotkey": {"mods": config.hotkey_mods, "key": config.hotkey_key},
        "audio_device": config.audio_device,
        "ollama_model": config.ollama_model,
        "auto_open": config.auto_open,
    }
