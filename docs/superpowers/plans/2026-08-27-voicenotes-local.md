# VoiceNotes Local Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 local macOS VoiceNotes tool: hotkey-triggered recording, local transcription, local cleanup and summarization, Markdown outputs, retry, diagnostics, installer, and uninstall.

**Architecture:** Hammerspoon is a thin controller that binds the hotkey, displays state, and calls the CLI. Python owns config parsing, recording state, ffmpeg device resolution and recording, queue locking, pipeline execution, validation, retry, and diagnostics. Runtime state is file-based under `~/.voicenotes/run`; session artifacts are flat folders under `~/VoiceNotes`.

**Tech Stack:** Python 3.11+, stdlib `argparse`, `tomllib`, `json`, `subprocess`, `urllib.request`; ffmpeg avfoundation; mlx-whisper; huggingface_hub; Ollama HTTP API; Hammerspoon Lua; shell scripts for install and uninstall; pytest for tests.

**Spec:** `docs/superpowers/specs/2026-08-27-voicenotes-local-design.md`

## Global Constraints

- macOS on Apple Silicon only; Intel Macs fail early during install with a clear Apple Silicon requirement.
- Default paths: app source `~/.voicenotes/app`, runtime/state `~/.voicenotes/run`, config `~/.voicenotes/config.toml`, models `~/.voicenotes/models`, output sessions `~/VoiceNotes`.
- Session names use second precision: `YYYY-MM-DD_HHMMSS`.
- Hammerspoon writes/loads `~/.hammerspoon/voicenotes.lua` and exactly one `require("voicenotes")` line in `~/.hammerspoon/init.lua`.
- Hammerspoon never parses TOML and never implements recording, device resolution, pipeline state, or queue logic.
- User config exposes exactly `output_root`, `hotkey`, `audio_device`, `ollama_model`, and `auto_open`.
- Whisper model is fixed to `mlx-community/whisper-large-v3-mlx`; it is not user-configurable.
- Ollama defaults to `qwen2.5:14b`, temperature `0.2`, and keep-alive `30s`.
- Whisper and Ollama must never run concurrently; pipeline execution is serialized with an atomic lock.
- All generated artifacts and JSON state files are written atomically with temp-file-plus-rename.
- Failures preserve completed artifacts, write `error.log`, and notify the user.
- Retry resumes from the first missing or invalid artifact and skips valid existing artifacts.
- Summary validation requires exactly the five required Markdown sections in order, allowing only the provenance comment before the first heading.
- Successful summaries open with `open -g` when `auto_open` is true.
- No real voice recordings are committed; local fixtures are gitignored.
- V1 does not include a daemon, GUI beyond Hammerspoon menu bar and notifications, prompt customization, Whisper model customization, active network sandboxing, or automatic summary repair.

---

## File Structure

- `requirements.txt`: pinned Python dependencies for runtime and tests.
- `config.example.toml`: example user config with the exact five supported fields.
- `.gitignore`: ignores local fixtures, caches, virtualenvs, and runtime outputs.
- `voicenotes/__main__.py`: `python -m voicenotes` entrypoint.
- `voicenotes/cli.py`: argparse command routing, exit-code mapping, user-facing stderr.
- `voicenotes/config.py`: path constants, config load/write/defaults, JSON config output.
- `voicenotes/state.py`: atomic writes, JSON state helpers, session metadata, artifact validation, status calculation, notifications.
- `voicenotes/recorder.py`: ffmpeg device enumeration, exact device resolution, PID identity checks, start/stop/record-test, WAV repair.
- `voicenotes/queue.py`: atomic pipeline lock, queue item writes, worker spawning, queue draining.
- `voicenotes/ollama.py`: Ollama API preflight and non-streaming generate call with timeout.
- `voicenotes/pipeline.py`: transcription, cleanup, summarization, validation, retry/process orchestration.
- `voicenotes/doctor.py`: environment and permission diagnostics.
- `hammerspoon/voicenotes.lua`: hotkey, menu-bar state, path watcher, CLI invocation.
- `install.sh`: idempotent installer.
- `uninstall.sh`: conservative uninstaller.
- `scripts/smoke-test.sh`: local smoke test using `voicenotes record-test`.
- `README.md`: focused user docs.
- `LICENSE`: MIT license.
- `tests/`: pytest suite using temp homes and mocked subprocess/network/model calls.

---

### Task 1: Python Foundation, Config, State, And CLI Skeleton

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `config.example.toml`
- Create: `voicenotes/__init__.py`
- Create: `voicenotes/__main__.py`
- Create: `voicenotes/cli.py`
- Create: `voicenotes/config.py`
- Create: `voicenotes/state.py`
- Create: `tests/test_config.py`
- Create: `tests/test_state.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `voicenotes.config.AppConfig`, `load_config() -> AppConfig`, `default_paths(home: Path | None = None) -> Paths`, `config_as_dict(config: AppConfig) -> dict[str, object]`.
- Produces: `voicenotes.state.atomic_write_text(path: Path, text: str) -> None`, `atomic_write_json(path: Path, payload: dict[str, object]) -> None`, `read_json(path: Path) -> dict[str, object]`.
- Produces: `voicenotes.state.required_summary_headings() -> list[str]`, `validate_summary(path: Path) -> tuple[bool, str]`, `status_snapshot(paths: Paths) -> dict[str, object]`.
- Produces: CLI commands `config --json` and `status --json`.
- Later tasks consume the config, path, atomic write, summary validation, and CLI dispatch interfaces.

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Write failing state and summary-validation tests**

Create `tests/test_state.py`:

```python
import json

from voicenotes.config import Paths
from voicenotes.state import (
    atomic_write_json,
    atomic_write_text,
    required_summary_headings,
    status_snapshot,
    validate_summary,
)


def test_atomic_write_text_replaces_file(tmp_path):
    path = tmp_path / "artifact.md"
    atomic_write_text(path, "first")
    atomic_write_text(path, "second")

    assert path.read_text(encoding="utf-8") == "second"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_json_round_trips(tmp_path):
    path = tmp_path / "session.json"
    atomic_write_json(path, {"status": "ready", "count": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ready", "count": 1}


def test_validate_summary_accepts_provenance_comment_and_exact_headings(tmp_path):
    path = tmp_path / "summary.md"
    path.write_text(
        "\n".join(
            [
                "<!-- Generated by VoiceNotes from session 2026-08-27_143012 -->",
                "",
                "## Meeting Metadata",
                "Date: not specified",
                "",
                "## Key Discussion Points",
                "- point",
                "",
                "## Decisions Made",
                "- not specified",
                "",
                "## Action Items",
                "- [ ] task — owner: unassigned — deadline: no deadline given",
                "",
                "## Open Questions",
                "- none",
            ]
        ),
        encoding="utf-8",
    )

    assert required_summary_headings() == [
        "## Meeting Metadata",
        "## Key Discussion Points",
        "## Decisions Made",
        "## Action Items",
        "## Open Questions",
    ]
    assert validate_summary(path) == (True, "ok")


def test_validate_summary_rejects_extra_top_level_section(tmp_path):
    path = tmp_path / "summary.md"
    path.write_text(
        "\n".join(
            [
                "## Intro",
                "extra",
                "## Meeting Metadata",
                "## Key Discussion Points",
                "## Decisions Made",
                "## Action Items",
                "## Open Questions",
            ]
        ),
        encoding="utf-8",
    )

    valid, reason = validate_summary(path)

    assert valid is False
    assert "unexpected heading" in reason


def test_status_snapshot_reports_recording_processing_queue_and_error(tmp_path):
    run = tmp_path / ".voicenotes" / "run"
    queue = run / "queue"
    queue.mkdir(parents=True)
    (run / "current-recording.json").write_text('{"session_path": "/tmp/session"}', encoding="utf-8")
    (run / "pipeline.lock").write_text('{"pid": 123}', encoding="utf-8")
    (queue / "20260827T143012_session.json").write_text('{"session_path": "/tmp/queued"}', encoding="utf-8")
    (run / "last-error.txt").write_text("failed", encoding="utf-8")

    snapshot = status_snapshot(Paths(tmp_path / ".voicenotes/app", run, run.parent / "config.toml", run.parent / "models", tmp_path / "VoiceNotes"))

    assert snapshot == {
        "recording": True,
        "processing": True,
        "queued_count": 1,
        "last_error": "failed",
        "active_session": "/tmp/session",
        "state_label": "recording",
    }
```

- [ ] **Step 3: Write failing CLI skeleton tests**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 4: Run foundation tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_config.py tests/test_state.py tests/test_cli.py -v
```

Expected: FAIL because the `voicenotes` package and functions do not exist.

- [ ] **Step 5: Implement requirements, config example, gitignore, config, state, and CLI skeleton**

Create `requirements.txt` with pinned dependencies:

```text
huggingface_hub[hf_xet]==1.8.0
mlx-whisper==0.4.3
pytest==8.4.2
```

Create `.gitignore`:

```gitignore
__pycache__/
.pytest_cache/
.venv/
venv/
fixtures/
*.pyc
```

Create `config.example.toml`:

```toml
output_root = "~/VoiceNotes"
audio_device = "default"
ollama_model = "qwen2.5:14b"
auto_open = true

[hotkey]
mods = ["cmd"]
key = "`"
```

Implement `voicenotes/config.py` with frozen dataclasses:

```python
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
```

Implement `voicenotes/state.py` with atomic writes, summary validation, and status snapshot:

```python
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

from .config import Paths


SUMMARY_HEADINGS = [
    "## Meeting Metadata",
    "## Key Discussion Points",
    "## Decisions Made",
    "## Action Items",
    "## Open Questions",
]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def required_summary_headings() -> list[str]:
    return list(SUMMARY_HEADINGS)


def validate_summary(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing summary.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    headings = [line.strip() for line in lines if line.startswith("## ")]
    if headings != SUMMARY_HEADINGS:
        unexpected = [heading for heading in headings if heading not in SUMMARY_HEADINGS]
        if unexpected:
            return False, f"unexpected heading: {unexpected[0]}"
        return False, "required headings missing or out of order"
    first_heading_index = next((i for i, line in enumerate(lines) if line.startswith("## ")), -1)
    prefix = "\n".join(lines[:first_heading_index]).strip()
    if prefix and (not prefix.startswith("<!-- Generated by VoiceNotes from session ") or not prefix.endswith("-->")):
        return False, "unexpected content before first heading"
    return True, "ok"


def status_snapshot(paths: Paths) -> dict[str, object]:
    recording_path = paths.run / "current-recording.json"
    lock_path = paths.run / "pipeline.lock"
    queue_path = paths.run / "queue"
    error_path = paths.run / "last-error.txt"
    active_session = None
    if recording_path.exists():
        try:
            active_session = str(read_json(recording_path).get("session_path") or "")
        except (OSError, json.JSONDecodeError):
            active_session = ""
    queued_count = len(list(queue_path.glob("*.json"))) if queue_path.exists() else 0
    last_error = error_path.read_text(encoding="utf-8").strip() if error_path.exists() else None
    recording = recording_path.exists()
    processing = lock_path.exists()
    state_label = "recording" if recording else "processing" if processing else "queued" if queued_count else "error" if last_error else "idle"
    return {
        "recording": recording,
        "processing": processing,
        "queued_count": queued_count,
        "last_error": last_error,
        "active_session": active_session,
        "state_label": state_label,
    }
```

Implement `voicenotes/cli.py` with `argparse`, `main(argv: list[str] | None = None) -> int`, and command handlers for `config --json` and `status --json`. Unknown arguments return `2` without raising.

Implement `voicenotes/__main__.py`:

```python
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 6: Run foundation tests to verify they pass**

Run:

```bash
python3.11 -m pytest tests/test_config.py tests/test_state.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add requirements.txt .gitignore config.example.toml voicenotes tests
git commit -m "Build VoiceNotes Python foundation"
```

---

### Task 2: Device Resolution, Recording Start, And Record Test

**Files:**
- Create: `voicenotes/recorder.py`
- Create: `tests/test_recorder_start.py`
- Modify: `voicenotes/cli.py`
- Modify: `voicenotes/state.py`

**Interfaces:**
- Consumes: `AppConfig`, `Paths`, `atomic_write_json`, `atomic_write_text`, `read_json`.
- Produces: `list_audio_devices(ffmpeg: str = "ffmpeg") -> list[str]`.
- Produces: `resolve_audio_device(configured: str, devices: list[str]) -> int`.
- Produces: `create_session_dir(output_root: Path, now: datetime | None = None) -> Path`.
- Produces: `start_recording(config: AppConfig, paths: Paths) -> Path`.
- Produces: `record_test(config: AppConfig, paths: Paths, duration_seconds: int = 10) -> Path`.
- CLI exposes `devices`, `start`, and `record-test`.
- Later tasks consume the active recording state file and ffmpeg process identity.

- [ ] **Step 1: Write failing recorder start tests**

Create `tests/test_recorder_start.py`:

```python
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from voicenotes.config import AppConfig, Paths
from voicenotes.recorder import create_session_dir, list_audio_devices, record_test, resolve_audio_device, start_recording
from voicenotes.state import read_json


def config(tmp_path, audio_device="default"):
    return AppConfig(
        output_root=tmp_path / "VoiceNotes",
        hotkey_mods=["cmd"],
        hotkey_key="`",
        audio_device=audio_device,
        ollama_model="qwen2.5:14b",
        auto_open=True,
    )


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_create_session_dir_uses_second_precision(tmp_path):
    session = create_session_dir(tmp_path, datetime(2026, 8, 27, 14, 30, 12))

    assert session == tmp_path / "2026-08-27_143012"
    assert session.is_dir()


def test_list_audio_devices_parses_avfoundation_stderr(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stderr="\n".join(
                [
                    "[AVFoundation indev @ 0x1] AVFoundation audio devices:",
                    "[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone",
                    "[AVFoundation indev @ 0x1] [1] External Mic",
                ]
            )
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    assert list_audio_devices() == ["MacBook Pro Microphone", "External Mic"]


def test_resolve_audio_device_default_is_index_zero():
    assert resolve_audio_device("default", ["Built-in", "USB"]) == 0


def test_resolve_audio_device_requires_exact_name():
    assert resolve_audio_device("USB Mic", ["Built-in", "USB Mic"]) == 1
    with pytest.raises(ValueError, match="Configured audio device not found"):
        resolve_audio_device("USB", ["Built-in", "USB Mic"])


def test_start_recording_writes_state_and_ffmpeg_log(tmp_path, monkeypatch):
    launched = {}

    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])
    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: False)
    monkeypatch.setattr("voicenotes.recorder.create_session_dir", lambda output_root, now=None: output_root / "2026-08-27_143012")

    class FakePopen:
        pid = 4321

        def __init__(self, args, stderr, stdout, stdin):
            launched["args"] = args
            launched["stderr_name"] = Path(stderr.name).name

    monkeypatch.setattr("subprocess.Popen", FakePopen)

    session = start_recording(config(tmp_path), paths(tmp_path))

    assert session == tmp_path / "VoiceNotes" / "2026-08-27_143012"
    assert launched["args"][-1] == str(session / "audio.wav")
    assert launched["stderr_name"] == "ffmpeg.log"
    state = read_json(tmp_path / "run" / "current-recording.json")
    assert state["session_path"] == str(session)
    assert state["pid"] == 4321
    assert state["resolved_device"] == "MacBook Pro Microphone"


def test_record_test_uses_shared_device_resolution_without_state_file(tmp_path, monkeypatch):
    launched = {}
    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])

    def fake_run(args, check):
        launched["args"] = args
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    session = record_test(config(tmp_path), paths(tmp_path), duration_seconds=3)

    assert "-t" in launched["args"]
    assert "3" in launched["args"]
    assert not (tmp_path / "run" / "current-recording.json").exists()
    assert session.name.startswith("record-test_")
```

- [ ] **Step 2: Run recorder start tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_recorder_start.py -v
```

Expected: FAIL because `voicenotes.recorder` and CLI commands do not exist.

- [ ] **Step 3: Implement device parsing, session creation, start, and record-test**

Implement `voicenotes/recorder.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import re
import subprocess

from .config import AppConfig, Paths
from .state import atomic_write_json, read_json


DEVICE_RE = re.compile(r"\[(\d+)\]\s+(.+)$")


def list_audio_devices(ffmpeg: str = "ffmpeg") -> list[str]:
    result = subprocess.run(
        [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        check=False,
    )
    devices: list[str] = []
    in_audio = False
    for line in result.stderr.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if in_audio and "AVFoundation video devices:" in line:
            break
        if in_audio:
            match = DEVICE_RE.search(line)
            if match:
                devices.append(match.group(2).strip())
    return devices


def resolve_audio_device(configured: str, devices: list[str]) -> int:
    if configured == "default":
        if not devices:
            raise ValueError("No avfoundation audio devices found")
        return 0
    for index, name in enumerate(devices):
        if name == configured:
            return index
    available = "\n".join(f"- {name}" for name in devices)
    raise ValueError(f"Configured audio device not found: {configured}\nAvailable devices:\n{available}")


def create_session_dir(output_root: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    session = output_root / stamp
    session.mkdir(parents=True, exist_ok=False)
    return session


def is_live_ffmpeg(pid: int) -> bool:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True, check=False)
    return result.returncode == 0 and Path(result.stdout.strip()).name == "ffmpeg"


def _active_recording_path(paths: Paths) -> Path:
    return paths.run / "current-recording.json"


def _clear_stale_recording_state(paths: Paths) -> None:
    state_path = _active_recording_path(paths)
    if state_path.exists():
        state_path.unlink()


def start_recording(config: AppConfig, paths: Paths) -> Path:
    state_path = _active_recording_path(paths)
    if state_path.exists():
        state = read_json(state_path)
        pid = int(state.get("pid", 0))
        if pid and is_live_ffmpeg(pid):
            raise RuntimeError(f"Recording already active: {state.get('session_path')}")
        _clear_stale_recording_state(paths)
    devices = list_audio_devices()
    device_index = resolve_audio_device(config.audio_device, devices)
    resolved_name = devices[device_index]
    session = create_session_dir(config.output_root)
    ffmpeg_log = session / "ffmpeg.log"
    log_handle = ffmpeg_log.open("ab")
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{device_index}", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(session / "audio.wav")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=log_handle,
    )
    atomic_write_json(
        state_path,
        {
            "session_path": str(session),
            "pid": process.pid,
            "resolved_device": resolved_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ffmpeg_log_path": str(ffmpeg_log),
        },
    )
    return session


def record_test(config: AppConfig, paths: Paths, duration_seconds: int = 10) -> Path:
    devices = list_audio_devices()
    device_index = resolve_audio_device(config.audio_device, devices)
    session = config.output_root / f"record-test_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    session.mkdir(parents=True, exist_ok=False)
    with (session / "ffmpeg.log").open("ab") as log_handle:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{device_index}", "-t", str(duration_seconds), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(session / "audio.wav")],
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
            check=True,
        )
    return session
```

Modify `voicenotes/cli.py` to add `devices`, `start`, and `record-test`. `devices` prints one device per line. `start` prints the session path. `record-test` accepts `--duration`, defaults to `10`, and prints the session path.

- [ ] **Step 4: Run recorder start tests to verify they pass**

Run:

```bash
python3.11 -m pytest tests/test_recorder_start.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add voicenotes/recorder.py voicenotes/cli.py voicenotes/state.py tests/test_recorder_start.py tests/test_cli.py
git commit -m "Add recording start and device resolution"
```

---

### Task 3: Recording Stop, WAV Repair, Queue Enqueue, And Toggle

**Files:**
- Create: `voicenotes/queue.py`
- Create: `tests/test_recorder_stop.py`
- Create: `tests/test_queue.py`
- Modify: `voicenotes/recorder.py`
- Modify: `voicenotes/cli.py`
- Modify: `voicenotes/state.py`

**Interfaces:**
- Consumes: `is_live_ffmpeg(pid: int) -> bool`, active recording state, atomic JSON writes.
- Produces: `stop_recording(paths: Paths) -> Path`.
- Produces: `repair_wav(session: Path) -> bool`.
- Produces: `enqueue_session(paths: Paths, session: Path) -> Path`.
- Produces: `try_spawn_worker(paths: Paths) -> bool`.
- CLI exposes `stop` and `toggle`.
- Later tasks consume queue JSON files and `drain_queue`.

- [ ] **Step 1: Write failing stop tests**

Create `tests/test_recorder_stop.py`:

```python
import signal
from pathlib import Path

import pytest

from voicenotes.config import Paths
from voicenotes.recorder import stop_recording
from voicenotes.state import atomic_write_json, read_json


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_stop_sends_sigint_clears_state_and_enqueues(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 4096)
    atomic_write_json(p.run / "current-recording.json", {"session_path": str(session), "pid": 555, "resolved_device": "Mic"})
    sent = []
    enqueued = []

    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: True)
    monkeypatch.setattr("os.kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr("voicenotes.recorder._wait_pid_exit", lambda pid, timeout: True)
    monkeypatch.setattr("voicenotes.queue.enqueue_session", lambda paths, session_path: enqueued.append(session_path) or (p.run / "queue" / "item.json"))
    monkeypatch.setattr("voicenotes.queue.try_spawn_worker", lambda paths: True)

    stopped = stop_recording(p)

    assert stopped == session
    assert sent == [(555, signal.SIGINT)]
    assert enqueued == [session]
    assert not (p.run / "current-recording.json").exists()


def test_stop_escalates_and_repairs_after_sigkill(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 4096)
    atomic_write_json(p.run / "current-recording.json", {"session_path": str(session), "pid": 777, "resolved_device": "Mic"})
    sent = []
    waits = iter([False, False])

    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: True)
    monkeypatch.setattr("os.kill", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr("voicenotes.recorder._wait_pid_exit", lambda pid, timeout: next(waits, True))
    monkeypatch.setattr("voicenotes.recorder.repair_wav", lambda session_path: True)
    monkeypatch.setattr("voicenotes.queue.enqueue_session", lambda paths, session_path: p.run / "queue" / "item.json")
    monkeypatch.setattr("voicenotes.queue.try_spawn_worker", lambda paths: True)

    stop_recording(p)

    assert sent == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    session_state = read_json(session / "session.json")
    assert session_state["recording_interrupted"] is True
    assert session_state["wav_repair_succeeded"] is True


def test_stop_refuses_recycled_pid(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    atomic_write_json(p.run / "current-recording.json", {"session_path": str(session), "pid": 888})
    monkeypatch.setattr("voicenotes.recorder.is_live_ffmpeg", lambda pid: False)

    with pytest.raises(RuntimeError, match="stale recording state"):
        stop_recording(p)
```

- [ ] **Step 2: Write failing queue tests**

Create `tests/test_queue.py`:

```python
import json
from pathlib import Path

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
```

- [ ] **Step 3: Run stop and queue tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_recorder_stop.py tests/test_queue.py -v
```

Expected: FAIL because stop and queue functions do not exist.

- [ ] **Step 4: Implement queue enqueue, atomic lock, stop, repair, and toggle**

Implement `voicenotes/queue.py` with:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import errno
import json
import os
import subprocess
import sys

from .config import Paths
from .state import atomic_write_json, read_json


def enqueue_session(paths: Paths, session: Path) -> Path:
    queue_dir = paths.run / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    item = queue_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{session.name}.json"
    atomic_write_json(item, {"session_path": str(session), "enqueued_at": datetime.now().isoformat(timespec="seconds")})
    return item


def _lock_path(paths: Paths) -> Path:
    return paths.run / "pipeline.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_pipeline_lock(paths: Paths) -> bool:
    paths.run.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(paths)
    if lock.exists():
        try:
            payload = read_json(lock)
            if _pid_alive(int(payload.get("pid", 0))):
                return False
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")}, handle)
    return True


def release_pipeline_lock(paths: Paths) -> None:
    _lock_path(paths).unlink(missing_ok=True)


def try_spawn_worker(paths: Paths) -> bool:
    if _lock_path(paths).exists():
        return False
    worker_log = paths.run / "worker.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    handle = worker_log.open("ab")
    subprocess.Popen([sys.executable, "-m", "voicenotes", "worker"], stdin=subprocess.DEVNULL, stdout=handle, stderr=handle)
    return True
```

Modify `voicenotes/recorder.py` to implement:

- `_wait_pid_exit(pid: int, timeout: float) -> bool`.
- `repair_wav(session: Path) -> bool`, moving `audio.wav` to `audio.interrupted.wav`, running `ffmpeg -y -i audio.interrupted.wav -c copy audio.wav`, and returning success.
- `stop_recording(paths: Paths) -> Path`, with SIGINT, SIGTERM, SIGKILL, repair, atomic `session.json`, state cleanup, `enqueue_session`, and `try_spawn_worker`.

Modify `voicenotes/cli.py`:

- `stop` prints the stopped session path.
- `toggle` calls `stop_recording` when `current-recording.json` exists, else `start_recording`.
- `toggle` prints the session path and returns `0` on success.

- [ ] **Step 5: Run stop and queue tests to verify they pass**

Run:

```bash
python3.11 -m pytest tests/test_recorder_stop.py tests/test_queue.py tests/test_recorder_start.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add voicenotes/queue.py voicenotes/recorder.py voicenotes/cli.py voicenotes/state.py tests/test_recorder_stop.py tests/test_queue.py tests/test_cli.py
git commit -m "Add recording stop and queue enqueue"
```

---

### Task 4: Ollama Client, Pipeline, Artifact Validation, And Retry

**Files:**
- Create: `voicenotes/ollama.py`
- Create: `voicenotes/pipeline.py`
- Create: `tests/test_ollama.py`
- Create: `tests/test_pipeline.py`
- Modify: `voicenotes/cli.py`
- Modify: `voicenotes/state.py`

**Interfaces:**
- Consumes: config, state atomic writes, summary validation, queue item session paths.
- Produces: `ollama.generate(model: str, prompt: str, timeout_seconds: int = 1800) -> str`.
- Produces: `ollama.ensure_model_available(model: str) -> None`.
- Produces: `pipeline.process_session(session: Path, config: AppConfig, paths: Paths) -> None`.
- Produces: `pipeline.retry_session(session: Path, config: AppConfig, paths: Paths) -> None`.
- Produces: `pipeline.artifact_status(session: Path) -> dict[str, bool]`.
- CLI exposes `process <session>` and `retry <session>`.
- Later queue worker consumes `process_session`.

- [ ] **Step 1: Write failing Ollama tests**

Create `tests/test_ollama.py`:

```python
import json
from urllib.error import URLError

import pytest

from voicenotes.ollama import ensure_model_available, generate


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ensure_model_available_accepts_configured_model(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=10: FakeResponse({"models": [{"name": "qwen2.5:14b"}]}))

    ensure_model_available("qwen2.5:14b")


def test_ensure_model_available_fails_with_pull_instruction(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=10: FakeResponse({"models": []}))

    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:14b"):
        ensure_model_available("qwen2.5:14b")


def test_generate_posts_non_streaming_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"response": "clean transcript"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = generate("qwen2.5:14b", "Prompt text", timeout_seconds=1800)

    assert response == "clean transcript"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"] == {
        "model": "qwen2.5:14b",
        "prompt": "Prompt text",
        "stream": False,
        "options": {"temperature": 0.2},
        "keep_alive": "30s",
    }
    assert captured["timeout"] == 1800
```

- [ ] **Step 2: Write failing pipeline tests**

Create `tests/test_pipeline.py`:

```python
from pathlib import Path

import pytest

from voicenotes.config import AppConfig, Paths
from voicenotes.pipeline import artifact_status, format_timestamp, process_session, retry_session


def config(tmp_path):
    return AppConfig(
        output_root=tmp_path / "VoiceNotes",
        hotkey_mods=["cmd"],
        hotkey_key="`",
        audio_device="default",
        ollama_model="qwen2.5:14b",
        auto_open=False,
    )


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_format_timestamp_uses_hh_mm_ss():
    assert format_timestamp(3.2) == "00:00:03"
    assert format_timestamp(3661.9) == "01:01:01"


def test_process_session_writes_all_artifacts(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)

    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        "voicenotes.pipeline.transcribe_audio",
        lambda audio, models: [
            {"start": 3.0, "end": 9.0, "text": "We discussed roadmap and 中文部分."},
        ],
    )
    responses = iter(
        [
            "[00:00:03 - 00:00:09] We discussed roadmap and 中文部分.",
            "\n".join(
                [
                    "## Meeting Metadata",
                    "Date: not specified",
                    "",
                    "## Key Discussion Points",
                    "- roadmap and 中文部分",
                    "",
                    "## Decisions Made",
                    "- not specified",
                    "",
                    "## Action Items",
                    "- [ ] follow up — owner: unassigned — deadline: no deadline given",
                    "",
                    "## Open Questions",
                    "- not specified",
                ]
            ),
        ]
    )
    monkeypatch.setattr("voicenotes.ollama.generate", lambda model, prompt, timeout_seconds=1800: next(responses))
    opened = []
    monkeypatch.setattr("subprocess.run", lambda args, check=False: opened.append(args))

    process_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == "[00:00:03 - 00:00:09] We discussed roadmap and 中文部分.\n"
    assert (session / "transcript_clean.md").read_text(encoding="utf-8").startswith("[00:00:03 - 00:00:09]")
    summary = (session / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("<!-- Generated by VoiceNotes from session 2026-08-27_143012 -->")
    assert "## Open Questions" in summary


def test_invalid_summary_is_saved_raw_and_fails(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda audio, models: [{"start": 0, "end": 1, "text": "hello"}])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: "Here is the summary\n\n## Meeting Metadata")

    with pytest.raises(RuntimeError, match="summary validation failed"):
        process_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "summary.raw.md").exists()
    assert (session / "error.log").exists()


def test_retry_skips_valid_existing_artifacts(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    (session / "transcript_raw.md").write_text("[00:00:00 - 00:00:01] existing raw\n", encoding="utf-8")
    (session / "transcript_clean.md").write_text("[00:00:00 - 00:00:01] existing clean\n", encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda audio, models: pytest.fail("should skip transcription"))
    monkeypatch.setattr(
        "voicenotes.ollama.generate",
        lambda *args, **kwargs: "\n".join(
            [
                "## Meeting Metadata",
                "Date: not specified",
                "## Key Discussion Points",
                "- point",
                "## Decisions Made",
                "- none",
                "## Action Items",
                "- [ ] task — owner: unassigned — deadline: no deadline given",
                "## Open Questions",
                "- none",
            ]
        ),
    )

    retry_session(session, config(tmp_path), paths(tmp_path))

    assert artifact_status(session)["transcript_raw"] is True
    assert (session / "summary.md").exists()
```

- [ ] **Step 3: Run Ollama and pipeline tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_ollama.py tests/test_pipeline.py -v
```

Expected: FAIL because `ollama.py` and `pipeline.py` do not exist.

- [ ] **Step 4: Implement Ollama client**

Create `voicenotes/ollama.py`:

```python
from __future__ import annotations

import json
from urllib.request import Request, urlopen

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.2
OLLAMA_KEEP_ALIVE = "30s"
OLLAMA_TIMEOUT_SECONDS = 1800


def ensure_model_available(model: str) -> None:
    with urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names = {item.get("name") for item in payload.get("models", [])}
    if model not in names:
        raise RuntimeError(f"Ollama model missing. Run: ollama pull {model}")


def generate(model: str, prompt: str, timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    request = Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", ""))
```

- [ ] **Step 5: Implement pipeline**

Create `voicenotes/pipeline.py` with:

- Constants `WHISPER_REPO_ID = "mlx-community/whisper-large-v3-mlx"` and `PROMPT_VERSION = "2026-08-27-v1"`.
- Exact cleanup and summary prompts from the spec.
- `format_timestamp(seconds: float) -> str`.
- `transcribe_audio(audio: Path, models: Path) -> list[dict[str, object]]`, importing `mlx_whisper` inside the function and calling:

```python
mlx_whisper.transcribe(
    str(audio),
    path_or_hf_repo=str(models / "whisper-large-v3-mlx"),
    task="transcribe",
    language=None,
    initial_prompt="This recording mixes English and Mandarin Chinese, sometimes switching mid-sentence.",
)
```

- `write_raw_transcript(session: Path, segments: list[dict[str, object]]) -> None`, using `[HH:MM:SS - HH:MM:SS] text` and one blank line between segments.
- `artifact_status(session: Path) -> dict[str, bool]`, validating audio by size floor, transcript files by stripped length floor, and summary by `validate_summary`.
- `process_session(session, config, paths)` running all steps, writing `pipeline.log`, `error.log`, `session.json`, `summary.raw.md` on invalid summary, and opening with `subprocess.run(["open", "-g", str(summary_path)], check=False)` only when `config.auto_open` is true.
- `retry_session(session, config, paths)` as an alias to the same resumable processing logic.

The implementation must import `voicenotes.ollama` as a module, not `from voicenotes.ollama import generate`, so tests can monkeypatch `voicenotes.ollama.generate`.

- [ ] **Step 6: Add CLI process and retry commands**

Modify `voicenotes/cli.py` so:

```bash
voicenotes process ~/VoiceNotes/2026-08-27_143012
voicenotes retry ~/VoiceNotes/2026-08-27_143012
```

load config and paths, call `process_session` or `retry_session`, print the session path on success, return `1` for expected `RuntimeError`, and return `2` for unexpected exceptions.

- [ ] **Step 7: Run Ollama and pipeline tests to verify they pass**

Run:

```bash
python3.11 -m pytest tests/test_ollama.py tests/test_pipeline.py tests/test_state.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add voicenotes/ollama.py voicenotes/pipeline.py voicenotes/cli.py voicenotes/state.py tests/test_ollama.py tests/test_pipeline.py tests/test_state.py tests/test_cli.py
git commit -m "Add local processing pipeline"
```

---

### Task 5: Queue Worker, Status Hardening, And Notifications

**Files:**
- Create: `tests/test_worker.py`
- Modify: `voicenotes/queue.py`
- Modify: `voicenotes/cli.py`
- Modify: `voicenotes/state.py`
- Modify: `voicenotes/recorder.py`
- Modify: `voicenotes/pipeline.py`

**Interfaces:**
- Consumes: queue JSON item format, `process_session`, config load, status snapshot.
- Produces: `drain_queue(config: AppConfig, paths: Paths) -> None`.
- Produces: `notify(title: str, message: str) -> None`.
- Produces: `play_start_sound() -> None`.
- CLI exposes internal `worker` command used by queue spawning.
- Hammerspoon consumes compact status JSON.

- [ ] **Step 1: Write failing worker and notification tests**

Create `tests/test_worker.py`:

```python
from pathlib import Path

from voicenotes.config import AppConfig, Paths
from voicenotes.queue import drain_queue, enqueue_session
from voicenotes.state import notify, play_start_sound


def config(tmp_path):
    return AppConfig(tmp_path / "VoiceNotes", ["cmd"], "`", "default", "qwen2.5:14b", True)


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_drain_queue_processes_items_sequentially_and_removes_them(tmp_path, monkeypatch):
    p = paths(tmp_path)
    first = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    second = tmp_path / "VoiceNotes" / "2026-08-27_143045"
    first.mkdir(parents=True)
    second.mkdir()
    enqueue_session(p, first)
    enqueue_session(p, second)
    processed = []

    monkeypatch.setattr("voicenotes.pipeline.process_session", lambda session, cfg, paths: processed.append(session))

    drain_queue(config(tmp_path), p)

    assert processed == [first, second]
    assert list((p.run / "queue").glob("*.json")) == []


def test_drain_queue_releases_lock_after_failure(tmp_path, monkeypatch):
    p = paths(tmp_path)
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    enqueue_session(p, session)

    def fail(*args):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr("voicenotes.pipeline.process_session", fail)

    drain_queue(config(tmp_path), p)

    assert not (p.run / "pipeline.lock").exists()
    assert (session / "error.log").exists()


def test_notify_uses_osascript(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda args, check=False: calls.append(args))

    notify("Note ready", "session")

    assert calls[0][:3] == ["osascript", "-e", 'display notification "session" with title "Note ready"']


def test_play_start_sound_uses_afplay(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda args, check=False: calls.append(args))

    play_start_sound()

    assert calls[0][0] == "afplay"
```

- [ ] **Step 2: Run worker tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_worker.py -v
```

Expected: FAIL because worker draining and notification helpers do not exist.

- [ ] **Step 3: Implement notifications and start sound**

Modify `voicenotes/state.py`:

```python
import shlex
import subprocess


def notify(title: str, message: str) -> None:
    import json

    script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
    subprocess.run(["osascript", "-e", script], check=False)


def play_start_sound() -> None:
    subprocess.run(["afplay", "/System/Library/Sounds/Pop.aiff"], check=False)
```

Use double-quote escaping instead of `repr` if the test string is adjusted during implementation; the behavior must use built-in macOS tools and never add `terminal-notifier`.

Modify `recorder.start_recording` to call `play_start_sound()` after state is written. Modify `stop_recording` to call `notify("Recording stopped", "Session queued for processing")` after enqueue.

- [ ] **Step 4: Implement queue draining and worker command**

Modify `voicenotes/queue.py`:

- `drain_queue(config, paths)` acquires the atomic pipeline lock, processes queue JSON files in sorted filename order, removes each item only after `process_session` succeeds, writes worker events to `worker.log`, writes session `error.log` and leaves the failed item in place when a pipeline fails, releases the lock in `finally`, then exits.
- If `acquire_pipeline_lock` returns false, `drain_queue` returns immediately.

Modify `voicenotes/cli.py`:

- Add hidden/internal command `worker`.
- `worker` loads config and paths, calls `drain_queue`, and returns `0`.

- [ ] **Step 5: Run worker tests and full unit suite**

Run:

```bash
python3.11 -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add voicenotes/queue.py voicenotes/cli.py voicenotes/state.py voicenotes/recorder.py voicenotes/pipeline.py tests/test_worker.py
git commit -m "Add queue worker and notifications"
```

---

### Task 6: Doctor Command And Model Download Helper

**Files:**
- Create: `voicenotes/doctor.py`
- Create: `tests/test_doctor.py`
- Modify: `voicenotes/cli.py`
- Modify: `voicenotes/pipeline.py`

**Interfaces:**
- Consumes: paths, config, `list_audio_devices`, `record_test`, `ollama.ensure_model_available`, `WHISPER_REPO_ID`.
- Produces: `run_doctor(config: AppConfig, paths: Paths) -> int`.
- Produces: `download_whisper_model(paths: Paths) -> Path`.
- CLI exposes `doctor`.
- Installer consumes `python -m voicenotes doctor` and `python -m voicenotes download-whisper-model`.

- [ ] **Step 1: Write failing doctor tests**

Create `tests/test_doctor.py`:

```python
from pathlib import Path
from types import SimpleNamespace

from voicenotes.config import AppConfig, Paths
from voicenotes.doctor import run_doctor
from voicenotes.pipeline import download_whisper_model


def config(tmp_path):
    return AppConfig(tmp_path / "VoiceNotes", ["cmd"], "`", "default", "qwen2.5:14b", True)


def paths(tmp_path):
    return Paths(tmp_path / "app", tmp_path / "run", tmp_path / "config.toml", tmp_path / "models", tmp_path / "VoiceNotes")


def test_download_whisper_model_uses_pinned_repo_and_local_dir(tmp_path, monkeypatch):
    captured = {}

    def fake_snapshot_download(repo_id, local_dir):
        captured["repo_id"] = repo_id
        captured["local_dir"] = local_dir
        Path(local_dir).mkdir(parents=True)
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    model_dir = download_whisper_model(paths(tmp_path))

    assert captured["repo_id"] == "mlx-community/whisper-large-v3-mlx"
    assert model_dir == tmp_path / "models" / "whisper-large-v3-mlx"


def test_doctor_returns_success_when_checks_pass(tmp_path, monkeypatch, capsys):
    p = paths(tmp_path)
    p.models.mkdir(parents=True)
    (p.models / "whisper-large-v3-mlx").mkdir()
    (tmp_path / ".hammerspoon").mkdir()
    (tmp_path / ".hammerspoon" / "voicenotes.lua").write_text("-- ok", encoding="utf-8")
    (tmp_path / ".hammerspoon" / "init.lua").write_text('require("voicenotes")\n', encoding="utf-8")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("shutil.which", lambda command: f"/opt/homebrew/bin/{command}")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])
    monkeypatch.setattr("voicenotes.recorder.record_test", lambda config, paths, duration_seconds=2: tmp_path / "VoiceNotes" / "record-test")

    assert run_doctor(config(tmp_path), p) == 0
    assert "PASS Apple Silicon" in capsys.readouterr().out


def test_doctor_reports_tcc_hint_when_record_test_fails(tmp_path, monkeypatch, capsys):
    p = paths(tmp_path)
    p.models.mkdir(parents=True)
    (p.models / "whisper-large-v3-mlx").mkdir()
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("shutil.which", lambda command: f"/opt/homebrew/bin/{command}")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])
    monkeypatch.setattr("voicenotes.recorder.record_test", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("microphone denied")))

    assert run_doctor(config(tmp_path), p) == 1
    assert "Privacy & Security > Microphone" in capsys.readouterr().out
```

- [ ] **Step 2: Run doctor tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_doctor.py -v
```

Expected: FAIL because doctor and download helper do not exist.

- [ ] **Step 3: Implement Whisper download helper**

Modify `voicenotes/pipeline.py`:

```python
def whisper_model_dir(paths: Paths) -> Path:
    return paths.models / "whisper-large-v3-mlx"


def download_whisper_model(paths: Paths) -> Path:
    from huggingface_hub import snapshot_download

    target = whisper_model_dir(paths)
    snapshot_download(repo_id=WHISPER_REPO_ID, local_dir=target)
    return target
```

Update `transcribe_audio` to use `whisper_model_dir(paths)` through its existing models path behavior.

- [ ] **Step 4: Implement doctor**

Create `voicenotes/doctor.py`:

- Print one line per check with `PASS` or `FAIL`.
- Check `platform.machine() == "arm64"`.
- Check `shutil.which` for `brew`, `ffmpeg`, `ollama`, `git`, and `python3`.
- Check `paths.models / "whisper-large-v3-mlx"` exists.
- Check Ollama model with `ollama.ensure_model_available(config.ollama_model)`.
- Check `~/.hammerspoon/voicenotes.lua` exists.
- Check `~/.hammerspoon/init.lua` contains exactly one `require("voicenotes")`.
- Check `list_audio_devices()` returns at least one device.
- Run `record_test(config, paths, duration_seconds=2)` and print the TCC hint on failure.
- Return `0` when all checks pass, else `1`.

Modify `voicenotes/cli.py`:

- Add `doctor`.
- Add hidden/internal `download-whisper-model` for installer use.

- [ ] **Step 5: Run doctor tests and full unit suite**

Run:

```bash
python3.11 -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add voicenotes/doctor.py voicenotes/pipeline.py voicenotes/cli.py tests/test_doctor.py
git commit -m "Add diagnostics and model download helper"
```

---

### Task 7: Hammerspoon Integration

**Files:**
- Create: `hammerspoon/voicenotes.lua`
- Create: `tests/test_hammerspoon.py`

**Interfaces:**
- Consumes: CLI commands `voicenotes config --json`, `voicenotes status --json`, and `voicenotes toggle`.
- Produces: Hammerspoon module loaded by `require("voicenotes")`.
- Installer copies this file to `~/.hammerspoon/voicenotes.lua`.

- [ ] **Step 1: Write static Hammerspoon tests**

Create `tests/test_hammerspoon.py`:

```python
from pathlib import Path


def test_hammerspoon_config_uses_cli_for_config_status_and_toggle():
    lua = Path("hammerspoon/voicenotes.lua").read_text(encoding="utf-8")

    assert "voicenotes config --json" in lua
    assert "voicenotes status --json" in lua
    assert "voicenotes toggle" in lua
    assert "hs.pathwatcher.new" in lua
    assert "mkdir -p ~/.voicenotes/run" in lua
    assert "~/.voicenotes/run" in lua
    assert "toml" not in lua.lower()


def test_hammerspoon_config_has_no_recording_or_ffmpeg_logic():
    lua = Path("hammerspoon/voicenotes.lua").read_text(encoding="utf-8")

    assert "ffmpeg" not in lua
    assert "avfoundation" not in lua
    assert "current-recording.json" not in lua
```

- [ ] **Step 2: Run Hammerspoon tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_hammerspoon.py -v
```

Expected: FAIL because `hammerspoon/voicenotes.lua` does not exist.

- [ ] **Step 3: Implement Hammerspoon module**

Create `hammerspoon/voicenotes.lua`:

```lua
local M = {}

local menubar = hs.menubar.new()
local watcher = nil

local function run(command)
  local output, status = hs.execute(command, true)
  if status then
    return output
  end
  return nil
end

local function json(command)
  local output = run(command)
  if not output then
    return nil
  end
  return hs.json.decode(output)
end

local function setTitle(status)
  if not menubar then
    return
  end
  local label = "VN"
  if status and status.state_label == "recording" then
    label = "VN REC"
  elseif status and status.state_label == "processing" then
    label = "VN RUN"
  elseif status and status.state_label == "queued" then
    label = "VN Q"
  elseif status and status.state_label == "error" then
    label = "VN !"
  end
  menubar:setTitle(label)
end

local function refresh()
  setTitle(json("voicenotes status --json"))
end

function M.toggle()
  hs.task.new("/bin/zsh", function()
    refresh()
  end, {"-lc", "voicenotes toggle"}):start()
end

function M.start()
  hs.execute("mkdir -p ~/.voicenotes/run", true)
  local config = json("voicenotes config --json")
  local mods = {"cmd"}
  local key = "`"
  if config and config.hotkey then
    mods = config.hotkey.mods or mods
    key = config.hotkey.key or key
  end
  hs.hotkey.bind(mods, key, M.toggle)
  refresh()
  watcher = hs.pathwatcher.new(os.getenv("HOME") .. "/.voicenotes/run", refresh)
  watcher:start()
end

M.start()

return M
```

- [ ] **Step 4: Run Hammerspoon tests and full unit suite**

Run:

```bash
python3.11 -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```bash
git add hammerspoon/voicenotes.lua tests/test_hammerspoon.py
git commit -m "Add Hammerspoon hotkey integration"
```

---

### Task 8: Installer And Uninstaller

**Files:**
- Create: `install.sh`
- Create: `uninstall.sh`
- Create: `tests/test_install_scripts.py`
- Modify: `config.example.toml`

**Interfaces:**
- Consumes: repo layout, `config.example.toml`, `hammerspoon/voicenotes.lua`, CLI `doctor`, CLI `download-whisper-model`.
- Produces: idempotent install into `~/.voicenotes/app`, local venv, wrapper, Hammerspoon require line, model downloads.
- Produces: conservative uninstall that leaves Homebrew deps, Ollama model, and `~/VoiceNotes`.

- [ ] **Step 1: Write static installer tests**

Create `tests/test_install_scripts.py`:

```python
from pathlib import Path


def test_install_script_has_required_idempotent_behaviors():
    script = Path("install.sh").read_text(encoding="utf-8")

    assert "uname -m" in script
    assert "arm64" in script
    assert "VOICENOTES_REPO_URL" in script
    assert "$HOME/.voicenotes/app" in script
    assert "$HOME/.voicenotes/venv" in script
    assert "$CONFIG_DIR/run" in script
    assert "python3.11 -m venv" in script
    assert "pip install -r requirements.txt" in script
    assert "ollama pull" in script
    assert "download-whisper-model" in script
    assert 'require("voicenotes")' in script
    assert "hammerspoon://reload" in script
    assert "NONINTERACTIVE" in script


def test_uninstall_script_leaves_user_data_and_shared_deps():
    script = Path("uninstall.sh").read_text(encoding="utf-8")

    assert "$HOME/VoiceNotes" in script
    assert "not remove" in script
    assert "ollama rm qwen2.5:14b" in script
    assert "brew uninstall" not in script
    assert "ollama rm qwen2.5:14b" in script
    assert 'require("voicenotes")' in script
```

- [ ] **Step 2: Run installer tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_install_scripts.py -v
```

Expected: FAIL because `install.sh` and `uninstall.sh` do not exist.

- [ ] **Step 3: Implement install script**

Create `install.sh` with these concrete behaviors:

```bash
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/.voicenotes/app"
VENV_DIR="$HOME/.voicenotes/venv"
CONFIG_DIR="$HOME/.voicenotes"
CONFIG_FILE="$CONFIG_DIR/config.toml"
REPO_URL="${VOICENOTES_REPO_URL:-https://github.com/ezqqqq/voicenotes-local.git}"
BREW_PREFIX="/opt/homebrew"
WRAPPER_DIR="$BREW_PREFIX/bin"
WRAPPER="$WRAPPER_DIR/voicenotes"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "VoiceNotes Local v1 requires Apple Silicon (arm64)." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

brew install git ffmpeg python@3.11
brew install --cask hammerspoon ollama

mkdir -p "$CONFIG_DIR" "$CONFIG_DIR/run" "$CONFIG_DIR/models"
if [[ "$REPO_URL" == /* ]]; then
  rm -rf "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
elif [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only
else
  rm -rf "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
fi

"$BREW_PREFIX/bin/python3.11" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$CONFIG_FILE" ]]; then
  cp "$APP_DIR/config.example.toml" "$CONFIG_FILE"
fi

if [[ ! -w "$WRAPPER_DIR" ]]; then
  WRAPPER_DIR="$HOME/.local/bin"
  WRAPPER="$WRAPPER_DIR/voicenotes"
  mkdir -p "$WRAPPER_DIR"
fi

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" -m voicenotes "\$@"
EOF
chmod +x "$WRAPPER"

if [[ "${NONINTERACTIVE:-}" != "1" ]]; then
  echo "VoiceNotes will download local models. Combined download size is approximately 12GB."
  read -r -p "Continue? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled before model downloads."; exit 1 ;;
  esac
fi

ollama pull qwen2.5:14b
"$WRAPPER" download-whisper-model

mkdir -p "$HOME/.hammerspoon"
cp "$APP_DIR/hammerspoon/voicenotes.lua" "$HOME/.hammerspoon/voicenotes.lua"
touch "$HOME/.hammerspoon/init.lua"
if ! grep -Fxq 'require("voicenotes")' "$HOME/.hammerspoon/init.lua"; then
  cp "$HOME/.hammerspoon/init.lua" "$HOME/.hammerspoon/init.lua.$(date +%Y%m%d%H%M%S).bak"
  printf '\nrequire("voicenotes")\n' >> "$HOME/.hammerspoon/init.lua"
fi

open -g hammerspoon://reload || true

echo "If Cmd+\` conflicts on this Mac, clear macOS Move focus to next window in System Settings > Keyboard > Keyboard Shortcuts > Keyboard."
echo "Grant permissions in System Settings > Privacy & Security > Microphone and Accessibility."
"$WRAPPER" doctor || true
```

Implementation must preserve the behavior above. If `shellcheck` is available, quoting fixes are allowed only when they do not change behavior.

- [ ] **Step 4: Implement uninstall script**

Create `uninstall.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

rm -f "$HOME/.hammerspoon/voicenotes.lua"

if [[ -f "$HOME/.hammerspoon/init.lua" ]]; then
  tmp="$(mktemp)"
  grep -Fxv 'require("voicenotes")' "$HOME/.hammerspoon/init.lua" > "$tmp" || true
  mv "$tmp" "$HOME/.hammerspoon/init.lua"
fi

rm -f /opt/homebrew/bin/voicenotes
rm -f /usr/local/bin/voicenotes
rm -f "$HOME/.local/bin/voicenotes"
rm -rf "$HOME/.voicenotes/run"
rm -rf "$HOME/.voicenotes/app"
rm -rf "$HOME/.voicenotes/venv"

echo "Uninstalled VoiceNotes Local app files."
echo "Did not remove Homebrew dependencies, Ollama models, or $HOME/VoiceNotes."
echo "To remove the default Ollama model and reclaim roughly 9GB, run: ollama rm qwen2.5:14b"
```

- [ ] **Step 5: Run installer tests and shell parse checks**

Run:

```bash
python3.11 -m pytest tests/test_install_scripts.py -v
bash -n install.sh
bash -n uninstall.sh
```

Expected: PASS.

- [ ] **Step 6: Commit Task 8**

```bash
git add install.sh uninstall.sh config.example.toml tests/test_install_scripts.py
git commit -m "Add installer and uninstaller"
```

---

### Task 9: README, License, Smoke Test, And Final Verification

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `scripts/smoke-test.sh`
- Create: `tests/test_docs.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: all implemented commands and installer behavior.
- Produces: user-facing documentation, MIT license, smoke-test script.
- Final branch verification runs the full suite and shell syntax checks.

- [ ] **Step 1: Write static docs and smoke-test tests**

Create `tests/test_docs.py`:

```python
from pathlib import Path


def test_readme_covers_required_user_topics():
    readme = Path("README.md").read_text(encoding="utf-8")

    for text in [
        "curl -fsSL https://raw.githubusercontent.com/ezqqqq/voicenotes-local/main/install.sh | bash",
        "Microphone",
        "Accessibility",
        "Cmd+`",
        "output_root",
        "audio_device",
        "ollama_model",
        "auto_open",
        "voicenotes retry",
        "voicenotes doctor",
        "uninstall.sh",
        "Whisper model is fixed",
        "Ollama model is configurable",
    ]:
        assert text in readme


def test_license_is_mit():
    license_text = Path("LICENSE").read_text(encoding="utf-8")

    assert "MIT License" in license_text
    assert "Permission is hereby granted" in license_text


def test_smoke_test_uses_record_test_and_validates_summary_headings():
    script = Path("scripts/smoke-test.sh").read_text(encoding="utf-8")

    assert "voicenotes record-test" in script
    assert "voicenotes process" in script
    assert "## Meeting Metadata" in script
    assert "## Key Discussion Points" in script
    assert "## Decisions Made" in script
    assert "## Action Items" in script
    assert "## Open Questions" in script
```

- [ ] **Step 2: Run docs tests to verify they fail**

Run:

```bash
python3.11 -m pytest tests/test_docs.py -v
```

Expected: FAIL because README, LICENSE, and smoke test do not exist.

- [ ] **Step 3: Write README**

Create `README.md` with these sections:

- `# VoiceNotes Local`
- `What it does`: one paragraph explaining hotkey recording, local transcription, local cleanup, local Markdown summary.
- `Requirements`: Apple Silicon Mac, 16GB+, Homebrew, Hammerspoon, ffmpeg, Ollama, Python 3.11+.
- `Install` with:

```bash
curl -fsSL https://raw.githubusercontent.com/ezqqqq/voicenotes-local/main/install.sh | bash
```

- `Permissions`: first launch Hammerspoon, grant Accessibility to Hammerspoon, grant Microphone to Hammerspoon, grant Microphone to Terminal for `record-test`.
- `Hotkey`: default `Cmd+\``, macOS conflict path `System Settings > Keyboard > Keyboard Shortcuts > Keyboard > Move focus to next window`, and Hyper-key alternative.
- `Config`: document only `output_root`, `hotkey`, `audio_device`, `ollama_model`, and `auto_open`.
- `Why Whisper is fixed and Ollama is configurable`: Whisper `large-v3` is fixed for code-switching accuracy; Ollama model is configurable for different Mac memory sizes.
- `Commands`: `voicenotes devices`, `voicenotes status --json`, `voicenotes doctor`, `voicenotes retry <session>`, `voicenotes record-test`.
- `Output`: session folder layout.
- `Uninstall`: run `./uninstall.sh`; explain what remains.
- `Privacy`: no cloud inference after model downloads; no active network sandboxing in v1.

- [ ] **Step 4: Write MIT license**

Create `LICENSE` using this MIT license text:

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Write smoke test script**

Create `scripts/smoke-test.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SESSION="$(voicenotes record-test --duration 10 | tail -n 1)"
voicenotes process "$SESSION"

for file in audio.wav transcript_raw.md transcript_clean.md summary.md; do
  test -s "$SESSION/$file"
done

for heading in \
  "## Meeting Metadata" \
  "## Key Discussion Points" \
  "## Decisions Made" \
  "## Action Items" \
  "## Open Questions"; do
  grep -Fxq "$heading" "$SESSION/summary.md"
done

echo "Smoke test passed: $SESSION"
```

Mark it executable.

Modify `.gitignore` to include:

```gitignore
VoiceNotes/
*.wav
*.m4a
*.mp3
```

- [ ] **Step 6: Run docs tests and full verification**

Run:

```bash
python3.11 -m pytest tests -v
bash -n install.sh
bash -n uninstall.sh
bash -n scripts/smoke-test.sh
```

Expected: PASS.

- [ ] **Step 7: Commit Task 9**

```bash
git add README.md LICENSE scripts/smoke-test.sh .gitignore tests/test_docs.py
git commit -m "Add user documentation and smoke test"
```

---

## Final Verification

After all tasks are complete, run:

```bash
python3.11 -m pytest tests -v
bash -n install.sh
bash -n uninstall.sh
bash -n scripts/smoke-test.sh
git status --short
```

Expected:

- pytest passes.
- all shell scripts parse.
- `git status --short` is clean.

Do not run the real `scripts/smoke-test.sh` automatically during implementation unless the user has already granted the needed macOS microphone permissions and explicitly wants a live local recording test. The script exists for manual acceptance testing.
