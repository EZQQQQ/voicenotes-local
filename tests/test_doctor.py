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
    (p.models / "whisper-large-v3-mlx" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".hammerspoon").mkdir()
    (tmp_path / ".hammerspoon" / "voicenotes.lua").write_text("-- ok", encoding="utf-8")
    (tmp_path / ".hammerspoon" / "init.lua").write_text('require("voicenotes")\n', encoding="utf-8")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("shutil.which", lambda command: f"/opt/homebrew/bin/{command}")
    monkeypatch.setattr("voicenotes.doctor.importlib.import_module", lambda name: object())
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])
    monkeypatch.setattr("voicenotes.recorder.record_test", lambda config, paths, duration_seconds=2: tmp_path / "VoiceNotes" / "record-test")

    assert run_doctor(config(tmp_path), p) == 0
    assert "PASS Apple Silicon" in capsys.readouterr().out


def test_doctor_reports_tcc_hint_when_record_test_fails(tmp_path, monkeypatch, capsys):
    p = paths(tmp_path)
    p.models.mkdir(parents=True)
    (p.models / "whisper-large-v3-mlx").mkdir()
    (p.models / "whisper-large-v3-mlx" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("shutil.which", lambda command: f"/opt/homebrew/bin/{command}")
    monkeypatch.setattr("voicenotes.doctor.importlib.import_module", lambda name: object())
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.recorder.list_audio_devices", lambda: ["MacBook Pro Microphone"])
    monkeypatch.setattr("voicenotes.recorder.record_test", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("microphone denied")))

    assert run_doctor(config(tmp_path), p) == 1
    assert "Privacy & Security > Microphone" in capsys.readouterr().out
