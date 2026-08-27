from pathlib import Path
import subprocess


def test_readme_covers_required_user_topics():
    readme = Path("README.md").read_text(encoding="utf-8")

    for text in [
        "curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/voicenotes-local/main/install.sh | bash",
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


def test_gitignore_protects_session_artifacts_without_ignoring_package():
    for artifact in [
        "VoiceNotes/2026-08-27_143012/transcript_raw.md",
        "VoiceNotes/2026-08-27_143012/transcript_clean.md",
        "VoiceNotes/2026-08-27_143012/summary.md",
        "VoiceNotes/2026-08-27_143012/summary.raw.md",
        "VoiceNotes/2026-08-27_143012/session.json",
        "VoiceNotes/2026-08-27_143012/pipeline.log",
        "VoiceNotes/2026-08-27_143012/error.log",
        "VoiceNotes/2026-08-27_143012/ffmpeg.log",
    ]:
        assert subprocess.run(["git", "check-ignore", "--no-index", artifact], check=False).returncode == 0

    assert subprocess.run(["git", "check-ignore", "--no-index", "voicenotes/cli.py"], check=False).returncode == 1


def test_smoke_test_uses_record_test_and_validates_summary_headings():
    script = Path("scripts/smoke-test.sh").read_text(encoding="utf-8")

    assert "voicenotes record-test" in script
    assert "voicenotes process" in script
    assert "## Meeting Metadata" in script
    assert "## Key Discussion Points" in script
    assert "## Decisions Made" in script
    assert "## Action Items" in script
    assert "## Open Questions" in script
