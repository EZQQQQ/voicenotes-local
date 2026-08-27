from pathlib import Path


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


def test_smoke_test_uses_record_test_and_validates_summary_headings():
    script = Path("scripts/smoke-test.sh").read_text(encoding="utf-8")

    assert "voicenotes record-test" in script
    assert "voicenotes process" in script
    assert "## Meeting Metadata" in script
    assert "## Key Discussion Points" in script
    assert "## Decisions Made" in script
    assert "## Action Items" in script
    assert "## Open Questions" in script
