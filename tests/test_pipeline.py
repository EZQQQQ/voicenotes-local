import json
import sys
from pathlib import Path
from dataclasses import replace

import pytest

from voicenotes.config import AppConfig, Paths
from voicenotes.pipeline import CLEANUP_PROMPT, PROMPT_VERSION, SUMMARY_PROMPT, _merge_summaries, artifact_status, format_timestamp, process_session, retry_session, transcribe_audio


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


def summary_body(bullet: str) -> str:
    return "\n".join(
        [
            "## Summary",
            f"- {bullet}",
            "## Discussion by topic",
            "### Topic",
            f"- {bullet}",
            "## Feedback & critique",
            f"- {bullet}",
            "## Decisions",
            f"- {bullet}",
            "## Action items",
            f"- [ ] {bullet}",
            "## Blockers & open questions",
            "### Blockers",
            f"- {bullet}",
            "### Open questions",
            f"- {bullet}",
            "## Next steps",
            f"- {bullet}",
        ]
    )


def test_retry_resumes_summary_chunks_and_from_clean_discards_checkpoints(tmp_path, monkeypatch):
    from voicenotes import pipeline
    session = tmp_path / "session"
    session.mkdir()
    audio = b"RIFF" + b"0" * 10000
    raw = "[00:00:00 - 00:00:01] first sentence\n\n[00:00:01 - 00:00:02] last sentence\n"
    (session / "audio.wav").write_bytes(audio)
    (session / "transcript_raw.md").write_text(raw)
    monkeypatch.setattr(pipeline, "SUMMARY_CHUNK_MAX_TOKENS", 4)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr(pipeline, "notify", lambda *args: None)
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *args: pytest.fail("must preserve raw"))
    responses = iter([raw, summary_body("first sentence"), RuntimeError("interrupted")])

    def response(*args, **kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("voicenotes.ollama.generate", response)
    with pytest.raises(RuntimeError, match="interrupted"):
        process_session(session, config(tmp_path), paths(tmp_path))
    assert not (session / "summary.md").exists()
    responses = iter([summary_body("last sentence")])
    retry_session(session, config(tmp_path), paths(tmp_path))
    summary = (session / "summary.md").read_text()
    assert "first sentence" in summary and "last sentence" in summary
    assert (session / "audio.wav").read_bytes() == audio
    assert (session / "transcript_raw.md").read_text() == raw
    assert "interrupted" in (session / "pipeline.log").read_text()
    # Explicit regeneration must not silently return the old chunk cache.
    responses = iter([raw.replace("first", "fresh"), summary_body("fresh sentence"), summary_body("last sentence")])
    retry_session(session, config(tmp_path), paths(tmp_path), from_clean=True)
    assert "fresh sentence" in (session / "transcript_clean.md").read_text()


def test_format_timestamp_uses_hh_mm_ss():
    assert format_timestamp(3.2) == "00:00:03"
    assert format_timestamp(3661.9) == "01:01:01"


def test_transcribe_audio_runs_whisper_in_child_process(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + b"0" * 10000)
    captured = {}

    def fake_run(args, capture_output, text, check):
        captured["args"] = args
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check

        class Result:
            stdout = json.dumps([{"start": 0.0, "end": 1.0, "text": "hello"}])

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    segments = transcribe_audio(audio, paths(tmp_path))

    assert segments == [{"start": 0.0, "end": 1.0, "text": "hello"}]
    assert captured == {
        "args": [sys.executable, "-m", "voicenotes.transcriber", str(audio), str(tmp_path / "models" / "whisper-large-v3-mlx")],
        "capture_output": True,
        "text": True,
        "check": True,
    }


def test_prompts_preserve_raw_language_choice():
    cleanup = CLEANUP_PROMPT.format(transcript_raw="[00:00:00 - 00:00:03] Testing, testing, one, two, three.")
    summary = SUMMARY_PROMPT.format(transcript_clean="[00:00:00 - 00:00:03] Testing, testing, one, two, three.")

    assert PROMPT_VERSION == "2026-09-07-summary-v6"
    assert "Your default behavior is to leave text unchanged." in cleanup
    assert 'Never replace "Testing, testing, one, two, three" with "测试，测试，一，二，三"' in cleanup
    assert "When uncertain, keep the raw transcript exactly as written." in cleanup
    assert 'must not render it as "测试，测试，一，二，三"' in summary


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
            summary_body("roadmap and 中文部分"),
        ]
    )
    monkeypatch.setattr("voicenotes.ollama.generate", lambda model, prompt, timeout_seconds=1800, **kwargs: next(responses))
    opened = []
    monkeypatch.setattr("subprocess.run", lambda args, check=False: opened.append(args))

    process_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == "[00:00:03 - 00:00:09] We discussed roadmap and 中文部分.\n"
    assert (session / "transcript_clean.md").read_text(encoding="utf-8").startswith("[00:00:03 - 00:00:09]")
    summary = (session / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("<!-- Generated by VoiceNotes from session 2026-08-27_143012 -->")
    assert "## Blockers & open questions" in summary


@pytest.mark.parametrize(
    "override, expected_preflight, expected_calls",
    [
        (None, ["qwen2.5:14b"], ["qwen2.5:14b", "qwen2.5:14b"]),
        ("summary-model", ["qwen2.5:14b", "summary-model"], ["qwen2.5:14b", "summary-model"]),
    ],
)
def test_pipeline_selects_models_per_stage(tmp_path, monkeypatch, override, expected_preflight, expected_calls):
    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    raw = "[00:00:00 - 00:00:01] Keep this sentence.\n"
    (session / "transcript_raw.md").write_text(raw)
    preflight, calls = [], []
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", preflight.append)
    monkeypatch.setattr("voicenotes.pipeline.notify", lambda *args: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda *args: pytest.fail("reuse raw"))

    def generate(model, prompt, **kwargs):
        calls.append(model)
        return raw if len(calls) == 1 else summary_body("Keep this sentence.")

    monkeypatch.setattr("voicenotes.ollama.generate", generate)
    process_session(session, replace(config(tmp_path), summary_model=override), paths(tmp_path))

    assert preflight == expected_preflight
    assert calls == expected_calls
    assert (session / "transcript_raw.md").read_text() == raw
    versions = json.loads((session / "session.json").read_text())["command_versions"]
    assert versions["ollama"] == "qwen2.5:14b"
    assert versions.get("ollama_summary") == expected_calls[-1]


def test_summary_only_retry_does_not_require_unused_cleanup_model(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    source = "[00:00:00 - 00:00:01] Existing transcript.\n"
    for name in ["transcript_raw.md", "transcript_clean.md"]:
        (session / name).write_text(source)
    (session / "session.json").write_text(json.dumps({"command_versions": {"ollama": "original-cleanup"}}))
    checked = []

    def ensure(model):
        checked.append(model)
        if model == "qwen2.5:14b":
            raise RuntimeError("cleanup model is unavailable")

    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", ensure)
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: summary_body("Existing transcript."))
    monkeypatch.setattr("voicenotes.pipeline.notify", lambda *args: None)
    retry_session(session, replace(config(tmp_path), summary_model="summary-model"), paths(tmp_path))

    assert checked == ["summary-model"]
    assert artifact_status(session)["summary"] is True
    versions = json.loads((session / "session.json").read_text())["command_versions"]
    assert versions == {"ollama": "original-cleanup", "ollama_summary": "summary-model"}


def test_missing_summary_model_fails_before_cleanup(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    (session / "transcript_raw.md").write_text("[00:00:00 - 00:00:01] Keep raw.\n")
    provenance = {"prompt_version": "old-prompt", "command_versions": {"ollama": "old-cleanup"}}
    (session / "session.json").write_text(json.dumps(provenance))

    def ensure(model):
        if model == "missing-summary-model":
            raise RuntimeError("summary model missing")

    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", ensure)
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: pytest.fail("preflight must finish before generation"))
    monkeypatch.setattr("voicenotes.pipeline.notify", lambda *args: None)
    with pytest.raises(RuntimeError, match="summary model missing"):
        process_session(session, replace(config(tmp_path), summary_model="missing-summary-model"), paths(tmp_path))
    state = json.loads((session / "session.json").read_text())
    assert state["command_versions"] == provenance["command_versions"]
    assert state["prompt_version"] == provenance["prompt_version"]
    assert not (session / "transcript_clean.md").exists()


def test_summary_prompt_requires_source_language_details_and_uncertainty():
    assert "中文内容用中文概括，保留原有 English 术语；英文内容用英文概括。" in SUMMARY_PROMPT
    assert "Include every substantive topic" in SUMMARY_PROMPT
    assert "numbers with their units and conditions" in SUMMARY_PROMPT
    assert "Do not turn advice, possibilities, examples, or questions into agreements or tasks." in SUMMARY_PROMPT
    assert "A person mentioned is not necessarily a speaker or an action owner." in SUMMARY_PROMPT
    assert "Copy the exact source wording of explicitly committed tasks" in SUMMARY_PROMPT
    assert "include every concrete numeric example or threshold" in SUMMARY_PROMPT


def test_invalid_summary_is_saved_raw_and_fails(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda audio, models: [{"start": 0, "end": 1, "text": "hello"}])
    responses = iter(["[00:00:00 - 00:00:01] hello", "Here is the summary\n\n## Meeting Metadata"])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))

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
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: summary_body("point"))

    retry_session(session, config(tmp_path), paths(tmp_path))

    assert artifact_status(session)["transcript_raw"] is True
    assert (session / "summary.md").exists()


def test_summary_omits_segment_labels_without_modifying_transcripts(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    audio = b"RIFF" + b"0" * 10000
    clean = (
        "[00:00:00 - 00:00:03] Meet at 12:30:00; CPU 25%, 640 cores.\n\n"
        "[00:00:03 - 00:00:06] 周五完成，保留 English。\n\n"
        "[00:00:06 - 00:00:09] A literal [00:01:00 - 00:02:00] reference inside speech.\n"
    )
    raw = clean
    (session / "audio.wav").write_bytes(audio)
    (session / "transcript_raw.md").write_text(raw, encoding="utf-8")
    (session / "transcript_clean.md").write_text(clean, encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda *args: pytest.fail("must reuse raw"))
    requests = []

    def generate(model, prompt, **kwargs):
        requests.append((prompt, kwargs))
        return summary_body("CPU 25%, 640 cores; 周五完成")

    monkeypatch.setattr("voicenotes.ollama.generate", generate)

    process_session(session, config(tmp_path), paths(tmp_path))

    assert len(requests) == 1
    prompt, options = requests[0]
    assert prompt.split("Transcript:\n", 1)[1] == (
        "Meet at 12:30:00; CPU 25%, 640 cores.\n\n"
        "周五完成，保留 English。\n\n"
        "A literal [00:01:00 - 00:02:00] reference inside speech.\n"
    )
    assert options["max_output_tokens"] == 3500
    assert "中文内容必须用中文" in options["system_prompt"]
    assert (session / "audio.wav").read_bytes() == audio
    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == raw
    assert (session / "transcript_clean.md").read_text(encoding="utf-8") == clean


@pytest.mark.parametrize("invalid_second_chunk", [False, True])
def test_summary_chunks_speech_and_preserves_section_details(tmp_path, monkeypatch, invalid_second_chunk):
    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    source = "[00:00:00 - 00:00:03] " + "甲" * 800 + "\n\n[00:00:03 - 00:00:06] " + "乙" * 800 + "\n"
    for name in ["transcript_raw.md", "transcript_clean.md"]:
        (session / name).write_text(source, encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    requests = []

    def generate(model, prompt, **kwargs):
        requests.append(prompt)
        if len(requests) == 2 and invalid_second_chunk:
            return "## Summary\n- broken second chunk"
        return summary_body(f"detail {len(requests)}")

    monkeypatch.setattr("voicenotes.ollama.generate", generate)
    if invalid_second_chunk:
        with pytest.raises(RuntimeError, match="summary validation failed"):
            process_session(session, config(tmp_path), paths(tmp_path))
        assert (session / "summary.raw.md").read_text().strip() == "## Summary\n- broken second chunk"
        assert not (session / "summary.md").exists()
    else:
        process_session(session, config(tmp_path), paths(tmp_path))
        summary = (session / "summary.md").read_text()
        assert artifact_status(session)["summary"] is True
        assert summary.count("## Summary\n") == 1
        assert summary.count("### Blockers\n") == 1
        assert summary.count("### Open questions\n") == 1
        for section in ["## Feedback & critique", "## Decisions", "## Action items", "### Blockers", "### Open questions", "## Next steps"]:
            body = summary.split(section + "\n", 1)[1].split("\n#", 1)[0]
            assert "detail 1" in body and "detail 2" in body
    assert len(requests) == 2
    assert "甲" * 800 in requests[0] and "乙" not in requests[0]
    assert "乙" * 800 in requests[1] and "甲" not in requests[1]
    for name in ["transcript_raw.md", "transcript_clean.md"]:
        assert (session / name).read_text() == source


def test_summary_rejects_oversized_speech_paragraph_before_generation(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    source = "[00:00:00 - 00:00:03] " + "甲" * 1201 + "\n"
    for name in ["transcript_raw.md", "transcript_clean.md"]:
        (session / name).write_text(source, encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: pytest.fail("oversized input must not reach model"))
    with pytest.raises(ValueError, match="exceeds chunk budget"):
        process_session(session, config(tmp_path), paths(tmp_path))
    assert (session / "transcript_clean.md").read_text() == source
    assert not (session / "summary.md").exists()


def test_summary_merge_preserves_topic_subheadings_and_omits_empty_placeholders(tmp_path):
    first = summary_body("none noted").replace("### Topic\n- none noted", "### Blockers\n- topic detail 1")
    second = summary_body("detail 2").replace("### Topic", "### Blockers")
    result = _merge_summaries([first, second])
    discussion = result.split("## Discussion by topic\n", 1)[1].split("## Feedback & critique", 1)[0]
    assert "topic detail 1" in discussion and "detail 2" in discussion
    assert discussion.count("### Blockers") == 2
    assert "none noted" not in result
    path = tmp_path / "summary.md"
    path.write_text(result)
    from voicenotes.state import validate_summary
    assert validate_summary(path) == (True, "ok")
    assert _merge_summaries([first]) == first


def test_summary_chunks_include_bounded_adjacent_context(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    first, second = "新版API保持兼容", "这些接口暂不删除"
    source = f"[00:00:00 - 00:00:03] {first}\n\n[00:00:03 - 00:00:06] {second}\n"
    for name in ["transcript_raw.md", "transcript_clean.md"]:
        (session / name).write_text(source)
    monkeypatch.setattr("voicenotes.pipeline.SUMMARY_CHUNK_MAX_TOKENS", 15)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    requests = []

    def generate(model, prompt, **kwargs):
        requests.append(prompt)
        return summary_body("新版API的接口暂不删除")

    monkeypatch.setattr("voicenotes.ollama.generate", generate)
    process_session(session, config(tmp_path), paths(tmp_path))
    assert len(requests) == 2
    assert f"Context after (reference only):\n{second}" in requests[0]
    assert f"Context before (reference only):\n{first}" in requests[1]
    assert requests[0].split("Transcript:\n", 1)[1] == first + "\n"
    assert requests[1].split("Transcript:\n", 1)[1] == second + "\n"


def test_retry_regenerates_malformed_utf8_transcript(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    (session / "transcript_raw.md").write_bytes(b"\xff\xfe")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        "voicenotes.pipeline.transcribe_audio",
        lambda audio, models: [{"start": 0, "end": 1, "text": "regenerated raw"}],
    )
    responses = iter(
        [
            "[00:00:00 - 00:00:01] regenerated clean",
            summary_body("point"),
        ]
    )
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))

    retry_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == "[00:00:00 - 00:00:01] regenerated raw\n"
    assert artifact_status(session)["summary"] is True


def test_retry_regenerates_downstream_artifacts_after_invalid_raw_transcript(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    (session / "transcript_raw.md").write_text("", encoding="utf-8")
    (session / "transcript_clean.md").write_text("stale clean\n", encoding="utf-8")
    (session / "summary.md").write_text(summary_body("stale"), encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda audio, models: [{"start": 0, "end": 1, "text": "fresh raw"}])
    responses = iter(
        [
            "[00:00:00 - 00:00:01] fresh clean",
            summary_body("fresh"),
        ]
    )
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))

    retry_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "transcript_clean.md").read_text(encoding="utf-8") == "[00:00:00 - 00:00:01] fresh clean\n"
    assert "- fresh" in (session / "summary.md").read_text(encoding="utf-8")


def test_retry_from_clean_reuses_raw_and_regenerates_derived_artifacts(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    raw = "[00:00:00 - 00:00:01] preserved raw\n"
    (session / "transcript_raw.md").write_text(raw, encoding="utf-8")
    (session / "transcript_clean.md").write_text("[00:00:00 - 00:00:01] stale clean\n", encoding="utf-8")
    (session / "summary.md").write_text(summary_body("stale"), encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda *args: pytest.fail("raw must be reused"))
    responses = iter(["[00:00:00 - 00:00:01] fresh clean", summary_body("fresh")])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))

    retry_session(session, config(tmp_path), paths(tmp_path), from_clean=True)

    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == raw
    assert "fresh clean" in (session / "transcript_clean.md").read_text(encoding="utf-8")
    assert "- fresh" in (session / "summary.md").read_text(encoding="utf-8")


def test_cleanup_failure_invalidates_stale_summary_before_cleanup(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    audio = b"RIFF" + b"0" * 10000
    raw = "[00:00:00 - 00:00:01] preserved raw\n"
    (session / "audio.wav").write_bytes(audio)
    (session / "transcript_raw.md").write_text(raw, encoding="utf-8")
    (session / "transcript_clean.md").write_text("truncated cached cleanup\n", encoding="utf-8")
    (session / "summary.md").write_text(summary_body("stale"), encoding="utf-8")
    (session / "summary.raw.md").write_text("stale generated output\n", encoding="utf-8")
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")))

    with pytest.raises(RuntimeError, match="cleanup failed"):
        process_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "audio.wav").read_bytes() == audio
    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == raw
    assert (session / "transcript_clean.md").read_text(encoding="utf-8") == "truncated cached cleanup\n"
    assert not (session / "summary.md").exists()
    assert not (session / "summary.raw.md").exists()


@pytest.mark.parametrize("failure_stage", ["cleanup", "summary"])
def test_retry_from_clean_failure_preserves_raw_audio_and_removes_stale_artifacts(tmp_path, monkeypatch, failure_stage):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    audio = b"RIFF" + b"0" * 10000
    raw = "[00:00:00 - 00:00:01] preserved raw\n"
    (session / "audio.wav").write_bytes(audio)
    (session / "transcript_raw.md").write_text(raw, encoding="utf-8")
    (session / "transcript_clean.md").write_text("[00:00:00 - 00:00:01] stale clean\n", encoding="utf-8")
    (session / "summary.md").write_text(summary_body("stale"), encoding="utf-8")
    (session / "summary.raw.md").write_text("stale generated output\n", encoding="utf-8")
    (session / "session.json").write_text(json.dumps({"command_versions": {"ollama": "old-cleanup"}}))
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    if failure_stage == "cleanup":
        monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")))
    else:
        responses = iter(["[00:00:00 - 00:00:01] fresh clean", RuntimeError("summary failed")])

        def fail_summary(*args, **kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr("voicenotes.ollama.generate", fail_summary)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        retry_session(session, config(tmp_path), paths(tmp_path), from_clean=True)

    assert (session / "audio.wav").read_bytes() == audio
    assert (session / "transcript_raw.md").read_text(encoding="utf-8") == raw
    assert not (session / "summary.md").exists()
    assert not (session / "summary.raw.md").exists()
    if failure_stage == "cleanup":
        assert not (session / "transcript_clean.md").exists()
        assert json.loads((session / "session.json").read_text())["command_versions"] == {"ollama": "old-cleanup"}
    else:
        assert (session / "transcript_clean.md").read_text(encoding="utf-8") == "[00:00:00 - 00:00:01] fresh clean\n"
        assert json.loads((session / "session.json").read_text())["command_versions"] == {"ollama": "qwen2.5:14b"}


def test_process_session_rejects_empty_raw_transcript_before_cleanup(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: None)
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda audio, models: [])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: pytest.fail("should not clean an invalid raw transcript"))

    with pytest.raises(RuntimeError, match="raw transcript validation failed"):
        process_session(session, config(tmp_path), paths(tmp_path))


def test_process_session_opens_valid_existing_summary_when_enabled(tmp_path, monkeypatch):
    session = tmp_path / "VoiceNotes" / "2026-08-27_143012"
    session.mkdir(parents=True)
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    (session / "transcript_raw.md").write_text("[00:00:00 - 00:00:01] existing raw\n", encoding="utf-8")
    (session / "transcript_clean.md").write_text("[00:00:00 - 00:00:01] existing clean\n", encoding="utf-8")
    (session / "summary.md").write_text(
        "<!-- Generated by VoiceNotes from session 2026-08-27_143012 -->\n\n" + summary_body("point"),
        encoding="utf-8",
    )
    opened = []
    monkeypatch.setattr("subprocess.run", lambda args, check=False: opened.append(args))
    monkeypatch.setattr("voicenotes.pipeline.notify", lambda title, message: None)
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda model: pytest.fail("should skip model preflight"))

    process_session(session, replace(config(tmp_path), auto_open=True), paths(tmp_path))

    assert opened == [["open", "-g", str(session / "summary.md")]]


@pytest.mark.parametrize("known_provenance", [False, True])
def test_cached_retry_does_not_relabel_existing_generation(tmp_path, monkeypatch, known_provenance):
    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    for name in ["transcript_raw.md", "transcript_clean.md"]:
        (session / name).write_text("[00:00:00 - 00:00:01] Existing words.\n")
    (session / "summary.md").write_text(summary_body("Existing words."))
    provenance = {
        "prompt_version": "previous-prompt",
        "command_versions": {"ollama": "previous-model"},
        "completed_at": "2026-01-01T12:00:00",
    } if known_provenance else {}
    (session / "session.json").write_text(json.dumps(provenance))
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda *args: pytest.fail("reuse valid cache"))
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: pytest.fail("reuse valid cache"))
    monkeypatch.setattr("voicenotes.pipeline.notify", lambda *args: None)

    retry_session(session, replace(config(tmp_path), summary_model="new-summary-model"), paths(tmp_path))

    state = json.loads((session / "session.json").read_text())
    assert state["status"] == "complete"
    for key in ["prompt_version", "command_versions", "completed_at"]:
        if known_provenance:
            assert state[key] == provenance[key]
        else:
            assert key not in state


def test_retry_rebuilds_truncated_cached_cleanup_and_summary(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    audio = b"RIFF" + b"0" * 10000
    raw = "[00:00:00 - 00:00:01] First detail.\n\n[00:00:01 - 00:00:02] Last detail.\n"
    (session / "audio.wav").write_bytes(audio)
    (session / "transcript_raw.md").write_text(raw)
    (session / "transcript_clean.md").write_text(raw.split("\n\n")[0])
    (session / "summary.md").write_text(summary_body("stale"))
    monkeypatch.setattr("voicenotes.pipeline.transcribe_audio", lambda *args: pytest.fail("reuse raw"))
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda *args: None)
    responses = iter([raw, summary_body("First detail; Last detail")])
    monkeypatch.setattr("voicenotes.ollama.generate", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("voicenotes.pipeline.notify", lambda *args: None)

    retry_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "transcript_clean.md").read_text() == raw
    assert "Last detail" in (session / "summary.md").read_text()
    assert (session / "transcript_raw.md").read_text() == raw
    assert (session / "audio.wav").read_bytes() == audio


@pytest.mark.parametrize("collapsed", [False, True])
def test_retry_reuses_faithful_cached_cleanup_with_blank_and_filler_segments(tmp_path, monkeypatch, collapsed):
    from voicenotes.pipeline import collapse_filler_runs

    session = tmp_path / "session"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"RIFF" + b"0" * 10000)
    paragraphs = [
        "[00:00:00 - 00:00:01] Words.",
        "[00:00:01 - 00:00:01]",
        "[00:00:01 - 00:00:01]",
        "[00:00:01 - 00:00:02] 嗯",
        "[00:00:02 - 00:00:03] 嗯",
        "[00:00:03 - 00:00:04] 嗯",
    ]
    raw = "\n\n".join(paragraphs) + "\n"
    clean = "\n\n".join(collapse_filler_runs(paragraphs) if collapsed else paragraphs) + "\n"
    (session / "transcript_raw.md").write_text(raw)
    (session / "transcript_clean.md").write_text(clean)
    (session / "summary.md").write_text(summary_body("Words."))
    monkeypatch.setattr("voicenotes.ollama.ensure_model_available", lambda *args: pytest.fail("reuse valid cache"))
    monkeypatch.setattr("voicenotes.pipeline.notify", lambda *args: None)

    retry_session(session, config(tmp_path), paths(tmp_path))

    assert (session / "transcript_clean.md").read_text() == clean
    assert (session / "transcript_raw.md").read_text() == raw


@pytest.mark.parametrize("clean", [b"\xff\xfe", b"[00:00:00 - 00:00:01]", b"[00:00:00 - 00:00:01] Short."])
def test_artifact_status_rejects_unreadable_or_content_losing_cleanup(tmp_path, clean):
    (tmp_path / "transcript_raw.md").write_text("[00:00:00 - 00:00:01] " + "Substantive details. " * 5)
    (tmp_path / "transcript_clean.md").write_bytes(clean)
    assert artifact_status(tmp_path)["transcript_clean"] is False
