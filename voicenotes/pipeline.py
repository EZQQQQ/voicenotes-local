from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess

import voicenotes.ollama as ollama

from .config import AppConfig, Paths
from .state import atomic_write_json, atomic_write_text, clear_last_error, notify, read_json, validate_summary, write_last_error


WHISPER_REPO_ID = "mlx-community/whisper-large-v3-mlx"
PROMPT_VERSION = "2026-08-27-v1"
AUDIO_MIN_BYTES = 4096
TRANSCRIPT_MIN_CHARACTERS = 1

CLEANUP_PROMPT = """You are a bilingual (English / Mandarin Chinese) transcript proofreader.

You will be given a raw transcript produced by automatic speech recognition
from a recording that freely mixes English and Chinese, sometimes switching
mid-sentence. The ASR system is known to make these specific error types:
- misidentifying which language a phrase is in
- phonetically-plausible wrong words
- occasionally translating a short phrase instead of transcribing it literally

Your task: correct likely ASR errors using context. Do NOT translate anything
from one language into the other — preserve the speaker's original language
choice exactly as spoken. Do NOT summarize, shorten, or omit any content.
Do NOT normalize Chinese script (leave Simplified/Traditional as transcribed).

Output only the corrected transcript, preserving original paragraph structure.

Transcript:
{transcript_raw}
"""

SUMMARY_PROMPT = """You are summarizing a cleaned transcript of a recording that may mix English
and Mandarin Chinese. Do not translate or normalize the language — preserve
terms, names, and phrases exactly as they appear in the transcript.

Produce a Markdown summary with exactly these sections, in this order:

## Meeting Metadata
Date, objective, and attendee list. Only include what is explicitly stated
or clearly inferable from the transcript — do not invent attendees or an
objective if the recording doesn't state them; note "not specified" instead.

## Key Discussion Points
Main topics, ordered chronologically or by relevance.

## Decisions Made
A clear log of confirmed agreements only — not proposals or open debate.

## Action Items
A checkable list (- [ ] task). Include owner and deadline where stated;
write "unassigned" / "no deadline given" where not stated.

## Open Questions
Unresolved issues or blockers requiring follow-up.

Transcript:
{transcript_clean}
"""


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def whisper_model_dir(paths: Paths) -> Path:
    return paths.models / "whisper-large-v3-mlx"


def download_whisper_model(paths: Paths) -> Path:
    from huggingface_hub import snapshot_download

    target = whisper_model_dir(paths)
    snapshot_download(repo_id=WHISPER_REPO_ID, local_dir=target)
    return target


def transcribe_audio(audio: Path, paths: Paths) -> list[dict[str, object]]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=str(whisper_model_dir(paths)),
        task="transcribe",
        language=None,
        initial_prompt="This recording mixes English and Mandarin Chinese, sometimes switching mid-sentence.",
    )
    return list(result["segments"])


def write_raw_transcript(session: Path, segments: list[dict[str, object]]) -> None:
    paragraphs = [
        f"[{format_timestamp(float(segment['start']))} - {format_timestamp(float(segment['end']))}] {str(segment['text']).strip()}"
        for segment in segments
    ]
    atomic_write_text(session / "transcript_raw.md", "\n\n".join(paragraphs) + "\n")


def _valid_text(path: Path) -> bool:
    try:
        return path.exists() and len(path.read_text(encoding="utf-8").strip()) >= TRANSCRIPT_MIN_CHARACTERS
    except (OSError, UnicodeDecodeError):
        return False


def _valid_summary(path: Path) -> bool:
    try:
        valid, _ = validate_summary(path)
        return valid
    except (OSError, UnicodeDecodeError):
        return False


def artifact_status(session: Path) -> dict[str, bool]:
    return {
        "audio": (session / "audio.wav").is_file() and (session / "audio.wav").stat().st_size >= AUDIO_MIN_BYTES,
        "transcript_raw": _valid_text(session / "transcript_raw.md"),
        "transcript_clean": _valid_text(session / "transcript_clean.md"),
        "summary": _valid_summary(session / "summary.md"),
    }


def _session_state(session: Path) -> dict[str, object]:
    path = session / "session.json"
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def _write_session_state(session: Path, status: str, config: AppConfig, error: str | None = None) -> None:
    state = _session_state(session)
    now = datetime.now().isoformat(timespec="seconds")
    state.update(
        {
            "status": status,
            "updated_at": now,
            "prompt_version": PROMPT_VERSION,
            "command_versions": {"whisper": WHISPER_REPO_ID, "ollama": config.ollama_model},
            "error": error,
        }
    )
    if status == "processing":
        state["processing_started_at"] = now
    if status == "complete":
        state["completed_at"] = now
    atomic_write_json(session / "session.json", state)


def _write_failure(session: Path, config: AppConfig, paths: Paths, error: Exception) -> None:
    message = str(error)
    atomic_write_text(session / "error.log", message + "\n")
    atomic_write_text(session / "pipeline.log", f"failed: {message}\n")
    _write_session_state(session, "error", config, message)
    write_last_error(paths, message)
    notify("Note processing failed", session.name)


def process_session(session: Path, config: AppConfig, paths: Paths) -> None:
    try:
        status = artifact_status(session)
        if not status["audio"]:
            raise RuntimeError("audio validation failed")
        if all(status.values()):
            _write_session_state(session, "complete", config)
            atomic_write_text(session / "pipeline.log", "complete: existing artifacts are valid\n")
            clear_last_error(paths)
            if config.auto_open:
                subprocess.run(["open", "-g", str(session / "summary.md")], check=False)
            notify("Note ready", session.name)
            return

        _write_session_state(session, "processing", config)
        ollama.ensure_model_available(config.ollama_model)

        raw_path = session / "transcript_raw.md"
        needs_raw = not status["transcript_raw"]
        needs_clean = needs_raw or not status["transcript_clean"]
        needs_summary = needs_clean or not status["summary"]
        if needs_raw:
            write_raw_transcript(session, transcribe_audio(session / "audio.wav", paths))
            if not _valid_text(raw_path):
                raise RuntimeError("raw transcript validation failed")

        clean_path = session / "transcript_clean.md"
        if needs_clean:
            raw_transcript = raw_path.read_text(encoding="utf-8")
            atomic_write_text(clean_path, ollama.generate(config.ollama_model, CLEANUP_PROMPT.format(transcript_raw=raw_transcript)) + "\n")
            if not _valid_text(clean_path):
                raise RuntimeError("clean transcript validation failed")

        summary_path = session / "summary.md"
        if needs_summary:
            clean_transcript = clean_path.read_text(encoding="utf-8")
            generated = ollama.generate(config.ollama_model, SUMMARY_PROMPT.format(transcript_clean=clean_transcript))
            atomic_write_text(summary_path, f"<!-- Generated by VoiceNotes from session {session.name} -->\n\n{generated.strip()}\n")
            summary_valid, reason = validate_summary(summary_path)
            if not summary_valid:
                atomic_write_text(session / "summary.raw.md", generated + "\n")
                raise RuntimeError(f"summary validation failed: {reason}")

        _write_session_state(session, "complete", config)
        atomic_write_text(session / "pipeline.log", "complete\n")
        (session / "error.log").unlink(missing_ok=True)
        clear_last_error(paths)
        if config.auto_open:
            subprocess.run(["open", "-g", str(summary_path)], check=False)
        notify("Note ready", session.name)
    except Exception as error:
        _write_failure(session, config, paths, error)
        raise


def retry_session(session: Path, config: AppConfig, paths: Paths) -> None:
    process_session(session, config, paths)
