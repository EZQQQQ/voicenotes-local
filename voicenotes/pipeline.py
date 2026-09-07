from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import voicenotes.ollama as ollama

from .config import AppConfig, Paths
from .state import atomic_write_json, atomic_write_text, clear_last_error, notify, read_json, validate_summary, write_last_error


WHISPER_REPO_ID = "mlx-community/whisper-large-v3-mlx"
PROMPT_VERSION = "2026-09-07-cleanup-v1"
AUDIO_MIN_BYTES = 4096
TRANSCRIPT_MIN_CHARACTERS = 1
CLEANUP_CHUNK_MAX_TOKENS = 1200
CLEANUP_MAX_OUTPUT_TOKENS = 3500


def estimate_tokens(text: str) -> int:
    cjk_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
    other_chars = len(text) - cjk_chars
    return cjk_chars + math.ceil(other_chars / 4)


def chunk_paragraphs(paragraphs: list[str], max_tokens: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    separator_tokens = estimate_tokens("\n\n")
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > max_tokens:
            raise ValueError(f"transcript paragraph exceeds chunk budget: {paragraph_tokens} > {max_tokens}")
        added_tokens = paragraph_tokens + (separator_tokens if current else 0)
        if current and current_tokens + added_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
            added_tokens = paragraph_tokens
        current.append(paragraph)
        current_tokens += added_tokens
    if current:
        chunks.append(current)
    return chunks

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

Your default behavior is to leave text unchanged. Only edit words when there is
clear contextual evidence of an ASR mistake. Language choice is evidence: if
the raw transcript contains English words, keep them in English; if it contains
Chinese characters, keep them in Chinese.
Never replace "Testing, testing, one, two, three" with "测试，测试，一，二，三" or any other translation.
When uncertain, keep the raw transcript exactly as written.

Output only the corrected transcript, preserving original paragraph structure.
Copy every input timestamp occurrence exactly once, unchanged and in the same order.
Do not merge, split, add, remove, or reorder paragraphs.
Begin your output with the first input timestamp.
Do not include commentary, headings, code fences, or a preamble.

Transcript:
{transcript_raw}
"""

SUMMARY_PROMPT = """You are summarizing a cleaned transcript of a recording that may mix English
and Mandarin Chinese. Do not translate or normalize the language — preserve
terms, names, and phrases exactly as they appear in the transcript.
Language choice is locked to the transcript: if the transcript says "Testing, testing, one, two, three", the summary must not render it as "测试，测试，一，二，三".

Produce a Markdown summary with exactly these sections, in this order. Only
include what is explicitly stated or clearly inferable from the transcript —
do not invent a title, date, attendees, or content; write "not specified" /
"none noted" instead.

# Meeting title
A short title inferred from the discussion, followed by a line with the date
(if stated) and participant names (if stated), separated by " · ".

## Summary
2-5 bullets: the main discussion points and overall context.

## Discussion by topic
One "### Topic name" subsection per distinct topic discussed, in the order
raised. Under each, a few bullets on what was covered and any key details
(facts, metrics, requirements, examples).

## Feedback & critique
Points raised about the work being reviewed — critique, suggestions, or
advice given, and by whom if clear. This is distinct from confirmed
Decisions below: capture guidance and opinions even if nothing was decided.

## Decisions
A clear log of confirmed agreements only — not proposals or open debate.

## Action items
A checkable list (- [ ] task). Include owner and deadline where stated;
write "unassigned" / "no deadline given" where not stated.

## Blockers & open questions
### Blockers
Issues blocking progress and any next step discussed.
### Open questions
Unresolved questions requiring follow-up.

## Next steps
Planned follow-ups, milestones, or future discussions.

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
    result = subprocess.run(
        [sys.executable, "-m", "voicenotes.transcriber", str(audio), str(whisper_model_dir(paths))],
        capture_output=True,
        text=True,
        check=True,
    )
    return list(json.loads(result.stdout))


def write_raw_transcript(session: Path, segments: list[dict[str, object]]) -> None:
    paragraphs = [
        f"[{format_timestamp(float(segment['start']))} - {format_timestamp(float(segment['end']))}] {str(segment['text']).strip()}"
        for segment in segments
    ]
    atomic_write_text(session / "transcript_raw.md", "\n\n".join(paragraphs) + "\n")


FILLER_TOKENS = {"嗯", "呃", "啊", "嗯嗯", "呃呃"}
FILLER_RUN_MIN_SEGMENTS = 3
FILLER_MAX_GAP_SECONDS = 2
TRANSCRIPT_PARAGRAPH = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2}) - (?P<end>\d{2}:\d{2}:\d{2})\](?: (?P<text>.*))?$"
)


def _timestamp_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _parse_timestamped_paragraph(paragraph: str) -> tuple[str, str, str] | None:
    match = TRANSCRIPT_PARAGRAPH.fullmatch(paragraph)
    if match is None:
        return None
    return match.group("start"), match.group("end"), match.group("text") or ""


def _normalized_filler(text: str) -> str | None:
    stripped = text.strip().strip("。.!?！？,，")
    return stripped if stripped in FILLER_TOKENS else None


def collapse_filler_runs(paragraphs: list[str]) -> list[str]:
    collapsed: list[str] = []
    run: list[tuple[str, str, str, str]] = []
    run_token: str | None = None

    def flush_run() -> None:
        nonlocal run_token
        if len(run) < FILLER_RUN_MIN_SEGMENTS:
            collapsed.extend(item[0] for item in run)
        elif run:
            collapsed.append(f"[{run[0][1]} - {run[-1][2]}] {run[0][3].strip()}")
        run.clear()
        run_token = None

    for paragraph in paragraphs:
        parsed = _parse_timestamped_paragraph(paragraph)
        token = _normalized_filler(parsed[2]) if parsed is not None else None
        if parsed is None or token is None:
            flush_run()
            collapsed.append(paragraph)
            continue
        start, end, text = parsed
        gap = _timestamp_seconds(start) - _timestamp_seconds(run[-1][2]) if run else 0
        if run and (token != run_token or gap > FILLER_MAX_GAP_SECONDS):
            flush_run()
        if not run:
            run_token = token
        run.append((paragraph, start, end, text))
    flush_run()
    return collapsed


class _CleanupValidationError(RuntimeError):
    pass


def _validate_cleaned_chunk(source: str, cleaned: str) -> None:
    source_paragraphs = source.strip().split("\n\n")
    cleaned_paragraphs = cleaned.strip().split("\n\n")
    source_parts = [_parse_timestamped_paragraph(paragraph) for paragraph in source_paragraphs]
    cleaned_parts = [_parse_timestamped_paragraph(paragraph) for paragraph in cleaned_paragraphs]
    source_timestamps = [(part[0], part[1]) for part in source_parts if part is not None]
    cleaned_timestamps = [(part[0], part[1]) for part in cleaned_parts if part is not None]
    if len(source_timestamps) != len(source_paragraphs) or source_timestamps != cleaned_timestamps:
        raise _CleanupValidationError("clean transcript validation failed: timestamps missing or reordered")
    if len(cleaned_parts) != len(source_parts) or any(part is None for part in cleaned_parts):
        raise _CleanupValidationError("clean transcript validation failed: timestamps missing or reordered")
    for source_part, cleaned_part in zip(source_parts, cleaned_parts):
        if source_part is not None and cleaned_part is not None and source_part[2].strip() and not cleaned_part[2].strip():
            raise _CleanupValidationError("clean transcript validation failed: segment content missing")
        if source_part is not None and cleaned_part is not None:
            source_length = len(re.sub(r"\s", "", source_part[2]))
            cleaned_length = len(re.sub(r"\s", "", cleaned_part[2]))
            if source_length >= 40 and cleaned_length < source_length * 0.8:
                raise _CleanupValidationError("clean transcript validation failed: segment content shortened")


def _clean_chunk(model: str, paragraphs: list[str]) -> list[str]:
    source = "\n\n".join(paragraphs)
    cleaned = ollama.generate(model, CLEANUP_PROMPT.format(transcript_raw=source), max_output_tokens=CLEANUP_MAX_OUTPUT_TOKENS).strip()
    try:
        _validate_cleaned_chunk(source, cleaned)
    except _CleanupValidationError:
        if len(paragraphs) == 1:
            raise
        midpoint = len(paragraphs) // 2
        return _clean_chunk(model, paragraphs[:midpoint]) + _clean_chunk(model, paragraphs[midpoint:])
    return cleaned.split("\n\n")


def clean_transcript(model: str, raw_text: str) -> str:
    if not raw_text.strip():
        raise RuntimeError("raw transcript validation failed")
    paragraphs = collapse_filler_runs(raw_text.strip().split("\n\n"))
    blank_paragraphs: dict[int, str] = {}
    cleanup_paragraphs: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        parsed = _parse_timestamped_paragraph(paragraph)
        if parsed is not None and not parsed[2].strip():
            blank_paragraphs[index] = paragraph
        else:
            cleanup_paragraphs.append(paragraph)
    chunks = chunk_paragraphs(cleanup_paragraphs, CLEANUP_CHUNK_MAX_TOKENS)
    cleaned_paragraphs = [paragraph for chunk in chunks for paragraph in _clean_chunk(model, chunk)]
    cleaned_iter = iter(cleaned_paragraphs)
    return "\n\n".join(blank_paragraphs[index] if index in blank_paragraphs else next(cleaned_iter) for index in range(len(paragraphs)))


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
        clean_path = session / "transcript_clean.md"
        summary_path = session / "summary.md"
        needs_raw = not status["transcript_raw"]
        needs_clean = needs_raw or not status["transcript_clean"]
        needs_summary = needs_clean or not status["summary"]
        if needs_raw:
            segments = transcribe_audio(session / "audio.wav", paths)
            clean_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
            (session / "summary.raw.md").unlink(missing_ok=True)
            write_raw_transcript(session, segments)
            if not _valid_text(raw_path):
                raise RuntimeError("raw transcript validation failed")

        if needs_clean:
            raw_transcript = raw_path.read_text(encoding="utf-8")
            summary_path.unlink(missing_ok=True)
            (session / "summary.raw.md").unlink(missing_ok=True)
            cleaned_output = clean_transcript(config.ollama_model, raw_transcript)
            atomic_write_text(clean_path, cleaned_output + "\n")
            if not _valid_text(clean_path):
                raise RuntimeError("clean transcript validation failed")

        if needs_summary:
            cleaned_transcript_text = clean_path.read_text(encoding="utf-8")
            generated = ollama.generate(config.ollama_model, SUMMARY_PROMPT.format(transcript_clean=cleaned_transcript_text))
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


def retry_session(session: Path, config: AppConfig, paths: Paths, from_clean: bool = False) -> None:
    if from_clean:
        (session / "transcript_clean.md").unlink(missing_ok=True)
        (session / "summary.md").unlink(missing_ok=True)
        (session / "summary.raw.md").unlink(missing_ok=True)
    process_session(session, config, paths)
