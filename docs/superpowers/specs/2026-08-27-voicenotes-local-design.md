# VoiceNotes Local Design

## Purpose

VoiceNotes Local is a personal macOS tool for recording spoken notes with a global hotkey, transcribing them locally with Whisper large-v3, cleaning the transcript with a local bilingual LLM, and generating a structured Markdown summary.

The v1 priority is simplicity and reliability for one user on Apple Silicon Macs. There is no hosted backend, no account system, no sync layer, no database, and no cloud inference.

## Target Environment

- macOS on Apple Silicon only.
- 16GB unified memory minimum.
- Homebrew-managed Python 3.11+.
- Hammerspoon for the global hotkey and menu-bar state.
- ffmpeg with avfoundation for microphone capture.
- mlx-whisper for local Whisper `large-v3` transcription.
- Ollama with `qwen2.5:14b` by default for cleanup and summarization.

Intel Macs fail early during install with a clear Apple Silicon requirement. There is no Rosetta fallback in v1.

## Default Paths

```text
App source:       ~/.voicenotes/app
Runtime/state:    ~/.voicenotes/run
Config:           ~/.voicenotes/config.toml
Models:           ~/.voicenotes/models
Output sessions:  ~/VoiceNotes
```

Session folders use second precision to avoid collisions:

```text
~/VoiceNotes/2026-08-27_143012/
  audio.wav
  ffmpeg.log
  transcript_raw.md
  transcript_clean.md
  summary.md
  pipeline.log
  error.log
  session.json
```

If an interrupted WAV is repaired, the original is kept as `audio.interrupted.wav` and the repaired file is written to `audio.wav`.

## Repository Layout

```text
voicenotes-local/
  install.sh
  uninstall.sh
  requirements.txt
  README.md
  LICENSE
  config.example.toml
  scripts/
    smoke-test.sh
  hammerspoon/
    voicenotes.lua
  voicenotes/
    __main__.py
    cli.py
    config.py
    recorder.py
    pipeline.py
    state.py
    ollama.py
```

This is a plain Python package. There is no framework, daemon, database, or plugin system.

## Configuration

The installer writes user config to `~/.voicenotes/config.toml`. The repo includes `config.example.toml`, not a real user config.

V1 exposes exactly these user-editable fields:

- `output_root`
- `hotkey`
- `audio_device`
- `ollama_model`
- `auto_open`

`ollama_model` is configurable because the most likely second-machine change is moving to a larger local model, such as `qwen2.5:32b` on a 32GB Mac. The Whisper model is intentionally not configurable in v1 because `large-v3` is part of the product promise for English/Mandarin code-switching accuracy.

The prompts, Ollama temperature, Ollama keep-alive, and Whisper model are source constants. The Ollama keep-alive is `30s`, long enough to reuse the model between cleanup and summary while avoiding unnecessary memory residency afterward. Ollama calls use temperature `0.2`.

Hammerspoon does not parse TOML. It asks the CLI for JSON via `voicenotes config --json` and `voicenotes status --json`.

## CLI

The `voicenotes` command exposes:

- `voicenotes toggle`
- `voicenotes start`
- `voicenotes stop`
- `voicenotes process <session>`
- `voicenotes retry <session>`
- `voicenotes status --json`
- `voicenotes config --json`
- `voicenotes doctor`
- `voicenotes devices`
- `voicenotes record-test`

Hammerspoon calls `voicenotes toggle` for the hotkey. `start` and `stop` exist for debugging and scripts.

Exit codes are:

- `0`: success
- `1`: expected actionable failure
- `2`: internal or unexpected error

Details go to stderr and, for session work, into session logs.

## Hammerspoon Integration

The installer writes a self-contained `~/.hammerspoon/voicenotes.lua` and idempotently appends exactly one line to `~/.hammerspoon/init.lua`:

```lua
require("voicenotes")
```

Before modifying `init.lua`, the installer creates one timestamped backup. It does not back up when no change is needed. It does not block-replace managed regions inside the user's config.

Reload is best-effort with:

```bash
open -g hammerspoon://reload
```

If reload fails, install still succeeds and prints that the user should restart Hammerspoon or use Reload Config.

Manual v1 setup is limited to first launching Hammerspoon and granting macOS permissions. The installer cannot grant Microphone or Accessibility permissions.

The default hotkey is `Cmd+\`` because that is the intended personal default. The hotkey is configurable. The README documents a Hyper-key alternative. On second Macs, the installer detects that `Cmd+\`` may conflict with the macOS "Move focus to next window" shortcut and prints the System Settings path to clear it.

Hammerspoon owns only:

- global hotkey binding
- menu-bar state display
- invoking the CLI
- reacting to state-file changes

It does not implement recording, device resolution, pipeline state, or queue logic.

The menu-bar state is derived from `voicenotes status --json`, not Lua in-memory state. Hammerspoon watches `~/.voicenotes/run/` with `hs.pathwatcher` instead of timer polling.

On recording start, v1 uses the menu-bar state change plus a short system sound. It does not show a notification on every start. Notifications are used for stopped/queued, ready, and error states.

## Audio Capture

The CLI owns ffmpeg recording via avfoundation.

`audio_device` in config is either:

- `default`: resolve avfoundation index `0` at recording time.
- exact device name: enumerate avfoundation devices at recording time and use the current matching index.

Device indices are never frozen at install time because avfoundation indices can shift when devices are added or removed. V1 requires exact device-name matches. If no exact match exists, recording fails with a clear list of available devices. There is no fuzzy matching in v1.

ffmpeg stderr is written to the session folder as `ffmpeg.log`.

Microphone permissions are process-specific under macOS TCC. Hammerspoon-triggered recording requires Hammerspoon microphone permission; `voicenotes record-test` run from Terminal requires Terminal microphone permission. The README and `doctor` explain this directly.

## Recording State

Active recording state lives at:

```text
~/.voicenotes/run/current-recording.json
```

It is written atomically with temp-file-plus-rename and includes:

- session path
- ffmpeg PID
- resolved audio device
- start time
- ffmpeg log path

`start` refuses to begin if `current-recording.json` points to a live ffmpeg process. PID liveness must check both that the PID exists and that `ps -p <pid> -o comm=` matches ffmpeg, because PIDs can be recycled.

If the state file is stale, the CLI does not auto-process the orphaned partial audio. It leaves all files in place, clears or quarantines stale runtime state, and notifies the user. The user can inspect the session and use `retry` if appropriate.

## Stop Semantics

`stop` reads `current-recording.json`, validates the ffmpeg process identity, and finalizes recording with this escalation:

1. Send SIGINT and wait up to 10 seconds.
2. Send SIGTERM and wait up to 3 seconds.
3. Send SIGKILL.
4. Attempt WAV header repair.
5. Mark the session as interrupted in `session.json`.

SIGKILL is acceptable in v1 because WAV PCM frames are written incrementally and damaged RIFF chunk sizes are often repairable. This would not be acceptable for formats such as m4a where metadata finalization is more fragile.

After stop, the session is enqueued for processing and a worker spawn is attempted.

## Pipeline Queue

Whisper and Ollama must never run concurrently. The pipeline is serialized by a lock and queue under `~/.voicenotes/run/`.

Queue items are JSON files in:

```text
~/.voicenotes/run/queue/
```

Each queue item filename includes enqueue timestamp plus session basename. JSON is used instead of symlinks for portability and easy inspection.

Lock acquisition must be atomic, using `O_EXCL` file creation or `mkdir`, never check-then-create. The lock records the worker PID. A lock is stale if the PID is dead. This prevents one crashed worker from permanently wedging the queue.

There is no daemon in v1. Enqueue always attempts to spawn a worker. Spawning is a cheap no-op when the lock is held. This avoids the race where a session is enqueued after a worker observes an empty queue but before it exits.

The worker drains queued sessions sequentially and exits. It writes only startup and drain events to:

```text
~/.voicenotes/run/worker.log
```

Session-specific logs stay inside each session folder.

## Pipeline

The pipeline runs:

1. Transcribe `audio.wav` with mlx-whisper `large-v3`.
2. Write `transcript_raw.md`.
3. Clean transcript with Ollama.
4. Write `transcript_clean.md`.
5. Summarize with Ollama.
6. Validate `summary.md`.
7. Open `summary.md` with `open -g` if `auto_open` is true.
8. Notify that the note is ready.

The Whisper call is in-process Python so the code can explicitly set:

- `task="transcribe"`
- `language=None`
- the configured local model path
- the initial prompt for English/Mandarin code-switching

The initial prompt is:

```text
This recording mixes English and Mandarin Chinese, sometimes switching mid-sentence.
```

The installer pre-downloads Whisper `large-v3` via `huggingface_hub.snapshot_download()` into `~/.voicenotes/models/`. This satisfies the offline-after-install requirement. `doctor` verifies the model is present on disk.

The pinned MLX-compatible Hugging Face repo ID is `mlx-community/whisper-large-v3-mlx`. The Hugging Face model card documents use with `mlx_whisper.transcribe(..., path_or_hf_repo="mlx-community/whisper-large-v3-mlx")`. It is not exposed in user config.

Raw transcript timestamps use this format, one segment per paragraph:

```markdown
[00:00:03 - 00:00:09] transcript text
```

Cleanup preserves paragraph and segment structure.

All generated artifacts are written atomically with temp-file-plus-rename so retry never mistakes a partial file for a completed step.

## Cleanup Prompt

```text
You are a bilingual (English / Mandarin Chinese) transcript proofreader.

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
```

## Summary Prompt

```text
You are summarizing a cleaned transcript of a recording that may mix English
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
```

`summary.md` begins with only this generated provenance comment:

```markdown
<!-- Generated by VoiceNotes from session 2026-08-27_143012 -->
```

The `Meeting Metadata` section still says `not specified` for transcript-derived fields when the transcript does not state them. It does not silently use session metadata as meeting metadata.

If Ollama returns extra prose before or after the five required Markdown sections, validation fails. The raw response is written to `summary.raw.md`, and details are written to `error.log`. V1 does not silently repair malformed summaries with another LLM or postprocessor.

Ollama calls use:

```json
{
  "model": "qwen2.5:14b",
  "stream": false,
  "options": { "temperature": 0.2 },
  "keep_alive": "30s"
}
```

Each Ollama call has a 30-minute timeout. The pipeline preflights the Ollama server and verifies the configured model is present via `/api/tags` before starting model-dependent work. Missing models fail with `ollama pull` instructions.

## Retry And Validation

`voicenotes retry <session>` resumes from the first missing or invalid artifact. It skips valid existing artifacts.

Completion checks are:

- `audio.wav`: exists and passes an audio size/validity floor stronger than non-empty. A killed ffmpeg can leave a valid WAV header with no frames, so non-empty is insufficient.
- `transcript_raw.md`: exists and passes a minimum content floor.
- `transcript_clean.md`: exists and passes a minimum content floor.
- `summary.md`: exists and contains exactly the five required headings in the required order, with no extra top-level sections. The provenance comment is allowed before the first heading.

`session.json` records status, timestamps, command versions, error state, prompt version, whether recording was interrupted, and any WAV repair status. It is written atomically and is not the source of truth for artifact completion.

## Doctor And Smoke Tests

`voicenotes doctor` checks:

- Apple Silicon architecture.
- Homebrew dependency availability.
- Homebrew Python 3.11+ and venv package imports.
- Whisper model presence under `~/.voicenotes/models/`.
- Ollama server availability.
- Configured Ollama model presence.
- Hammerspoon file installation.
- Hotkey config readability.
- ffmpeg avfoundation device enumeration.
- whether a short test recording can be created.

Doctor cannot perfectly inspect macOS TCC permissions. It infers likely permission problems from recording failures and prints the exact System Settings panes to check.

`voicenotes record-test` performs a fixed-duration local recording using shared device-resolution code. It is a narrow test harness, not a competing recorder.

The repo includes `scripts/smoke-test.sh`. It uses `voicenotes record-test`, validates artifact presence, validates the five summary headings, and exits 0 on success. Real semantic validation is manual against the user's own local bilingual clip. No real voice recordings are committed to the public repo; local fixtures are gitignored.

## Installer

`install.sh` is the public curl-pipe entrypoint:

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/voicenotes-local/main/install.sh | bash
```

The installer source URL is configurable with `VOICENOTES_REPO_URL`. This supports local testing before the public repo exists:

```bash
VOICENOTES_REPO_URL=/path/to/local/repo ./install.sh
```

The installer:

- fails early on non-Apple-Silicon Macs.
- checks for or installs Homebrew.
- installs ffmpeg, Hammerspoon, Ollama, git, and Homebrew Python 3.11+.
- clones or updates the repo into `~/.voicenotes/app`.
- creates or refreshes a local venv under `~/.voicenotes/venv` from pinned `requirements.txt`.
- writes `~/.voicenotes/config.toml` only if missing, preserving existing user config.
- installs a `voicenotes` wrapper on PATH.
- downloads the configured Ollama model.
- downloads Whisper `large-v3` into `~/.voicenotes/models/`.
- installs Hammerspoon integration.
- runs or suggests `voicenotes doctor`.

Before model downloads, interactive install prints the approximate combined download size, around 12GB, and asks for confirmation. `NONINTERACTIVE=1` proceeds without prompting.

The PATH wrapper is installed into `/opt/homebrew/bin` on Apple Silicon when writable. If not writable, it falls back to `~/.local/bin` and prints a PATH instruction.

Rerunning install is idempotent. It updates source files, preserves `~/.voicenotes/config.toml`, preserves `~/VoiceNotes/`, refreshes venv dependencies, and leaves Hammerspoon with exactly one `require("voicenotes")`.

## Uninstall

`uninstall.sh` removes:

- `~/.hammerspoon/voicenotes.lua`
- exactly the `require("voicenotes")` line from `~/.hammerspoon/init.lua`
- the PATH wrapper
- runtime files under `~/.voicenotes/run`
- app files under `~/.voicenotes/app`
- the local venv under `~/.voicenotes/venv`

It does not remove:

- Homebrew dependencies.
- the pulled Ollama model.
- `~/VoiceNotes/` recordings and outputs.

It prints what was left behind, including that `ollama rm qwen2.5:14b` can reclaim roughly 9GB if the user no longer wants the model.

## Privacy And Network Behavior

After installation and model downloads, the intended pipeline is fully local. Audio and text do not leave the machine.

V1 does not actively block network calls during execution. Robust macOS network isolation would add too much complexity for this personal tool. The privacy guarantee is implemented by choosing local tools and documenting the design clearly.

## Acceptance Criteria

V1 is complete when:

1. `voicenotes start` and `voicenotes stop` reliably record WAV audio through ffmpeg.
2. Hammerspoon `Cmd+\`` toggles recording end to end.
3. A stopped recording queues processing without terminal interaction.
4. Only one pipeline runs at a time.
5. Whisper transcribes with `task="transcribe"`, no forced single language, and the code-switching initial prompt.
6. Raw transcript, cleaned transcript, summary, logs, and session state are written to the session folder.
7. Cleanup preserves language choice and does not summarize or translate.
8. Summary output has exactly the required five sections in order.
9. Missing metadata becomes `not specified` rather than invented content.
10. Successful summaries open with `open -g`.
11. Failures preserve completed artifacts, write `error.log`, and notify the user.
12. `retry <session>` resumes from the first invalid or missing artifact.
13. Installer is idempotent and leaves manual work limited to first Hammerspoon launch plus macOS permission grants.
14. `doctor` verifies dependencies, models, config, Hammerspoon install, device enumeration, and a short recording.

## Deferred From V1

- Speaker diarization.
- GUI beyond Hammerspoon menu-bar state and notifications.
- Live transcription.
- Hosted sync.
- Homebrew tap.
- Prompt customization.
- Whisper model customization.
- Active network sandboxing.
- Backup rotation.
- A long-running daemon.
- Automatic summary repair.

## Sources Checked During Design

- Hugging Face model card for `mlx-community/whisper-large-v3-mlx`: https://huggingface.co/mlx-community/whisper-large-v3-mlx
