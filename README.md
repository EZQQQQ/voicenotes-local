# VoiceNotes Local

A lightweight, local-first voice notes tool for Apple Silicon Macs. Press one hotkey to record, press it again to stop, and turn spoken notes or conversations into timestamped transcripts and structured Markdown notes in `~/VoiceNotes`.

VoiceNotes combines Hammerspoon recording controls, Whisper large-v3 through MLX for transcription, and Ollama for transcript cleanup and summarization. It focuses on English and Mandarin, including speech that switches between them, while keeping the original recording and raw transcript available for reference.

```mermaid
flowchart LR
    A[Record audio] --> B[Whisper transcription]
    B --> C[Raw transcript]
    C --> D[Chunked cleanup]
    D --> E[Markdown summary]
```

Cleanup works in bounded, paragraph-preserving chunks and rejects responses with missing timestamps or heavily shortened segments. Repeated filler-only runs are reduced in cleanup input; the raw transcript stays unchanged. Processing runs after recording stops, and failed stages can be retried without starting over.

## Requirements

- Apple Silicon Mac with 16GB memory or more
- Homebrew
- GitHub CLI access to this private repo

## Install

```bash
brew install gh
gh auth login
gh auth setup-git
tmp="$(mktemp -d)" && gh repo clone ezqqqq/voicenotes-local "$tmp/voicenotes-local" && VOICENOTES_REPO_URL="https://github.com/ezqqqq/voicenotes-local.git" "$tmp/voicenotes-local/install.sh"
```

The installer sets up ffmpeg, Hammerspoon, Ollama, Python dependencies, and local models. Model downloads are about 12GB. Run `voicenotes doctor` afterward to check dependencies, models, recording and Hammerspoon integration.

## Everyday Use

Keep Hammerspoon running. Press the default hotkey once to start recording and again to stop:

```text
Cmd+`
```

If it does not trigger, clear this macOS shortcut:

```text
System Settings > Keyboard > Keyboard Shortcuts > Keyboard > Move focus to next window
```

The `VN` menu also offers Start or Stop and Open VoiceNotes Folder. Quit hides the menu but leaves Hammerspoon and the hotkey running. Successful summaries open automatically unless `auto_open` is disabled.

## Config

Edit `~/.voicenotes/config.toml`:

```toml
output_root = "~/VoiceNotes"
audio_device = "default"
ollama_model = "qwen2.5:14b"
auto_open = true

[hotkey]
mods = ["cmd"]
key = "`"
```

- `output_root`: where recordings and notes are saved in per-session folders.
- `audio_device`: `default` selects audio input index 0; use `voicenotes devices` to find an exact device name.
- `ollama_model`: the installed local model used for cleanup and summaries.
- `auto_open`: whether completed summaries open automatically.
- `[hotkey]`: the Hammerspoon modifiers and key.

Whisper stays fixed to large-v3 for English and Mandarin code-switching accuracy. The Ollama model is configurable for different Mac memory budgets; the default is `qwen2.5:14b`. Install a replacement with `ollama pull <model>` before changing the setting. VoiceNotes opens Ollama automatically during processing if needed.

## Commands

```bash
voicenotes devices
voicenotes status --json
voicenotes doctor
voicenotes retry ~/VoiceNotes/2026-08-27_143012
voicenotes record-test --duration 10
```

`voicenotes retry <session>` resumes from the first missing or invalid artifact, reusing valid earlier results. To replace an existing cleanup and summary while retaining the raw transcript and audio:

```bash
voicenotes retry ~/VoiceNotes/2026-08-27_143012 --from-clean
```

If processing fails, check `error.log` and `pipeline.log` in the session folder, or use `voicenotes status --json` for the current status.

## Output

```text
~/VoiceNotes/
  2026-08-27_143012/
    audio.wav
    audio.m4a
    ffmpeg.log
    transcript_raw.md
    transcript_clean.md
    summary.md
```

- `audio.wav` is the original recording used for transcription; `audio.m4a` is a best-effort playback copy.
- `transcript_raw.md` preserves timestamped speech-recognition output.
- `transcript_clean.md` contains the validated cleanup output.
- `summary.md` organizes the cleaned text into the sections below.

`summary.md` contains:

- `## Summary`
- `## Discussion by topic`
- `## Feedback & critique`
- `## Decisions`
- `## Action items`
- `## Blockers & open questions`
- `## Next steps`

## Privacy and Limitations

After the initial downloads, the default pipeline records, transcribes and generates notes locally, without a hosted VoiceNotes service or cloud inference. Network access is not sandboxed; this is a local-processing design, not an enforced offline environment.

Long or noisy recordings can still lose details during cleanup and summarization, even when processing succeeds. Cleanup is chunked, but summarization currently uses a single model request. Verify important names, numbers and decisions against `transcript_raw.md` and the recording.

## Further Reference

For a more comprehensive speech-transcription project, see [WhisperX](https://github.com/m-bain/whisperX). Its documentation covers word-level alignment, speaker diarization and batched inference. It is a useful reference for those workflows; VoiceNotes uses its own smaller, MLX-based transcription pipeline.

## Uninstall

Run `./uninstall.sh` from the repository checkout. It removes the app, runtime files, Python environment, command wrapper and Hammerspoon integration, while retaining your recordings, config, downloaded models and Homebrew dependencies.
