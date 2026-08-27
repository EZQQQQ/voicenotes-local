# VoiceNotes Local

## What It Does

VoiceNotes Local records spoken notes from a global Hammerspoon hotkey, transcribes them locally with Whisper large-v3, cleans likely ASR mistakes with a local Ollama model, and writes a structured Markdown summary. It is a personal macOS tool with no server, account, billing, or cloud inference.

## Requirements

- Apple Silicon Mac.
- 16GB unified memory or more.
- Homebrew.
- Hammerspoon.
- ffmpeg.
- Ollama.
- Python 3.11+.

The installer checks or installs the software dependencies. macOS permission prompts still require your approval.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ezqqqq/voicenotes-local/main/install.sh | bash
```

For local testing before the public repo URL is final:

```bash
VOICENOTES_REPO_URL=/path/to/local/repo ./install.sh
```

The installer downloads local models. The combined download is roughly 12GB.

## Permissions

Launch Hammerspoon once after install. Then grant:

- Accessibility to Hammerspoon.
- Microphone to Hammerspoon for hotkey-triggered recording.
- Microphone to Terminal if you use `voicenotes record-test` from a shell.

Check these in System Settings > Privacy & Security > Accessibility and System Settings > Privacy & Security > Microphone.

## Hotkey

The default hotkey is `Cmd+``. On a second Mac, this may conflict with the macOS shortcut at System Settings > Keyboard > Keyboard Shortcuts > Keyboard > Move focus to next window. Clear or change that shortcut if VoiceNotes does not receive the keypress.

The hotkey is configurable in `~/.voicenotes/config.toml`. A Hyper-key alternative is recommended if `Cmd+`` is still useful for window focus on that Mac.

## Config

User config lives at `~/.voicenotes/config.toml` and exposes only these fields:

```toml
output_root = "~/VoiceNotes"
audio_device = "default"
ollama_model = "qwen2.5:14b"
auto_open = true

[hotkey]
mods = ["cmd"]
key = "`"
```

- `output_root`: where session folders are written.
- `hotkey`: Hammerspoon modifiers and key.
- `audio_device`: `default` for avfoundation index 0 at record time, or an exact device name from `voicenotes devices`.
- `ollama_model`: local Ollama model used for cleanup and summarization.
- `auto_open`: whether successful summaries open with `open -g`.

VoiceNotes opens the Ollama app automatically during processing if the local API is not already running. The configured model still must be installed locally.

## Why Whisper Model Is Fixed And Ollama Model Is Configurable

Whisper model is fixed to large-v3 because English/Mandarin code-switching accuracy is the central product requirement. Changing it is likely to degrade the core behavior.

Ollama model is configurable because different Macs have different memory budgets. The default is `qwen2.5:14b`; a 32GB Mac may choose a larger local model.

## Commands

```bash
voicenotes devices
voicenotes status --json
voicenotes doctor
voicenotes retry ~/VoiceNotes/2026-08-27_143012
voicenotes record-test --duration 10
```

Use `voicenotes retry <session>` when transcription, cleanup, or summarization fails after some artifacts were already written. Retry skips valid existing artifacts and resumes from the first missing or invalid one.

Run `voicenotes doctor` after install to check dependencies, local models, Hammerspoon integration, audio devices, and microphone behavior.

## Output

Sessions are written as flat folders:

```text
~/VoiceNotes/
  2026-08-27_143012/
    audio.wav
    audio.m4a
    ffmpeg.log
    transcript_raw.md
    transcript_clean.md
    summary.md
    pipeline.log
    error.log
    session.json
```

`audio.wav` is the canonical recording used for transcription because PCM WAV is robust if recording is interrupted. `audio.m4a` is a best-effort listening copy for Finder, QuickTime, and other macOS apps.

`summary.md` contains exactly:

- `## Meeting Metadata`
- `## Key Discussion Points`
- `## Decisions Made`
- `## Action Items`
- `## Open Questions`

## Uninstall

From the repo checkout:

```bash
./uninstall.sh
```

The uninstaller removes the VoiceNotes app files, runtime files, venv, command wrapper, and Hammerspoon integration. It does not remove Homebrew dependencies, Ollama models, or `~/VoiceNotes`. To remove the default Ollama model later, run:

```bash
ollama rm qwen2.5:14b
```

## Privacy

After initial model downloads, recording, transcription, cleanup, and summarization are local. Audio and text are not sent to a hosted VoiceNotes service because there is no service.

V1 does not actively sandbox network access. The privacy design is based on local tools and local model execution.
