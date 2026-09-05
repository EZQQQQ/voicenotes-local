# VoiceNotes Local

A local macOS voice notes tool. Press one hotkey to record, press it again to stop, then get a Markdown summary in `~/VoiceNotes`.

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

The installer sets up ffmpeg, Hammerspoon, Ollama, Python dependencies, and local models. Model downloads are about 12GB.

## Permissions

Open Hammerspoon once after install. In System Settings, grant:

- Accessibility
- Microphone

## Hotkey

```text
Cmd+`
```

If it does not trigger, clear this macOS shortcut:

```text
System Settings > Keyboard > Keyboard Shortcuts > Keyboard > Move focus to next window
```

Keep Hammerspoon running. The hotkey cannot work when Hammerspoon is closed.

## Menu Bar

The `VN` menu has Start or Stop, Open VoiceNotes Folder, and Quit. Quit hides `VN` but keeps Hammerspoon running so the hotkey can still work.

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

Use `voicenotes devices` to find an exact `audio_device` name. `auto_open` controls whether `summary.md` opens after processing.

Whisper stays fixed to large-v3 for English and Mandarin code-switching accuracy. The Ollama model is configurable for different Mac memory sizes. VoiceNotes opens Ollama automatically during processing if needed.

## Commands

```bash
voicenotes devices
voicenotes status --json
voicenotes doctor
voicenotes retry ~/VoiceNotes/2026-08-27_143012
voicenotes record-test --duration 10
```

Use `voicenotes doctor` after install. Use `voicenotes retry <session>` after a failed note.

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

`audio.wav` is used for transcription. `audio.m4a` is for playback.

`summary.md` contains:

- `## Summary`
- `## Discussion by topic`
- `## Feedback & critique`
- `## Decisions`
- `## Action items`
- `## Blockers & open questions`
- `## Next steps`
