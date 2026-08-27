from __future__ import annotations

import json
from pathlib import Path
import sys


def transcribe(audio: Path, model_dir: Path) -> list[dict[str, object]]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=str(model_dir),
        task="transcribe",
        language=None,
        initial_prompt="This recording mixes English and Mandarin Chinese, sometimes switching mid-sentence.",
    )
    return [
        {"start": float(segment["start"]), "end": float(segment["end"]), "text": str(segment["text"])}
        for segment in result["segments"]
    ]


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m voicenotes.transcriber <audio.wav> <model_dir>", file=sys.stderr)
        return 2
    print(json.dumps(transcribe(Path(args[0]), Path(args[1])), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
