import sys
from types import SimpleNamespace

from voicenotes.transcriber import transcribe


def test_transcribe_disables_previous_text_conditioning_without_translating(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    model = tmp_path / "model"

    def recognize(path, **options):
        assert path == str(audio)
        assert options["path_or_hf_repo"] == str(model)
        assert options.get("condition_on_previous_text", True) is False
        assert options["task"] == "transcribe"
        assert options["language"] is None
        return {"segments": [{"start": 0, "end": 2, "text": "保留 API names"}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=recognize))

    assert transcribe(audio, model) == [{"start": 0.0, "end": 2.0, "text": "保留 API names"}]
