"""Local English ASR with Whisper word-level timestamps."""
import os
import threading
from typing import Any

import numpy as np


class SpeechRecognizer:
    """Lazy-load a local faster-whisper model and return ordered timed words."""

    def __init__(self):
        self.model_name = os.getenv("SPEECH_RECOGNITION_MODEL", "base.en")
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from faster_whisper import WhisperModel
                    except ImportError as exc:
                        raise RuntimeError(
                            "Speech recognition is not installed. Install requirements.txt first."
                        ) from exc
                    self._model = WhisperModel(
                        self.model_name,
                        device="cpu",
                        compute_type="int8",
                    )
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   beam_size: int = 5) -> dict[str, Any]:
        if sample_rate != 16000:
            raise ValueError("Audio must be resampled to 16 kHz before recognition.")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size < int(sample_rate * 0.25):
            raise ValueError("Please provide at least 0.25 seconds of audio.")
        if not np.isfinite(samples).all():
            raise ValueError("Audio contains invalid sample values.")

        model = self._get_model()
        segments, info = model.transcribe(
            samples,
            language="en",
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        words = []
        transcript_parts = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                transcript_parts.append(text)
            for item in segment.words or []:
                token = (item.word or "").strip()
                if not token:
                    continue
                words.append({
                    "word": token,
                    "start": round(float(item.start), 3),
                    "end": round(float(item.end), 3),
                })

        return {
            "transcript": " ".join(transcript_parts),
            "words": words,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "duration_seconds": round(float(info.duration), 3),
            "model": self.model_name,
        }
