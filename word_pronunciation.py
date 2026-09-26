"""Word-level inference using the separately trained Speechocean annotation model."""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

import joblib
import numpy as np

from backend.feature_extractor import PhonemeFeatureExtractor
from backend.word_alignment import align_words


class WordPronunciationAnalyzer:
    def __init__(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(root, "models", "word_pronunciation_model.joblib")
        self.config_path = os.path.join(root, "models", "word_model_config.json")
        self.model = None
        self.config = None
        self._lock = threading.Lock()
        self.load()

    def load(self):
        if self.model is not None:
            return
        with self._lock:
            if self.model is None and os.path.isfile(self.model_path):
                self.model = joblib.load(self.model_path)
                with open(self.config_path, encoding="utf-8") as f:
                    self.config = json.load(f)

    @staticmethod
    def expected_tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*", text or "")

    def analyze(self, audio: np.ndarray, sr: int, expected_text: str,
                recognized: dict[str, Any]) -> list[dict[str, Any]]:
        self.load()
        if self.model is None:
            raise RuntimeError("The word-level model is not trained. Run scripts/train_word_model.py.")
        expected = self.expected_tokens(expected_text)
        matches = align_words(expected, recognized.get("words") or [])
        results = []
        for expected_word, mapping in zip(expected, matches):
            recognized_index = mapping["recognized_index"]
            if recognized_index is None:
                results.append({"expected_word": expected_word, "status": "unresolved",
                                "result": "Could not locate this word in the audio.",
                                "feedback": "Please record the complete expected sentence again.",
                                "asr_hypothesis": None, "start": None, "end": None,
                                "text_match": False})
                continue
            item = recognized["words"][recognized_index]
            start = max(0, int(float(item["start"]) * sr))
            end = min(len(audio), int(float(item["end"]) * sr))
            if end - start < int(0.06 * sr):
                results.append({"expected_word": expected_word, "status": "unresolved",
                                "result": "Not enough aligned audio to analyze this word.",
                                "feedback": "Try recording the sentence again with a short pause between words.",
                                "asr_hypothesis": item.get("word"), "start": item.get("start"),
                                "end": item.get("end"), "text_match": mapping["text_match"]})
                continue
            clip = np.asarray(audio[start:end], dtype=np.float32)
            features = PhonemeFeatureExtractor.extract_recording_features(clip, sr)
            probabilities = self.model.predict_proba([features])[0]
            pos = list(self.model.classes_).index(1)
            # Keep the score internal; the interface presents only the model result.
            flagged = float(probabilities[pos]) >= float(self.config.get("decision_threshold", 0.5))
            results.append({
                "expected_word": expected_word,
                "asr_hypothesis": str(item.get("word", "")).strip(),
                "status": "possible_issue" if flagged else "no_issue_detected",
                "result": "Possible pronunciation issue" if flagged else "No possible word-level issue detected",
                "feedback": (f"Practice saying {expected_word} slowly and clearly." if flagged
                             else "Continue practicing the sentence."),
                "text_match": bool(mapping["text_match"]),
                "start": round(float(item["start"]), 3),
                "end": round(float(item["end"]), 3),
                "_internal_score": float(probabilities[pos]),
            })
        return results
