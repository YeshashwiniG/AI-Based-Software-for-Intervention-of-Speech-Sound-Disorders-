"""Sound-specific rating prediction using aligned whole-word audio crops.

SpeechOcean762 supplies ordered phone annotations and phone ratings but no
phone-level timestamps. Predictions here are consequently sound-conditioned
whole-word rating predictions, not phoneme alignment or GOP.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

import joblib
import numpy as np

from backend.feature_extractor import PhonemeFeatureExtractor
from backend.phoneme_dictionary import ARPABET_TO_IPA, get_word_phonemes
from backend.word_alignment import align_words
from scripts.train_phoneme_model import PHONES, _phone_features

PHONE_DISPLAY = {"R": "r", "L": "l", "W": "w", "F": "f", "V": "v",
                 "S": "s", "Z": "z", "SH": "sh", "ZH": "zh",
                 "TH": "th", "DH": "th", "CH": "ch", "JH": "j",
                 "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g"}


class PhonemePronunciationAnalyzer:
    def __init__(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(root, "models", "phoneme_rating_model.joblib")
        self.config_path = os.path.join(root, "models", "phoneme_model_config.json")
        self.metrics_path = os.path.join(root, "models", "phoneme_training_metrics.json")
        self.model = None
        self.config = None
        self.metrics = None
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
                if os.path.isfile(self.metrics_path):
                    with open(self.metrics_path, encoding="utf-8") as f:
                        self.metrics = json.load(f)

    @staticmethod
    def _phone_id(phone: str) -> int | None:
        clean = "".join(c for c in str(phone).upper() if not c.isdigit())
        try:
            return PHONES.index(clean)
        except ValueError:
            return None

    def _supported(self) -> set[str]:
        if not self.metrics:
            return set()
        # A phone is surfaced only when the held-out speaker split has enough
        # positive and total examples, and the model beats a trivial F1 baseline.
        supported = set()
        for phone, result in self.metrics.get("per_phone_test_metrics", {}).items():
            dist = result.get("class_distribution", {})
            n_pos = int(dist.get("rating_below_2", 0))
            n_total = int(result.get("test_samples", 0))
            trivial_positive_f1 = (2 * n_pos / (n_total + n_pos)) if n_total else 0
            if n_total >= 20 and n_pos >= 5 and result.get("f1", 0) > trivial_positive_f1:
                supported.add(phone)
        return supported

    def analyze(self, audio: np.ndarray, sr: int, expected_text: str,
                recognized: dict[str, Any]) -> list[dict[str, Any]]:
        self.load()
        if self.model is None:
            raise RuntimeError("Sound-analysis model is not trained. Run scripts/train_phoneme_model.py.")
        expected_words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*", expected_text or "")
        matches = align_words(expected_words, recognized.get("words") or [])
        supported = self._supported()
        threshold = float(self.config["threshold"])
        results = []
        for word, mapping in zip(expected_words, matches):
            phones = get_word_phonemes(word)
            targets = []
            seen = set()
            for phone in phones:
                clean = "".join(c for c in phone.upper() if not c.isdigit())
                if clean in supported and clean not in seen:
                    seen.add(clean)
                    targets.append(clean)
            rec_index = mapping["recognized_index"]
            if rec_index is None:
                results.append({"expected_word": word, "asr_hypothesis": None,
                                "status": "unresolved", "sound_results": [],
                                "result": "Could not locate this word in the audio."})
                continue
            item = recognized["words"][rec_index]
            start_t, end_t = float(item["start"]), float(item["end"])
            start, end = max(0, int(start_t * sr)), min(len(audio), int(end_t * sr))
            if end - start < int(0.06 * sr):
                results.append({"expected_word": word, "asr_hypothesis": item.get("word"),
                                "start": start_t, "end": end_t,
                                "status": "unresolved", "sound_results": [],
                                "result": "Not enough aligned audio to analyze this word."})
                continue
            if not targets:
                results.append({"expected_word": word, "asr_hypothesis": item.get("word"),
                                "start": start_t, "end": end_t,
                                "status": "no_supported_sound", "sound_results": [],
                                "result": "No target sound in this word has sufficient held-out evidence yet."})
                continue
            word_audio = np.asarray(audio[start:end], dtype=np.float32)
            features = PhonemeFeatureExtractor.extract_recording_features(word_audio, sr)
            sound_results = []
            for phone in targets:
                vector = _phone_features(features, phone)
                proba = self.model.predict_proba([vector])[0]
                pos = list(self.model.classes_).index(1)
                positive = float(proba[pos]) >= threshold
                ipa = PHONE_DISPLAY.get(phone, ARPABET_TO_IPA.get(phone, phone.lower()))
                sound_results.append({"phone": phone, "sound": ipa,
                                      "status": "possible_issue" if positive else "no_issue_detected",
                                      "result": "Sound may need attention" if positive else "No possible issue detected for this sound"})
            flagged = [s for s in sound_results if s["status"] == "possible_issue"]
            results.append({
                "expected_word": word,
                "asr_hypothesis": str(item.get("word", "")).strip(),
                "text_match": bool(mapping["text_match"]),
                "start": round(start_t, 3), "end": round(end_t, 3),
                "status": "possible_issue" if flagged else "no_issue_detected",
                "sound_results": sound_results,
                "problematic_sounds": [s["sound"] for s in flagged],
                "result": "Possible pronunciation issue" if flagged else "No possible sound issue detected",
            })
        return results
