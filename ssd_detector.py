"""Recording-level inference for the valid Speechocean762 rating task.

This intentionally does not manufacture word/phone timestamps: the supplied
dataset contains no phone boundaries, so the trained model has whole-recording
features and labels.
"""
import json
import os
import time

import joblib
import numpy as np

from backend.audio_processor import AudioProcessor
from backend.feature_extractor import PhonemeFeatureExtractor
from backend.phoneme_dictionary import get_word_phonemes, ARPABET_TO_IPA, is_clinical_target


class SpeechSoundDisorderDetector:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.load_trained_model()

    def load_trained_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "ssd_ml_model.joblib")
        metrics_path = os.path.join(base_dir, "models", "training_metrics.json")
        if os.path.isfile(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as exc:
                print(f"[speech model] Could not load saved model: {exc}")
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = None

    @staticmethod
    def _candidate_words_and_sounds(text):
        candidates = []
        for raw in (text or "").split():
            word = raw.strip(".,!?:;\"'()[]{}").lower()
            if not word:
                continue
            sounds = ["/" + ARPABET_TO_IPA.get(phone, phone.lower()) + "/"
                      for phone in get_word_phonemes(word) if is_clinical_target(phone)]
            if sounds:
                candidates.append({"word": word, "sounds_in_text": sorted(set(sounds))})
        return candidates

    def analyze_continuous_speech(self, audio, sr=16000, prompt_text=None):
        started = time.time()
        if self.model is None:
            self.load_trained_model()
        if self.model is None:
            raise RuntimeError("No trained model is available. Run `python scripts/train_model.py` first.")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < int(sr * 0.25):
            raise ValueError("Please provide at least 0.25 seconds of speech audio.")
        speech_regions = AudioProcessor.detect_voice_activity(audio, sr)
        features = PhonemeFeatureExtractor.extract_recording_features(audio, sr)
        probs = self.model.predict_proba([features])[0]
        positive_idx = list(self.model.classes_).index(1)
        score = float(probs[positive_idx])
        has_lower_rating = bool(self.model.predict([features])[0] == 1)
        if has_lower_rating:
            result_text = "Possible pronunciation difficulty detected."
            feedback = "Practice the complete sentence slowly and clearly. This model cannot identify a specific word or sound to target."
            summary = ("The trained model predicts this complete recording is more consistent with recordings "
                       "that contain at least one phone rating below 2.0 in Speechocean762. This is a screening-style "
                       "dataset rating result, not a diagnosis.")
        else:
            result_text = "No lower-rated pronunciation pattern detected by this recording-level model."
            feedback = "Keep practicing the complete sentence. This result does not assess individual words or sounds."
            summary = ("The trained model predicts this complete recording is more consistent with recordings "
                       "whose phone ratings were all 2.0 in Speechocean762. This is not a clinical conclusion.")
        return {
            "sentence": prompt_text or "",
            "audio_duration_seconds": round(len(audio) / sr, 2),
            "processing_time_seconds": round(time.time() - started, 3),
            "overall_summary": summary,
            "analysis_scope": "recording_level_only",
            "recording_result": result_text,
            "feedback": feedback,
            "model_class": "contains_rating_below_2.0" if has_lower_rating else "all_ratings_2.0",
            "model_score": round(score, 4),
            "score_note": "Uncalibrated model score for the recording-level label; not a clinical probability.",
            "metrics": {"speech_regions_detected": len(speech_regions), "target_sounds_localized": 0},
            "speech_regions": [[float(a), float(b)] for a, b in speech_regions],
            "candidate_words_and_sounds_from_text": self._candidate_words_and_sounds(prompt_text or ""),
            "flagged_difficulties": [],
            "words": [],
            "waveform_envelope": AudioProcessor.compute_waveform_envelope(audio, points=120),
            "ml_model_info": self.metrics,
        }
