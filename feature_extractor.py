"""
Acoustic Feature Extraction Module for Machine Learning Training & Inference.
Extracts 50-dimensional acoustic feature vectors from an audio segment:
- MFCCs (mean & std across frames)
- Spectral Centroid, Flatness, Rolloff, ZCR
- Formants (F1, F2, F3, F3-F2 diff) via Linear Predictive Coding
- High-frequency energy ratio and RMS energy
- Phonetic duration & category encodings
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from backend.acoustic_analyzer import AcousticAnalyzer


# Target phone class mapping for categorical one-hot features
PHONE_FAMILIES = {
    "RHOTIC": ["R", "ER", "AXR"],
    "SIBILANT": ["S", "Z"],
    "POSTALVEOLAR": ["SH", "ZH", "CH", "JH"],
    "DENTAL": ["TH", "DH"],
    "VELAR": ["K", "G"],
    "PLOSIVE": ["P", "B", "T", "D"],
    "VOWEL": ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
}
FAMILY_KEYS = list(PHONE_FAMILIES.keys())


class PhonemeFeatureExtractor:
    """Extracts unified 50-dimensional acoustic feature vectors for ML models."""

    FEATURE_DIM = 50

    @staticmethod
    def extract_recording_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract one vector from a complete recording (not a claimed phone segment)."""
        return PhonemeFeatureExtractor.extract_features(
            audio_segment=audio, sr=sr, phone_symbol="", duration_sec=len(audio) / max(1, sr)
        )

    @staticmethod
    def extract_features(
        audio_segment: np.ndarray,
        sr: int = 16000,
        phone_symbol: str = "R",
        duration_sec: float = 0.15
    ) -> np.ndarray:
        """
        Extracts a dense 50-dimensional feature vector from an audio segment.
        Guaranteed fixed shape (50,) and finite values (no NaNs or Infs).
        """
        feats = np.zeros(PhonemeFeatureExtractor.FEATURE_DIM, dtype=np.float32)
        
        # Guard for empty or ultra-short segments
        if len(audio_segment) < 64:
            audio_segment = np.pad(audio_segment, (0, 64 - len(audio_segment)))

        # 1. Base energy & duration (dims 0..2)
        rms = float(np.sqrt(np.mean(audio_segment**2) + 1e-12))
        peak_amp = float(np.max(np.abs(audio_segment)))
        feats[0] = rms
        feats[1] = peak_amp
        feats[2] = float(duration_sec)

        # 2. Formant estimation via LPC (dims 3..6)
        formants = AcousticAnalyzer.extract_formants_lpc(audio_segment, sr)
        feats[3] = formants["F1"] / 1000.0  # Normalized to kHz
        feats[4] = formants["F2"] / 1000.0
        feats[5] = formants["F3"] / 1000.0
        feats[6] = formants["F3_F2_diff"] / 1000.0

        # 3. Spectral Moments & Centroid (dims 7..10)
        moments = AcousticAnalyzer.extract_spectral_moments(audio_segment, sr)
        feats[7] = moments["spectral_centroid"] / 1000.0  # kHz
        feats[8] = moments["spectral_peak"] / 1000.0
        feats[9] = moments["spectral_flatness"]
        feats[10] = moments["high_freq_ratio"]

        # 4. Zero-crossing rate (dim 11)
        zcr = np.mean(np.abs(np.diff(np.sign(audio_segment + 1e-12)))) / 2.0
        feats[11] = float(zcr)

        # 5. Mel-frequency Filterbank / MFCC Approximation (dims 12..37) - 13 means + 13 stds
        n_fft = max(256, 1 << (len(audio_segment) - 1).bit_length())
        window = np.hanning(len(audio_segment))
        mag_spec = np.abs(np.fft.rfft(audio_segment * window, n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Compute triangular mel-filterbank energies (13 filters from 100 Hz to 7500 Hz)
        n_mels = 13
        mel_min = 2595.0 * np.log10(1.0 + 100.0 / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + 7500.0 / 700.0)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fbank_energies = np.zeros(n_mels, dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if k < len(mag_spec) and (f_m - f_m_minus) > 0:
                    fbank_energies[m - 1] += (k - f_m_minus) / (f_m - f_m_minus) * mag_spec[k]
            for k in range(f_m, f_m_plus):
                if k < len(mag_spec) and (f_m_plus - f_m) > 0:
                    fbank_energies[m - 1] += (f_m_plus - k) / (f_m_plus - f_m) * mag_spec[k]

        # Log mel energies
        log_mels = np.log(fbank_energies + 1e-6)
        
        # Discrete Cosine Transform (DCT) -> MFCCs
        mfccs = np.zeros(n_mels, dtype=np.float32)
        for i in range(n_mels):
            mfccs[i] = np.sum(log_mels * np.cos(np.pi * i * (np.arange(n_mels) + 0.5) / n_mels))

        feats[12:25] = mfccs / 10.0  # Normalized MFCC coefficients
        feats[25:38] = log_mels / 10.0  # Normalized Log-Mel energies

        # 6. Spectral Contrast across 5 octave bands (dims 38..42)
        bands = [(100, 500), (500, 1500), (1500, 3000), (3000, 5000), (5000, 7500)]
        for b_idx, (low, high) in enumerate(bands):
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_p = mag_spec[mask]
                contrast = float(np.max(band_p) - np.min(band_p)) / (float(np.mean(band_p)) + 1e-6)
                feats[38 + b_idx] = min(10.0, contrast) / 10.0
            else:
                feats[38 + b_idx] = 0.0

        # 7. One-hot phoneme family encoding (dims 43..49) -> 7 dims
        clean_p = "".join([c for c in phone_symbol.upper() if not c.isdigit()])
        for f_idx, family in enumerate(FAMILY_KEYS):
            if clean_p in PHONE_FAMILIES[family]:
                feats[43 + f_idx] = 1.0

        # Replace any potential NaN or Inf
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        return feats
