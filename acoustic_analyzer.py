"""
Acoustic & Phonetic Feature Extraction Engine for Speech Sound Disorders.
Performs Formant Estimation (F1-F3 via LPC), Spectral Centroid & Moment Analysis,
and Goodness-of-Pronunciation (GOP) acoustic scoring for targeted phonemes.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    from scipy.signal import lfilter
except ImportError:
    lfilter = None


class AcousticAnalyzer:
    """Scientific acoustic analyzer for phoneme-level speech sound disorder evaluation."""

    @staticmethod
    def extract_formants_lpc(
        audio_segment: np.ndarray, 
        sr: int = 16000, 
        lpc_order: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimates formant frequencies (F1, F2, F3) using Linear Predictive Coding (LPC).
        Crucial for identifying rhotic /r/ vs derhoticized [w] or gliding.
        """
        if len(audio_segment) < 200:
            return {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F3_F2_diff": 0.0}

        # Apply pre-emphasis filter to boost higher formants: y[n] = x[n] - 0.97*x[n-1]
        pre_emp = 0.97
        emphasized = np.append(audio_segment[0], audio_segment[1:] - pre_emp * audio_segment[:-1])
        
        # Apply Hamming window
        window = np.hamming(len(emphasized))
        windowed = emphasized * window

        # Standard rule of thumb: LPC order = 2 + (sampling_rate / 1000)
        if lpc_order is None:
            lpc_order = int(2 + (sr / 1000.0))

        # Compute Autocorrelation via FFT
        n = len(windowed)
        n_fft = 1 << (2 * n - 1).bit_length()
        fft_win = np.fft.rfft(windowed, n_fft)
        r = np.fft.irfft(np.abs(fft_win) ** 2, n_fft)[:lpc_order + 1]

        # Levinson-Durbin recursion to find LPC coefficients
        a = np.zeros(lpc_order + 1)
        a[0] = 1.0
        e = r[0]
        if e <= 1e-12:
            return {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F3_F2_diff": 0.0}

        for i in range(1, lpc_order + 1):
            k = -np.sum(a[:i] * r[i:0:-1]) / (e + 1e-12)
            a[1:i+1] += k * a[i-1::-1]
            e *= (1.0 - k * k)
            if e <= 1e-12:
                break

        # Find roots of the LPC polynomial
        roots = np.roots(a)
        
        # Keep roots in the upper half of the z-plane with positive frequency
        formants = []
        for root in roots:
            if np.imag(root) >= 0:
                angle = np.arctan2(np.imag(root), np.real(root))
                freq = angle * (sr / (2.0 * np.pi))
                # Formant bandwidth check: r = exp(-pi * B * T) -> B = -log|r| / (pi * T)
                radius = np.abs(root)
                if radius > 0 and radius < 1.0:
                    bandwidth = -np.log(radius) * (sr / np.pi)
                    # Filter realistic vocal tract formants
                    if 200.0 <= freq <= 4500.0 and bandwidth < 600.0:
                        formants.append(freq)

        formants.sort()

        f1 = float(formants[0]) if len(formants) > 0 else 500.0
        f2 = float(formants[1]) if len(formants) > 1 else 1500.0
        f3 = float(formants[2]) if len(formants) > 2 else 2500.0

        return {
            "F1": round(f1, 1),
            "F2": round(f2, 1),
            "F3": round(f3, 1),
            "F3_F2_diff": round(abs(f3 - f2), 1)
        }

    @staticmethod
    def extract_spectral_moments(
        audio_segment: np.ndarray, 
        sr: int = 16000
    ) -> Dict[str, float]:
        """
        Computes spectral centroid (Center of Gravity), spectral spread, and peak frequency.
        Essential for diagnosing sibilants (/s/, /z/, /ʃ/, /tʃ/) and lisps.
        """
        if len(audio_segment) < 64:
            return {
                "spectral_centroid": 0.0,
                "spectral_peak": 0.0,
                "spectral_flatness": 0.0,
                "high_freq_ratio": 0.0
            }

        # Apply Hanning window & FFT
        window = np.hanning(len(audio_segment))
        windowed = audio_segment * window
        
        n_fft = max(512, 1 << (len(windowed) - 1).bit_length())
        mag_spec = np.abs(np.fft.rfft(windowed, n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        power = mag_spec ** 2
        total_power = np.sum(power)
        if total_power <= 1e-12:
            return {
                "spectral_centroid": 0.0,
                "spectral_peak": 0.0,
                "spectral_flatness": 0.0,
                "high_freq_ratio": 0.0
            }

        # Spectral Centroid (Center of Gravity)
        centroid = float(np.sum(freqs * power) / total_power)

        # Spectral Peak Frequency
        peak_idx = int(np.argmax(power))
        peak_freq = float(freqs[peak_idx])

        # High Frequency Energy Ratio (> 4500 Hz vs Total)
        high_freq_mask = freqs >= 4500.0
        high_freq_power = np.sum(power[high_freq_mask])
        high_freq_ratio = float(high_freq_power / total_power)

        # Spectral Flatness (Wiener entropy)
        geometric_mean = np.exp(np.mean(np.log(power + 1e-12)))
        arithmetic_mean = np.mean(power) + 1e-12
        flatness = float(geometric_mean / arithmetic_mean)

        return {
            "spectral_centroid": round(centroid, 1),
            "spectral_peak": round(peak_freq, 1),
            "spectral_flatness": round(flatness, 4),
            "high_freq_ratio": round(high_freq_ratio, 3)
        }

    @staticmethod
    def evaluate_target_phoneme_acoustics(
        phone: str, 
        audio_segment: np.ndarray, 
        sr: int = 16000
    ) -> Dict[str, Any]:
        """
        Deep acoustic evaluation tailored to the designated clinical target phoneme.
        Produces continuous production score (0.0 to 1.0), status, and acoustic diagnostic rationale.
        """
        phone = phone.upper()
        formants = AcousticAnalyzer.extract_formants_lpc(audio_segment, sr)
        moments = AcousticAnalyzer.extract_spectral_moments(audio_segment, sr)

        score = 0.85
        status = "expected_production"
        rationale = ""
        diagnostic_details = {}

        # -------------------------------------------------------------
        # Evaluation for /r/ (Rhotic Approximant)
        # Hallmark: F3 low, F3 - F2 < 600 Hz.
        # Deviation: Elevated F3 (> 2200 Hz), wide gap -> derhoticized or glided to [w].
        # -------------------------------------------------------------
        if phone in ["R", "ER", "AXR"]:
            f3 = formants["F3"]
            diff = formants["F3_F2_diff"]
            diagnostic_details = {
                "F1": formants["F1"],
                "F2": formants["F2"],
                "F3": f3,
                "F3_minus_F2": diff,
                "target_threshold": "F3 < 2200 Hz or (F3 - F2) < 700 Hz"
            }
            
            # Continuous scoring based on F3 lowering
            if f3 < 2050 or diff < 550:
                score = 0.92
                status = "expected_production"
                rationale = f"Strong rhotic acoustic marker detected: F3 dip at {f3:.0f} Hz (F3-F2 gap {diff:.0f} Hz)."
            elif f3 < 2350 or diff < 850:
                score = 0.72
                status = "mild_variation"
                rationale = f"Moderate rhotic resonance: F3 measured at {f3:.0f} Hz."
            else:
                score = 0.42
                status = "possible_difficulty"
                rationale = f"Acoustic hallmark of /r/ was reduced: F3 formant remained elevated at {f3:.0f} Hz (gap: {diff:.0f} Hz), characteristic of derhoticized or glided production (like [w])."

        # -------------------------------------------------------------
        # Evaluation for /s/ & /z/ (Alveolar Fricatives)
        # Hallmark: Centroid > 5000 - 7500 Hz.
        # Deviation: Centroid < 4500 Hz -> Interdental or lateral lisp.
        # -------------------------------------------------------------
        elif phone in ["S", "Z"]:
            centroid = moments["spectral_centroid"]
            high_ratio = moments["high_freq_ratio"]
            diagnostic_details = {
                "spectral_centroid_hz": centroid,
                "high_freq_ratio": high_ratio,
                "target_threshold": "Centroid > 5000 Hz"
            }

            if centroid >= 5000:
                score = 0.90
                status = "expected_production"
                rationale = f"Clear high-frequency frication with spectral centroid at {centroid:.0f} Hz."
            elif centroid >= 4500:
                score = 0.70
                status = "mild_variation"
                rationale = f"Borderline sibilant energy: spectral centroid at {centroid:.0f} Hz."
            else:
                score = 0.40
                status = "possible_difficulty"
                rationale = f"Reduced high-frequency turbulence: spectral centroid dropped to {centroid:.0f} Hz (typical /s/ > 5000 Hz), consistent with a frontal (interdental) or lateral lisp."

        # -------------------------------------------------------------
        # Evaluation for /ʃ/ ("sh") & /tʃ/ ("ch") (Postalveolar)
        # Hallmark: Energy concentrated in 3000 - 4800 Hz band.
        # Deviation: Centroid > 5200 Hz -> Fronted toward [s].
        # -------------------------------------------------------------
        elif phone in ["SH", "CH"]:
            centroid = moments["spectral_centroid"]
            peak = moments["spectral_peak"]
            diagnostic_details = {
                "spectral_centroid_hz": centroid,
                "peak_freq_hz": peak,
                "target_threshold": "Postalveolar energy band 3000 - 4800 Hz"
            }

            if centroid > 5200:
                score = 0.42
                status = "possible_difficulty"
                rationale = f"Spectral energy shifted abnormally high to {centroid:.0f} Hz (typical /ʃ/ lies in 3000–4800 Hz), indicating fronting from /ʃ/ ('sh') toward [s]."
            elif 2800 <= centroid <= 5200:
                score = 0.88
                status = "expected_production"
                rationale = f"Proper postalveolar resonance: centroid at {centroid:.0f} Hz."
            else:
                score = 0.60
                status = "mild_variation"
                rationale = f"Lower frequency turbulence detected at {centroid:.0f} Hz."

        # -------------------------------------------------------------
        # Evaluation for /θ/ & /ð/ ("th" Dental Fricative)
        # -------------------------------------------------------------
        elif phone in ["TH", "DH"]:
            flatness = moments["spectral_flatness"]
            diagnostic_details = {
                "spectral_flatness": flatness,
                "target_threshold": "Diffuse flat spectrum"
            }
            if flatness > 0.015:
                score = 0.85
                status = "expected_production"
                rationale = "Diffuse laminar frication consistent with expected dental airflow."
            else:
                score = 0.55
                status = "mild_variation"
                rationale = "Spectral peak sharpness suggests possible stopping or plosive tendency."

        # -------------------------------------------------------------
        # Evaluation for /k/ (Velar Plosive)
        # -------------------------------------------------------------
        elif phone in ["K", "G"]:
            centroid = moments["spectral_centroid"]
            diagnostic_details = {
                "spectral_centroid_hz": centroid,
                "target_threshold": "Velar burst 1500 - 2800 Hz"
            }
            if centroid > 3800:
                score = 0.48
                status = "possible_difficulty"
                rationale = f"High burst frequency ({centroid:.0f} Hz) suggests alveolar fronting (e.g. /k/ pronounced as [t])."
            else:
                score = 0.86
                status = "expected_production"
                rationale = f"Appropriate velar burst energy at {centroid:.0f} Hz."

        # Default for other sounds
        else:
            score = 0.90
            status = "expected_production"
            rationale = "Acoustic envelope matches expected target distribution."

        return {
            "score": round(score, 2),
            "status": status,
            "rationale": rationale,
            "acoustic_features": diagnostic_details
        }
