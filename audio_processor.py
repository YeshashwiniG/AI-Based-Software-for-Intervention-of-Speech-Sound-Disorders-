"""
Audio Processing & VAD (Voice Activity Detection) Module.
Handles 16 kHz mono resampling, silence trimming, waveform normalization,
and energy envelope computation for continuous speech streams.
"""

import io
import math
import numpy as np
from typing import Tuple, List, Dict, Any

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.io import wavfile
    from scipy import signal
except ImportError:
    wavfile = None
    signal = None


class AudioProcessor:
    """Robust audio loader, resampler, and preprocessor for speech analysis."""
    
    TARGET_SR = 16000  # Standard sample rate for speech & phoneme models
    
    @staticmethod
    def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Loads raw audio bytes (WAV, WEBM, OGG, etc.) into a float32 numpy array normalized to [-1.0, 1.0].
        Converts multi-channel to mono and resamples to 16,000 Hz.
        """
        # Attempt 1: soundfile
        if sf is not None:
            try:
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                return AudioProcessor.normalize_and_resample(data, sr)
            except Exception:
                pass
        
        # Attempt 2: scipy.io.wavfile
        if wavfile is not None:
            try:
                sr, raw_data = wavfile.read(io.BytesIO(audio_bytes))
                if raw_data.dtype == np.int16:
                    data = raw_data.astype(np.float32) / 32768.0
                elif raw_data.dtype == np.int32:
                    data = raw_data.astype(np.float32) / 2147483648.0
                elif raw_data.dtype == np.uint8:
                    data = (raw_data.astype(np.float32) - 128.0) / 128.0
                else:
                    data = raw_data.astype(np.float32)
                return AudioProcessor.normalize_and_resample(data, sr)
            except Exception:
                pass
                
        raise ValueError("Could not decode audio bytes. Ensure valid WAV/audio format.")

    @staticmethod
    def normalize_and_resample(audio: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
        """Converts stereo to mono and resamples to 16,000 Hz."""
        # Convert multi-channel to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        # Peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val
            
        # Resample to 16,000 Hz if needed
        if orig_sr != AudioProcessor.TARGET_SR:
            if signal is not None:
                num_target_samples = int(round(len(audio) * float(AudioProcessor.TARGET_SR) / orig_sr))
                audio = signal.resample(audio, num_target_samples)
            else:
                # Linear interpolation fallback
                duration = len(audio) / orig_sr
                target_times = np.linspace(0, duration, int(duration * AudioProcessor.TARGET_SR), endpoint=False)
                orig_times = np.linspace(0, duration, len(audio), endpoint=False)
                audio = np.interp(target_times, orig_times, audio).astype(np.float32)
                
        return audio.astype(np.float32), AudioProcessor.TARGET_SR

    @staticmethod
    def detect_voice_activity(
        audio: np.ndarray, 
        sr: int = 16000, 
        frame_ms: int = 20, 
        energy_threshold_db: float = -38.0
    ) -> List[Tuple[float, float]]:
        """
        Energy and zero-crossing based Voice Activity Detection (VAD).
        Returns continuous speech intervals [(start_sec, end_sec), ...].
        """
        frame_len = int(sr * (frame_ms / 1000.0))
        hop_len = frame_len // 2
        
        num_frames = (len(audio) - frame_len) // hop_len + 1
        if num_frames <= 0:
            return [(0.0, len(audio) / sr)]
            
        energies = []
        for i in range(num_frames):
            frame = audio[i * hop_len : i * hop_len + frame_len]
            rms = np.sqrt(np.mean(frame**2) + 1e-12)
            db = 20.0 * np.log10(rms + 1e-12)
            energies.append(db)
            
        energies = np.array(energies)
        max_energy = np.max(energies)
        threshold = max(max_energy + energy_threshold_db, -45.0)
        
        is_speech = energies > threshold
        
        # Smooth with morphological closing (merge short pauses < 250ms)
        min_silence_frames = int(0.25 / (hop_len / sr))
        silence_count = 0
        for i in range(len(is_speech)):
            if not is_speech[i]:
                silence_count += 1
            else:
                if silence_count < min_silence_frames and silence_count > 0:
                    is_speech[i - silence_count : i] = True
                silence_count = 0
                
        # Group into speech segments
        segments = []
        in_speech = False
        start_t = 0.0
        
        for i, val in enumerate(is_speech):
            t = (i * hop_len) / sr
            if val and not in_speech:
                in_speech = True
                start_t = t
            elif not val and in_speech:
                in_speech = False
                end_t = (i * hop_len + frame_len) / sr
                if (end_t - start_t) >= 0.15:  # Filter out clicks < 150ms
                    segments.append((round(start_t, 3), round(end_t, 3)))
                    
        if in_speech:
            end_t = len(audio) / sr
            if (end_t - start_t) >= 0.15:
                segments.append((round(start_t, 3), round(end_t, 3)))
                
        if not segments:
            # Fallback to full utterance if no speech segment was identified
            segments = [(0.0, round(len(audio) / sr, 3))]
            
        return segments

    @staticmethod
    def compute_waveform_envelope(audio: np.ndarray, points: int = 150) -> List[float]:
        """Generates a compressed downsampled RMS envelope for frontend waveform display."""
        if len(audio) == 0:
            return [0.0] * points
            
        chunk_size = max(1, len(audio) // points)
        envelope = []
        for i in range(points):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) > 0:
                val = float(np.sqrt(np.mean(chunk**2)))
                envelope.append(round(min(1.0, val * 3.5), 3))
            else:
                envelope.append(0.0)
        return envelope
