"""
Audio augmentation utilities
"""
import numpy as np
import librosa
from typing import Tuple


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """
    Apply pitch shifting to audio.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_steps: Number of semitones to shift (positive = higher, negative = lower)
    
    Returns:
        Pitch-shifted audio signal
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float = 1.2) -> np.ndarray:
    """
    Apply time stretching to audio (changes speed without changing pitch).
    
    Args:
        audio: Audio signal array
        rate: Stretch factor (>1.0 = faster, <1.0 = slower)
    
    Returns:
        Time-stretched audio signal
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def add_background_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add random background noise to audio.
    
    Args:
        audio: Audio signal array
        noise_factor: Amplitude of noise relative to signal (0.0-1.0)
    
    Returns:
        Audio signal with added noise
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio


def add_time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """
    Add time shifting (roll) to audio.
    
    Args:
        audio: Audio signal array
        shift_max: Maximum fraction of audio length to shift
    
    Returns:
        Time-shifted audio signal
    """
    shift = np.random.randint(-int(len(audio) * shift_max), 
                               int(len(audio) * shift_max))
    return np.roll(audio, shift)


def apply_augmentations(audio: np.ndarray, sr: int) -> dict:
    """
    Apply multiple augmentations to an audio sample.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
    
    Returns:
        Dictionary containing original and augmented audio samples
    """
    augmentations = {
        'original': audio,
        'pitch_shift_up': pitch_shift(audio, sr, n_steps=2.0),
        'pitch_shift_down': pitch_shift(audio, sr, n_steps=-2.0),
        'time_stretch_fast': time_stretch(audio, rate=1.2),
        'time_stretch_slow': time_stretch(audio, rate=0.8),
        'background_noise': add_background_noise(audio, noise_factor=0.005),
        'time_shift': add_time_shift(audio, shift_max=0.2)
    }
    
    return augmentations

