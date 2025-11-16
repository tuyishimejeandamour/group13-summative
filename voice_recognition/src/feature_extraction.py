"""
Feature extraction utilities for audio signals
"""
import numpy as np
import librosa
from typing import Dict, List


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC (Mel-frequency cepstral coefficients) features.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients to extract
    
    Returns:
        Array of MFCC features
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs


def extract_spectral_rolloff(audio: np.ndarray, sr: int, roll_percent: float = 0.85) -> np.ndarray:
    """
    Extract spectral roll-off features.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        roll_percent: Roll-off percentage (default: 0.85)
    
    Returns:
        Array of spectral roll-off values
    """
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=roll_percent)
    return rolloff


def extract_energy(audio: np.ndarray) -> np.ndarray:
    """
    Extract energy features (RMS energy).
    
    Args:
        audio: Audio signal array
    
    Returns:
        Array of energy values
    """
    energy = librosa.feature.rms(y=audio)
    return energy


def extract_zero_crossing_rate(audio: np.ndarray) -> np.ndarray:
    """
    Extract zero crossing rate features.
    
    Args:
        audio: Audio signal array
    
    Returns:
        Array of zero crossing rate values
    """
    zcr = librosa.feature.zero_crossing_rate(audio)
    return zcr


def extract_spectral_centroid(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract spectral centroid features.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
    
    Returns:
        Array of spectral centroid values
    """
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return centroid


def extract_spectral_bandwidth(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract spectral bandwidth features.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
    
    Returns:
        Array of spectral bandwidth values
    """
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    return bandwidth


def extract_chroma_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract chroma features.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
    
    Returns:
        Array of chroma features
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return chroma


def compute_statistics(feature_array: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical features from a feature array.
    
    Args:
        feature_array: Feature array (2D)
    
    Returns:
        Dictionary with mean, std, min, max for each feature dimension
    """
    stats = {}
    # Compute statistics across time frames
    stats['mean'] = np.mean(feature_array, axis=1).tolist()
    stats['std'] = np.std(feature_array, axis=1).tolist()
    stats['min'] = np.min(feature_array, axis=1).tolist()
    stats['max'] = np.max(feature_array, axis=1).tolist()
    
    return stats


def extract_all_features(audio: np.ndarray, sr: int) -> Dict[str, any]:
    """
    Extract all audio features and their statistics.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
    
    Returns:
        Dictionary containing all extracted features and their statistics
    """
    features = {}
    
    # Extract features
    mfccs = extract_mfcc(audio, sr, n_mfcc=13)
    rolloff = extract_spectral_rolloff(audio, sr)
    energy = extract_energy(audio)
    zcr = extract_zero_crossing_rate(audio)
    centroid = extract_spectral_centroid(audio, sr)
    bandwidth = extract_spectral_bandwidth(audio, sr)
    chroma = extract_chroma_features(audio, sr)
    
    # Compute statistics for each feature
    features['mfcc'] = compute_statistics(mfccs)
    features['spectral_rolloff'] = compute_statistics(rolloff)
    features['energy'] = compute_statistics(energy)
    features['zero_crossing_rate'] = compute_statistics(zcr)
    features['spectral_centroid'] = compute_statistics(centroid)
    features['spectral_bandwidth'] = compute_statistics(bandwidth)
    features['chroma'] = compute_statistics(chroma)
    
    # Additional scalar features
    features['duration'] = len(audio) / sr
    features['sample_rate'] = sr
    
    return features


def flatten_features(features: Dict[str, any], prefix: str = '') -> Dict[str, float]:
    """
    Flatten feature dictionary into a single-level dictionary for CSV export.
    
    Args:
        features: Nested feature dictionary
        prefix: Prefix for feature names
    
    Returns:
        Flattened dictionary with string keys and float values
    """
    flattened = {}
    
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict):
            # Handle statistical features
            for stat_name, stat_values in feature_data.items():
                if isinstance(stat_values, list):
                    # Multiple values (e.g., MFCC coefficients)
                    for idx, value in enumerate(stat_values):
                        key = f"{prefix}{feature_name}_{stat_name}_{idx}" if prefix else f"{feature_name}_{stat_name}_{idx}"
                        flattened[key] = float(value)
                else:
                    # Single value
                    key = f"{prefix}{feature_name}_{stat_name}" if prefix else f"{feature_name}_{stat_name}"
                    flattened[key] = float(stat_values)
        else:
            # Scalar features
            key = f"{prefix}{feature_name}" if prefix else feature_name
            flattened[key] = float(feature_data)
    
    return flattened

