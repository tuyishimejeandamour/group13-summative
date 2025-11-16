"""
Audio loading utilities
"""
import os
import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


def load_audio_file(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load a WAV audio file and return the audio signal and sample rate.
    
    Args:
        file_path: Path to the WAV audio file
        sr: Target sample rate (default: 22050)
    
    Returns:
        Tuple of (audio signal, sample rate)
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate


def load_all_audio_samples(data_dir: str) -> Dict[str, Dict[str, Tuple[np.ndarray, int]]]:
    """
    Load all WAV audio samples from the data directory.
    
    Args:
        data_dir: Path to the data directory containing member folders with WAV files
    
    Returns:
        Dictionary with structure: {member_name: {audio_file: (signal, sr)}}
    """
    audio_samples = {}
    data_path = Path(data_dir)
    
    # Iterate through each member's folder
    for member_dir in data_path.iterdir():
        if member_dir.is_dir():
            member_name = member_dir.name
            audio_samples[member_name] = {}
            
            # Load all WAV files in the member's folder
            audio_files = sorted(member_dir.glob("*.wav"))
            
            for audio_file in audio_files:
                audio_name = audio_file.stem
                try:
                    signal, sr = load_audio_file(str(audio_file))
                    audio_samples[member_name][audio_name] = (signal, sr)
                    print(f"Loaded: {member_name}/{audio_name}")
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
    
    return audio_samples

