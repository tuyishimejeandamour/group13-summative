"""
Audio visualization utilities for waveforms and spectrograms
"""
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from pathlib import Path
from typing import Tuple


def plot_waveform(audio: np.ndarray, sr: int, title: str = "Waveform", 
                  save_path: str = None, show: bool = False) -> None:
    """
    Plot the waveform of an audio signal.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        title: Plot title
        save_path: Optional path to save the figure
        show: If True, display the plot (for notebooks). If False, close after saving.
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if not show:
            print(f"Saved waveform: {save_path}")
    
    if not show:
        plt.close()


def plot_spectrogram(audio: np.ndarray, sr: int, title: str = "Spectrogram",
                     save_path: str = None, show: bool = False) -> None:
    """
    Plot the spectrogram of an audio signal.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        title: Plot title
        save_path: Optional path to save the figure
        show: If True, display the plot (for notebooks). If False, close after saving.
    """
    # Compute short-time Fourier transform
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', 
                             hop_length=512, cmap='viridis')
    plt.colorbar(format='%+2.0f dB', label='Magnitude (dB)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if not show:
            print(f"Saved spectrogram: {save_path}")
    
    if not show:
        plt.close()


def plot_waveform_and_spectrogram(audio: np.ndarray, sr: int, 
                                   title: str = "Audio Analysis",
                                   save_path: str = None,
                                   show: bool = False) -> None:
    """
    Plot both waveform and spectrogram side by side.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        title: Plot title
        save_path: Optional path to save the figure
        show: If True, display the plot (for notebooks). If False, close after saving.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[0])
    axes[0].set_title(f'{title} - Waveform', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # Spectrogram
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    im = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz',
                                   hop_length=512, cmap='viridis', ax=axes[1])
    axes[1].set_title(f'{title} - Spectrogram', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    # Use the mappable from specshow for the colorbar
    plt.colorbar(im, ax=axes[1], format='%+2.0f dB', 
                 label='Magnitude (dB)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if not show:
            print(f"Saved combined plot: {save_path}")
    
    if not show:
        plt.close()

