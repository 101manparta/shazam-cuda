"""
Audio loading and preprocessing.
Supports mono conversion, resampling, and normalization.
"""
from __future__ import annotations

import numpy as np
import librosa


def load_audio(
    path: str,
    sample_rate: int = 22050,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return samples + sample rate.

    Parameters
    ----------
    path : str
        Path to audio file (wav, mp3, m4a, etc.).
    sample_rate : int
        Target sample rate in Hz. 22050 is a good balance for fingerprinting.
    mono : bool
        Convert to mono if True.

    Returns
    -------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.
    """
    y, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return y, sr


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1].
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def frame_audio(
    y: np.ndarray,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """
    Split audio into overlapping frames.
    Useful for custom CPU-based STFT if needed.
    """
    num_frames = 1 + (len(y) - frame_size) // hop_size
    frames = np.lib.stride_tricks.sliding_window_view(
        y, frame_size
    )[::hop_size][:num_frames]
    return frames