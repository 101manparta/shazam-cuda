"""Unit tests for fingerprint extraction."""
from __future__ import annotations

import numpy as np
import pytest

from src.audio import load_audio, normalize_audio
from src.fingerprint import FingerprintExtractor


def test_normalize_audio_range():
    y = np.array([-0.5, 0.0, 0.5, 1.0])
    yn = normalize_audio(y)
    assert np.isclose(np.max(np.abs(yn)), 1.0)


def test_extractor_init():
    ext = FingerprintExtractor()
    assert ext.sample_rate == 22050
    assert ext.n_fft == 4096


def test_extract_on_synthetic_audio(tmp_path):
    # Create a synthetic sine wave file
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "test.wav"
    import soundfile as sf

    sf.write(str(path), y, sr)

    ext = FingerprintExtractor(sample_rate=sr, n_peaks=10)
    hashes = ext.extract(str(path))
    assert isinstance(hashes, list)
    if hashes:
        h, t_off = hashes[0]
        assert isinstance(h, int)
        assert isinstance(t_off, int)