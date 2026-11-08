"""
CPU-based audio fingerprint extraction.
Implements the Shazam constellation-map algorithm:
1. Spectrogram via STFT
2. Peak picking (local maxima)
3. Anchor-target pairing within a target zone
4. Hashing: (f_anchor, f_target, delta_t) -> hash
"""
from __future__ import annotations

import numpy as np
import librosa

from .audio import load_audio, normalize_audio


class FingerprintExtractor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_peaks: int = 15,
        fan_out: int = 5,
        target_zone_start: float = 0.0,
        target_zone_end: float = 2.0,  # seconds
        min_hash_time_delta: float = 0.0,
        max_hash_time_delta: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_peaks = n_peaks
        self.fan_out = fan_out
        self.target_zone_start = target_zone_start
        self.target_zone_end = target_zone_end
        self.min_hash_time_delta = min_hash_time_delta
        self.max_hash_time_delta = max_hash_time_delta

    def extract(self, audio_path: str) -> list[tuple[int, int]]:
        """
        Extract fingerprint hashes from an audio file.

        Returns
        -------
        list of (hash_value, time_offset)
            time_offset is in samples (absolute time in the song).
        """
        y, sr = load_audio(audio_path, sample_rate=self.sample_rate)
        y = normalize_audio(y)

        # --- STFT Spectrogram ---
        D = np.abs(
            librosa.stft(
                y,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        )
        # Convert to log scale (dB) to emphasize peaks
        S = librosa.amplitude_to_db(D, ref=np.max)

        # --- Peak Picking ---
        peaks = self._find_peaks(S)

        # --- Constellation Map & Hashing ---
        hashes = self._hash_peaks(peaks)

        return hashes

    def _find_peaks(self, S: np.ndarray) -> list[tuple[int, int, float]]:
        """
        Find local maxima in spectrogram.
        Returns list of (freq_bin, time_frame, magnitude).
        """
        from scipy.ndimage import maximum_filter

        size = (3, 3)
        local_max = maximum_filter(S, size=size) == S
        # Exclude edges
        local_max[0, :] = False
        local_max[-1, :] = False
        local_max[:, 0] = False
        local_max[:, -1] = False

        # Get peak coordinates and magnitudes
        peak_indices = np.argwhere(local_max)
        peak_magnitudes = S[local_max]

        # Sort by magnitude descending
        order = np.argsort(peak_magnitudes)[::-1]
        peak_indices = peak_indices[order]
        peak_magnitudes = peak_magnitudes[order]

        # Convert time frame to absolute sample offset
        peaks = []
        for (freq_bin, time_frame), mag in zip(peak_indices, peak_magnitudes):
            time_samples = int(time_frame * self.hop_length)
            peaks.append((int(freq_bin), int(time_samples), float(mag)))

        # Limit to top peaks globally for performance
        peaks = peaks[: self.n_peaks * 50]  # heuristic cap
        return peaks

    def _hash_peaks(
        self, peaks: list[tuple[int, int, float]]
    ) -> list[tuple[int, int]]:
        """
        Pair anchor peaks with target peaks within a target zone.
        Hash = combine(freq_anchor, freq_target, delta_time).
        """
        hashes: list[tuple[int, int]] = []

        # Sort peaks by time
        peaks_sorted = sorted(peaks, key=lambda p: p[1])

        for i, anchor in enumerate(peaks_sorted):
            f_anchor, t_anchor, _ = anchor
            # Look ahead for target peaks
            count = 0
            for j in range(i + 1, len(peaks_sorted)):
                f_target, t_target, _ = peaks_sorted[j]
                delta_t = (t_target - t_anchor) / self.sample_rate  # seconds

                if delta_t < self.min_hash_time_delta:
                    continue
                if delta_t > self.max_hash_time_delta:
                    break  # since sorted by time

                # Target zone check
                if not (self.target_zone_start <= delta_t <= self.target_zone_end):
                    continue

                # Create hash from frequencies and time delta
                # Quantize frequencies to reduce sensitivity
                fa = int(f_anchor // 10) * 10
                ft = int(f_target // 10) * 10
                dt = int(delta_t * 1000)  # ms precision

                hash_value = self._combine_hash(fa, ft, dt)
                hashes.append((hash_value, t_anchor))

                count += 1
                if count >= self.fan_out:
                    break

        return hashes

    @staticmethod
    def _combine_hash(fa: int, ft: int, dt: int) -> int:
        """
        Combine three integers into a single 64-bit hash.
        """
        # Simple bit mixing (similar spirit to Shazam's approach)
        return ((fa & 0xFFF) << 40) | ((ft & 0xFFF) << 20) | (dt & 0xFFFFF)