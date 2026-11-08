"""
CUDA-accelerated fingerprint spectrogram and peak picking using PyTorch.
Falls back to CPU automatically if CUDA is unavailable.
"""
from __future__ import annotations

import torch
import numpy as np

from .audio import load_audio, normalize_audio


class CudaFingerprintExtractor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_peaks: int = 15,
        fan_out: int = 5,
        target_zone_start: float = 0.0,
        target_zone_end: float = 2.0,
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

        # Pick device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CudaFingerprintExtractor] Using device: {self.device}")

    def extract(self, audio_path: str) -> list[tuple[int, int]]:
        y, sr = load_audio(audio_path, sample_rate=self.sample_rate)
        y = normalize_audio(y)

        # Convert to torch tensor
        y_tensor = torch.from_numpy(y).float().to(self.device)

        # --- STFT on CUDA ---
        # torch.stft returns complex tensor; take magnitude
        stft = torch.stft(
            y_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=self.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.abs(stft)  # shape: (freq_bins, time_frames)

        # Log magnitude (dB)
        # Avoid log(0)
        S = 10.0 * torch.log10(magnitude + 1e-10)

        # --- Peak picking on CUDA ---
        peaks = self._find_peaks_cuda(S)

        # Move to CPU for hashing (hashing is scalar/irregular, better on CPU)
        peaks_cpu = [(int(p[0]), int(p[1]), float(p[2])) for p in peaks]

        # --- Hashing ---
        hashes = self._hash_peaks(peaks_cpu)
        return hashes

    def _find_peaks_cuda(self, S: torch.Tensor) -> list[torch.Tensor]:
        """
        Find local maxima using max pooling on GPU.
        Returns list of (freq_bin, time_frame, magnitude).
        """
        # Add batch and channel dims for max_pool2d: (1, 1, F, T)
        S_batched = S.unsqueeze(0).unsqueeze(0)

        # 3x3 max pooling with stride 1, padding 1
        kernel = 3
        padded = torch.nn.functional.pad(S_batched, (1, 1, 1, 1), mode="reflect")
        local_max = torch.nn.functional.max_pool2d(
            padded, kernel_size=kernel, stride=1, padding=0
        )

        # Mask where original equals local max
        mask = (S_batched == local_max)

        # Exclude edges
        mask[:, :, 0, :] = False
        mask[:, :, -1, :] = False
        mask[:, :, :, 0] = False
        mask[:, :, :, -1] = False

        # Get indices
        indices = torch.nonzero(mask[0, 0])  # (N, 2): (freq, time)
        if indices.numel() == 0:
            return []

        # Get magnitudes at those indices
        freqs = indices[:, 0].float()
        times = indices[:, 1].float()
        mags = S_batched[0, 0][mask[0, 0]]

        # Sort by magnitude descending
        order = torch.argsort(mags, descending=True)
        freqs = freqs[order][: self.n_peaks * 200]  # cap
        times = times[order][: self.n_peaks * 200]
        mags = mags[order][: self.n_peaks * 200]

        # Stack as (freq, time, mag)
        peaks = torch.stack([freqs, times, mags], dim=1)
        return peaks.tolist()

    def _hash_peaks(
        self, peaks: list[tuple[int, int, float]]
    ) -> list[tuple[int, int]]:
        """
        Pair anchor-target peaks (same logic as CPU version).
        """
        hashes: list[tuple[int, int]] = []
        peaks_sorted = sorted(peaks, key=lambda p: p[1])

        for i, anchor in enumerate(peaks_sorted):
            f_anchor, t_anchor, _ = anchor
            count = 0
            for j in range(i + 1, len(peaks_sorted)):
                f_target, t_target, _ = peaks_sorted[j]
                delta_t = (t_target - t_anchor) / self.sample_rate

                if delta_t < self.min_hash_time_delta:
                    continue
                if delta_t > self.max_hash_time_delta:
                    break
                if not (self.target_zone_start <= delta_t <= self.target_zone_end):
                    continue

                fa = int(f_anchor // 10) * 10
                ft = int(f_target // 10) * 10
                dt = int(delta_t * 1000)

                hash_value = ((fa & 0xFFF) << 40) | ((ft & 0xFFF) << 20) | (dt & 0xFFFFF)
                hashes.append((hash_value, t_anchor))

                count += 1
                if count >= self.fan_out:
                    break

        return hashes