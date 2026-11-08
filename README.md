# 🎵 Shazam-like Audio Fingerprinting with CUDA

[![CI Python](https://img.shields.io/github/actions/workflow/status/101manparta/shazam-cuda/.github/workflows/ci.yml?branch=main)](https://github.com/101manparta/shazam-cuda/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/CUDA-PyTorch-7B42BC?logo=nvidia)](https://pytorch.org/)
[![librosa](https://img.shields.io/badge/librosa-audio%20analysis-green)](https://librosa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

Sistem identifikasi musik ala **Shazam** yang mencari lagu berdasarkan sidik jari audio (fingerprint), dengan percepatan **CUDA (NVIDIA GPU)** untuk komputasi spectrogram.

> **Analogi:** Setiap lagu dikonversi menjadi "sidik jari" berdasarkan puncak frekuensi yang tampak pada spectrogram. Saat kamu merekam suara di lingkungan bising, sistem mencocokkan sidik jari tersebut dengan database lagu yang telah terdaftar.

---

## 🧠 Algoritma yang Digunakan

Algoritma ini mengikuti pendekatan klasik **Wang (2003)** yang digunakan Shazam:

1. **STFT Spectrogram** — Mengubah waktu-domain audio ke frekuensi-waktu.
2. **Peak Picking** — Mendeteksi puncak energi (local maxima) pada spectrogram.
3. **Constellation Map** — Pasangkan setiap puncak (anchor) dengan puncak lain (target) dalam window waktu tertentu.
4. **Hashing** — Gabungkan `(frekuensi_anchor, frekuensi_target, delta_waktu)` menjadi hash tunggal.
5. **Matching** — Cari hash yang cocok dalam database, lakukan perataan waktu (time alignment), dan pilih lagu dengan jumlah kecocokan terbanyak.

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

| Alat | Versi | Keterangan |
|------|-------|------------|
| Python | 3.10+ | [python.org](https://www.python.org/) |
| CUDA Toolkit | 11.8+ | Hanya jika ingin GPU (NVIDIA) |
| ffmpeg | 4.0+ | Untuk baca file MP3/M4A via librosa |

### 2. Install

```powershell
git clone https://github.com/101manparta/shazam-cuda.git
cd shazam-cuda
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# atau: source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
pip install soundfile   # untuk tes
```

> **GPU (CUDA):** Jika memiliki GPU NVIDIA, PyTorch akan mendeteksi otomatis.  
> Untuk paksa CPU, tambahkan flag `--cpu` pada setiap perintah.

### 3. Tambahkan Lagu ke Database

```powershell
python -m src.cli add "musik/lagu1.mp3" --title "Judul Lagu" --artist "Nama Artis"
```

### 4. Identifikasi Lagu

```powershell
python -m src.cli match "rekaman_uji.mp3" --threshold 5
```

Contoh output:
```
[+] Mencocokkan: rekaman_uji.mp3
    1420 fingerprints diekstrak

Hasil pencocokan:
Rank  Judul                           Artis                     Votes  Confidence
--------------------------------------------------------------------------------
1     Blinding Lights                 The Weeknd              23       100%
2     Levitating                      Dua Lipa                8        80%
```

---

## ⚙️ Arsitektur & Teknologi

```
File Audio (MP3/WAV)
      ↓
librosa (load, mono, resample 22.05kHz)
      ↓
STFT Spectrogram
      ├─▶ CPU: NumPy + SciPy
      └─▶ CUDA: PyTorch torch.stft (GPU)
      ↓
Log Magnitude (dB)
      ↓
Peak Picking (local maxima)
      ↓
Constellation Map (anchor-target pairs)
      ↓
Hashing (f1, f2, delta_t) → Integer Hash
      ↓
SQLite Database
      ↓
Time-Alignment Voting → Hasil Identifikasi
```

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| Bahasa | Python 3.10+ | Logika utama |
| Audio | librosa, NumPy | Memuat & menganalisis suara |
| GPU | PyTorch + CUDA | Percepatan STFT & peak picking |
| Database | SQLite | Penyimpanan fingerprint lagu |
| CLI | argparse | Antarmuka baris perintah |

---

## 📂 Struktur Folder

```
shazam-cuda/
├── src/
│   ├── audio.py            # Load & preprocessing audio
│   ├── fingerprint.py      # Fingerprint CPU (referensi)
│   ├── cuda_fingerprint.py # Fingerprint CUDA via PyTorch
│   ├── database.py         # SQLite database
│   ├── matcher.py          # Logika pencocokan
│   └── cli.py              # CLI entry point
├── tests/
│   └── test_fingerprint.py
├── data/                   # Database & sample audio (tidak di-commit)
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```powershell
pytest tests/ -v
```

CI di GitHub Actions akan otomatis menjalankan pytest pada setiap push dan Pull Request.

---

## 🛠️ Roadmap

| Fitur | Status |
|-------|--------|
| Ekstraksi fingerprint CPU (NumPy/SciPy) | ✅ Selesai |
| Percepatan CUDA via PyTorch | ✅ Selesai |
| Database SQLite + pencocokan waktu | ✅ Selesai |
| CLI `add` dan `match` | ✅ Selesai |
| Web dashboard (Streamlit/Flask) | 🔄 Rencanaan |
| Streaming audio real-time (microphone) | 🔄 Rencanaan |
| Hashing lebih robust (SHA-256 + FNV) | 🔄 Rencanaan |

---

## 🤝 Kontribusi

1. Fork repository ini
2. Buat branch: `git checkout -b fitur/nama-fitur`
3. Commit: `git commit -m "Tambah fitur X"`
4. Push: `git push origin fitur/nama-fitur`
5. Buat Pull Request

---

## 📄 Lisensi

MIT License. Lihat file [LICENSE](LICENSE).

---

## 👤 Author

Dibuat oleh **[101manparta](https://github.com/101manparta)**