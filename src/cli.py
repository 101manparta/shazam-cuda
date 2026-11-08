"""
Command-line interface for the Shazam-like fingerprinting system.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .audio import load_audio
from .fingerprint import FingerprintExtractor
from .cuda_fingerprint import CudaFingerprintExtractor
from .database import FingerprintDatabase
from .matcher import AudioMatcher


def cmd_add(args: argparse.Namespace) -> None:
    """Add a song to the fingerprint database."""
    db = FingerprintDatabase(args.db)
    use_cuda = not args.cpu

    if use_cuda:
        extractor = CudaFingerprintExtractor()
    else:
        extractor = FingerprintExtractor()

    print(f"[+] Memproses: {args.file}")
    hashes = extractor.extract(args.file)

    song_id = db.add_song(args.title, args.artist, str(args.file))
    db.add_fingerprints(song_id, hashes)
    print(
        f"[+] Ditambahkan lagu ID {song_id}: {args.title} ({len(hashes)} fingerprints)"
    )
    db.close()


def cmd_match(args: argparse.Namespace) -> None:
    """Identify an unknown audio file."""
    db = FingerprintDatabase(args.db)
    matcher = AudioMatcher(db)

    use_cuda = not args.cpu
    if use_cuda:
        extractor = CudaFingerprintExtractor()
    else:
        extractor = FingerprintExtractor()

    print(f"[+] Mencocokkan: {args.file}")
    query_hashes = extractor.extract(args.file)
    print(f"    {len(query_hashes)} fingerprints diekstrak")

    results = matcher.match(query_hashes, threshold=args.threshold)

    if not results:
        print("[-] Tidak ada lagu yang cocok ditemukan.")
    else:
        print("\nHasil pencocokan:")
        print(f"{'Rank':<6} {'Judul':<30} {'Artis':<25} {'Votes':<7} {'Confidence'}")
        print("-" * 80)
        for i, r in enumerate(results[:10], 1):
            print(
                f"{i:<6} {r['title']:<30} {r['artist'] or '-':<25} "
                f"{r['votes']:<7} {r['confidence']}%"
            )

    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shazam-like Audio Fingerprinting with CUDA"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- add ---
    p_add = sub.add_parser("add", help="Tambahkan lagu ke database")
    p_add.add_argument("file", type=Path, help="Path file audio")
    p_add.add_argument("--title", required=True, help="Judul lagu")
    p_add.add_argument("--artist", default="Unknown", help="Nama artis")
    p_add.add_argument("--db", default="data/fingerprints.db", help="Path database SQLite")
    p_add.add_argument(
        "--cpu",
        action="store_true",
        help="Paksa CPU (non-CUDA)",
    )
    p_add.set_defaults(func=cmd_add)

    # --- match ---
    p_match = sub.add_parser("match", help="Identifikasi lagu dari file audio")
    p_match.add_argument("file", type=Path, help="Path file audio uji")
    p_match.add_argument("--db", default="data/fingerprints.db", help="Path database SQLite")
    p_match.add_argument(
        "--cpu",
        action="store_true",
        help="Paksa CPU (non-CUDA)",
    )
    p_match.add_argument(
        "--threshold", type=int, default=5, help="Minimum votes untuk dianggap cocok"
    )
    p_match.set_defaults(func=cmd_match)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()