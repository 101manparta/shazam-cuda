"""
SQLite database for storing and querying audio fingerprints.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


class FingerprintDatabase:
    def __init__(self, db_path: str = "data/fingerprints.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    artist TEXT,
                    file_path TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    hash INTEGER NOT NULL,
                    song_id INTEGER NOT NULL,
                    time_offset INTEGER NOT NULL,
                    FOREIGN KEY(song_id) REFERENCES songs(id)
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)"
            )

    def add_song(self, title: str, artist: str, file_path: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO songs (title, artist, file_path) VALUES (?, ?, ?)",
                (title, artist, file_path),
            )
            return cur.lastrowid

    def add_fingerprints(self, song_id: int, hashes: Iterable[tuple[int, int]]) -> None:
        rows = [(h, song_id, t) for h, t in hashes]
        with self.conn:
            self.conn.executemany(
                "INSERT INTO fingerprints (hash, song_id, time_offset) VALUES (?, ?, ?)",
                rows,
            )

    def query_hashes(
        self, hashes: Iterable[tuple[int, int]]
    ) -> list[tuple[int, int, int]]:
        """
        Return list of (hash, song_id, time_offset) for matching hashes.
        """
        results = []
        hash_list = [h for h, _ in hashes]
        if not hash_list:
            return results

        # Query in batches to avoid SQL limits
        batch = 500
        for i in range(0, len(hash_list), batch):
            chunk = hash_list[i : i + batch]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT hash, song_id, time_offset FROM fingerprints WHERE hash IN ({placeholders})",
                chunk,
            ).fetchall()
            results.extend(rows)
        return results

    def get_song(self, song_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, title, artist, file_path FROM songs WHERE id = ?",
            (song_id,),
        ).fetchone()
        if row:
            return {
                "id": row[0],
                "title": row[1],
                "artist": row[2],
                "file_path": row[3],
            }
        return None

    def close(self) -> None:
        self.conn.close()