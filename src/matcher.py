"""
Matching logic: collect candidate songs from hash matches,
align by time delta, and pick the song with the most votes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .database import FingerprintDatabase


class AudioMatcher:
    def __init__(self, db: FingerprintDatabase):
        self.db = db

    def match(
        self, query_hashes: Iterable[tuple[int, int]], threshold: int = 5
    ) -> list[dict]:
        """
        Match query fingerprints against database.

        Returns list of candidate songs sorted by confidence (votes desc).
        Each dict: {song_id, title, artist, votes, confidence}
        """
        query_list = list(query_hashes)

        # Build lookup: hash -> query time offset (take first occurrence)
        query_map: dict[int, int] = {}
        for h, t in query_list:
            query_map.setdefault(h, t)

        # Query database for all matching hashes
        hash_values = [h for h, _ in query_list]
        db_matches = self.db.query_hashes(hash_values)

        # For each matching hash, compute delta = query_time - db_time
        # Group by song_id, then count votes per (song_id, delta)
        song_delta_votes: dict[int, Counter] = defaultdict(Counter)
        song_total_votes: dict[int, int] = defaultdict(int)

        for hash_val, song_id, db_time in db_matches:
            q_time = query_map.get(hash_val)
            if q_time is None:
                continue
            delta = q_time - db_time
            song_delta_votes[song_id][delta] += 1
            song_total_votes[song_id] += 1

        # Determine best delta per song (the delta with most votes)
        best_delta_per_song: dict[int, int] = {}
        for song_id, delta_counter in song_delta_votes.items():
            best_delta_per_song[song_id] = delta_counter.most_common(1)[0][0]

        # Build candidate list, filter by threshold
        candidates = []
        for song_id, total_votes in song_total_votes.items():
            if total_votes < threshold:
                continue
            song = self.db.get_song(song_id)
            if song is None:
                continue
            candidates.append(
                {
                    "song_id": song_id,
                    "title": song["title"],
                    "artist": song["artist"],
                    "votes": total_votes,
                    "best_delta": best_delta_per_song[song_id],
                    "confidence": min(100, total_votes * 10),  # rough %
                }
            )

        candidates.sort(key=lambda x: x["votes"], reverse=True)
        return candidates