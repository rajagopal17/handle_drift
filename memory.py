"""Per-agent memory store.

Each subject agent owns an isolated namespace. Memory is a list of past
turns (query + answer + timestamp). Backend is Redis when reachable, else a
local JSON file so the system runs anywhere. The public API is identical
regardless of backend, so agents never know which one is live.
"""
from __future__ import annotations

import json
import time
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Turn:
    query: str
    answer: str
    ts: float

    def as_dict(self) -> dict:
        return asdict(self)


class _Backend:
    def read(self, subject: str) -> list[Turn]:  # pragma: no cover - interface
        raise NotImplementedError

    def append(self, subject: str, turn: Turn) -> None:  # pragma: no cover
        raise NotImplementedError


class RedisBackend(_Backend):
    """Stores each subject's turns as a Redis list of JSON strings."""

    def __init__(self, url: str, prefix: str):
        import redis  # imported lazily so the dep is optional

        self._r = redis.from_url(url, socket_connect_timeout=2, decode_responses=True)
        self._r.ping()  # fail fast if the server is down
        self._prefix = prefix

    def _key(self, subject: str) -> str:
        return f"{self._prefix}:mem:{subject}"

    def read(self, subject: str) -> list[Turn]:
        raw = self._r.lrange(self._key(subject), 0, -1)
        return [Turn(**json.loads(x)) for x in raw]

    def append(self, subject: str, turn: Turn) -> None:
        self._r.rpush(self._key(subject), json.dumps(turn.as_dict()))


class JsonFileBackend(_Backend):
    """Fallback backend: one JSON file holding {subject: [turns]}."""

    def __init__(self, path: pathlib.Path):
        self._path = path
        self._data: dict[str, list[dict]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    def _flush(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def read(self, subject: str) -> list[Turn]:
        return [Turn(**t) for t in self._data.get(subject, [])]

    def append(self, subject: str, turn: Turn) -> None:
        self._data.setdefault(subject, []).append(turn.as_dict())
        self._flush()


class Memory:
    """Facade over whichever backend is live."""

    def __init__(self, backend: _Backend, kind: str):
        self._b = backend
        self.kind = kind  # "redis" or "json" -- for status reporting

    @classmethod
    def build(cls, redis_url: Optional[str], prefix: str, json_path: pathlib.Path) -> "Memory":
        if redis_url:
            try:
                return cls(RedisBackend(redis_url, prefix), "redis")
            except Exception as e:  # lib missing or server down -> fall back
                print(f"[memory] Redis unavailable ({type(e).__name__}); using local JSON file.")
        return cls(JsonFileBackend(json_path), "json")

    def recall(self, subject: str, query: str, limit: int = 4) -> list[Turn]:
        """Return the most relevant past turns for this subject.

        Simple but effective: score by keyword overlap with the query, then
        fall back to recency. This is what lets an agent 'remember earlier
        queries' without a vector DB.
        """
        turns = self._b.read(subject)
        if not turns:
            return []
        q_words = {w for w in _tokenize(query) if len(w) > 2}
        scored = []
        for i, t in enumerate(turns):
            overlap = len(q_words & set(_tokenize(t.query + " " + t.answer)))
            scored.append((overlap, i, t))  # i keeps recency as tie-breaker
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [t for _, _, t in scored[:limit]]

    def remember(self, subject: str, query: str, answer: str) -> None:
        self._b.append(subject, Turn(query=query, answer=answer, ts=time.time()))


def _tokenize(text: str) -> list[str]:
    return [w.strip(".,!?;:()[]\"'").lower() for w in text.split()]
