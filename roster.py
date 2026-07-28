"""Persistent, mutable roster of subjects.

Unlike the fixed version, the roster is NOT hardcoded: it is seeded with a few
defaults but grows at runtime when the drift auditor spawns a new specialist.
It survives restarts via Redis (a hash) or a local JSON file fallback, mirroring
the memory backend so the whole system uses one storage story.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional


SEED: dict[str, str] = {
    "cooking": "recipes, ingredients, cooking techniques, food, kitchen, nutrition of meals",
    "finance": "money, investing, budgeting, taxes, loans, interest, markets, personal finance",
    "travel":  "destinations, flights, itineraries, visas, hotels, local customs for travellers",
    "fitness": "exercise, workouts, gym, running, muscles, training plans, sports performance",
}


class _RedisRoster:
    def __init__(self, url: str, prefix: str):
        import redis

        self._r = redis.from_url(url, socket_connect_timeout=2, decode_responses=True)
        self._r.ping()
        self._key = f"{prefix}:roster"

    def load(self) -> dict[str, str]:
        return self._r.hgetall(self._key)

    def put(self, name: str, desc: str) -> None:
        self._r.hset(self._key, name, desc)


class _JsonRoster:
    def __init__(self, path: pathlib.Path):
        self._path = path
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    def load(self) -> dict[str, str]:
        return dict(self._data)

    def put(self, name: str, desc: str) -> None:
        self._data[name] = desc
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


class Roster:
    """Facade over whichever backend is live; seeds defaults on first use."""

    def __init__(self, backend):
        self._b = backend
        current = self._b.load()
        if not current:  # first run -> lay down the seed subjects
            for name, desc in SEED.items():
                self._b.put(name, desc)

    @classmethod
    def build(cls, redis_url: Optional[str], prefix: str, json_path: pathlib.Path) -> "Roster":
        if redis_url:
            try:
                return cls(_RedisRoster(redis_url, prefix))
            except Exception:
                pass  # memory.py already reports the fallback
        return cls(_JsonRoster(json_path))

    def all(self) -> dict[str, str]:
        return self._b.load()

    def exists(self, name: str) -> bool:
        return name in self._b.load()

    def add(self, name: str, desc: str) -> None:
        self._b.put(name, desc)
