"""Thin OpenAI wrapper: one chat() for prose, one json() for structured calls.

Keeps the OpenAI client in a single place so agents stay provider-agnostic.
Config comes from the local .env (OPENAI_API_KEY, OPENAI_MODEL).
"""
from __future__ import annotations

import json
import os
import pathlib

from dotenv import load_dotenv
from openai import OpenAI


class LLM:
    def __init__(self):
        load_dotenv(pathlib.Path(__file__).with_name(".env"))
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing from .env")
        self._client = OpenAI(api_key=key)
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    def json(self, system: str, user: str) -> dict:
        """Chat call constrained to a JSON object; returns {} on parse failure."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {}
