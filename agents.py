"""Subject specialists, the router, the drift auditor, and the answer auditor.

Design (dynamic roster version):
- The roster of subjects is NOT fixed. It lives in `roster.py` and can grow.
- A Router (LLM) classifies each query against the CURRENT roster and reports a
  fit score (0-1) plus any overlaps.
- If the best fit is weak, that is DRIFT: the DriftAuditor invents a brand-new
  specialist (name + description), which is added to the roster and answers.
- Each SubjectAgent answers ONE subject and reads/writes only its own memory.
- If a query lands on the wrong existing agent, it HANDS OFF to a better one.
- After answering, an Auditor (LLM) reviews and the agent self-corrects once.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from memory import Memory, Turn


@dataclass
class RouteDecision:
    primary: str
    fit: float               # 0-1: how well `primary` covers the query
    overlaps: list[str]
    reason: str


@dataclass
class AgentResult:
    subject: str             # who actually answered
    answer: str
    handoffs: list[str]      # chain of agents involved, in order
    audit_note: str          # what the answer auditor said
    used_memory: int         # how many past turns were recalled
    created: bool = False    # was a new agent spawned for this query?
    fit: float = 1.0         # router's fit score for the chosen subject


class Router:
    """LLM classifier that maps a query onto the CURRENT roster, with a fit score."""

    def __init__(self, llm):
        self._llm = llm

    def route(self, query: str, roster: dict[str, str]) -> RouteDecision:
        listing = "\n".join(f"- {name}: {desc}" for name, desc in roster.items())
        sys = (
            "You are a routing classifier for a team of subject-specialist agents. "
            "Pick the single best PRIMARY subject from the roster for the query, and "
            "rate FIT from 0.0 to 1.0 for how well that subject actually covers the "
            "query (1.0 = squarely in that subject; below ~0.5 = the query is really "
            "about something the roster does not cover yet). Also list any other "
            "subjects it overlaps. Use only roster subjects. Respond strict JSON: "
            '{"primary":"<subject>","fit":<0..1>,"overlaps":["<subject>"...],"reason":"<short>"}'
        )
        user = f"Roster:\n{listing}\n\nQuery: {query}"
        raw = self._llm.json(sys, user)
        primary = raw.get("primary", "")
        if primary not in roster:
            primary = next(iter(roster))  # defensive default
        try:
            fit = float(raw.get("fit", 1.0))
        except (TypeError, ValueError):
            fit = 1.0
        overlaps = [o for o in raw.get("overlaps", []) if o in roster and o != primary]
        return RouteDecision(primary=primary, fit=fit, overlaps=overlaps, reason=raw.get("reason", ""))


class DriftAuditor:
    """Detects drift and invents a new specialist to cover it.

    This is the 'audit creates a new agent' step: when routing fit is weak, it
    proposes a new subject, but first checks the query isn't really an existing
    subject in disguise (to keep the roster from exploding with near-duplicates).
    """

    def __init__(self, llm):
        self._llm = llm

    def propose(self, query: str, roster: dict[str, str]) -> tuple[str, str, bool]:
        """Return (subject_name, description, is_new).

        is_new=False means an existing subject actually fits after all.
        """
        listing = "\n".join(f"- {name}: {desc}" for name, desc in roster.items())
        sys = (
            "A routing step found that a query does not fit any existing specialist "
            "well. Decide whether it truly needs a NEW specialist or actually belongs "
            "to an existing one. If new: invent a concise, reusable subject name "
            "(one or two lowercase words, e.g. 'gardening', 'legal', 'auto repair') "
            "and a short description of what that specialist covers. Prefer reusing an "
            "existing subject if it genuinely fits. Respond strict JSON: "
            '{"is_new": true|false, "subject": "<name>", "description": "<short>"}'
        )
        user = f"Existing subjects:\n{listing}\n\nQuery: {query}"
        raw = self._llm.json(sys, user)
        is_new = bool(raw.get("is_new", True))
        name = _slug(raw.get("subject", "")) or "general"
        # If the proposed name already exists, it's a reuse regardless of the flag.
        if name in roster:
            return name, roster[name], False
        if not is_new:
            # Model says reuse but named something new; fall back to closest existing.
            return name, f"queries about {name}", True
        desc = raw.get("description", "").strip() or f"queries about {name}"
        return name, desc, True


class Auditor:
    """Reviews an answer and demands a fix if it's wrong or off-subject."""

    def __init__(self, llm):
        self._llm = llm

    def audit(self, subject: str, description: str, query: str, answer: str) -> tuple[bool, str]:
        sys = (
            f"You are a strict reviewer for the '{subject}' specialist ({description}). "
            "Judge whether the ANSWER correctly and completely addresses the QUERY, "
            "stays on subject, and is free of factual errors or contradictions. "
            'Respond strict JSON: {"ok": true|false, "issues": "<what to fix, or empty>"}'
        )
        user = f"QUERY: {query}\n\nANSWER: {answer}"
        raw = self._llm.json(sys, user)
        return bool(raw.get("ok", True)), raw.get("issues", "")


class SubjectAgent:
    """A specialist bound to one subject and its private memory namespace."""

    def __init__(self, subject: str, description: str, llm, memory: Memory, auditor: Auditor):
        self.subject = subject
        self.description = description
        self._llm = llm
        self._mem = memory
        self._auditor = auditor

    def in_scope(self, query: str) -> bool:
        """Self-check: does this query actually belong to me?"""
        sys = (
            f"You are the '{self.subject}' specialist ({self.description}). "
            'Does the query fall within YOUR subject? Respond strict JSON: '
            '{"in_scope": true|false}'
        )
        raw = self._llm.json(sys, f"Query: {query}")
        return bool(raw.get("in_scope", True))

    def answer(self, query: str, consult: list[Turn] | None = None) -> tuple[str, str, int]:
        """Produce an answer grounded in this agent's memory (+ optional consult).

        Returns (answer, audit_note, n_memory_used).
        """
        recalled = self._mem.recall(self.subject, query, limit=4)
        memory_block = _format_turns("Your earlier Q&A on this subject", recalled)
        consult_block = _format_turns("Context handed over from another specialist", consult or [])

        sys = (
            f"You are the '{self.subject}' specialist ({self.description}). "
            "Answer the user's query concisely and accurately. Use your earlier Q&A to "
            "stay consistent with what you've already told this user (remember earlier "
            "queries). If earlier context is relevant, build on it explicitly."
        )
        user = f"{memory_block}{consult_block}Current query: {query}"
        answer = self._llm.chat(sys, user)

        # ---- audit + one self-correction pass ----
        ok, issues = self._auditor.audit(self.subject, self.description, query, answer)
        audit_note = "passed" if ok else f"revised: {issues}"
        if not ok:
            fix_sys = (
                f"You are the '{self.subject}' specialist. A reviewer found issues with "
                "your answer. Produce a corrected answer that resolves them."
            )
            fix_user = f"Query: {query}\n\nYour answer: {answer}\n\nReviewer issues: {issues}"
            answer = self._llm.chat(fix_sys, fix_user)

        self._mem.remember(self.subject, query, answer)
        return answer, audit_note, len(recalled)


def _format_turns(header: str, turns: list[Turn]) -> str:
    if not turns:
        return ""
    lines = [f"{header}:"]
    for t in turns:
        lines.append(f"  Q: {t.query}\n  A: {t.answer}")
    return "\n".join(lines) + "\n\n"


def _slug(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9 ]+", "", name)
    return re.sub(r"\s+", " ", name).strip()
