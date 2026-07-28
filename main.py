"""Subject-specialist agents with memory, handoff, and drift-spawned agents.

The roster is NOT fixed. When a query drifts away from every existing
specialist, the drift auditor spins up a brand-new agent for it.

Flow for each query:
  1. Router (LLM) classifies the query against the CURRENT roster and scores FIT.
  2. DRIFT CHECK (the audit): if fit is weak, the drift auditor either reuses an
     existing subject or invents a NEW specialist, adds it to the persistent
     roster, and that fresh agent (with empty memory) answers.
  3. Otherwise the routed agent self-checks scope and HANDS OFF if it isn't
     theirs (single hop). Overlapping subjects contribute consult context.
  4. The agent answers using its OWN memory of earlier queries, then an Auditor
     reviews and the agent self-corrects once if needed.
  5. The turn is stored in that agent's memory (Redis, or local JSON fallback).

Run:
  python main.py            # interactive REPL
  python main.py --demo     # scripted queries showing routing, handoff, memory,
                            # and a query that drifts and spawns a new agent
"""
from __future__ import annotations

import os
import pathlib
import sys

# Windows consoles default to cp1252, which crashes on ₹, em-dashes, etc.
# Force UTF-8 so agent output (and any currency/unicode) prints cleanly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from llm import LLM
from memory import Memory
from roster import Roster
from agents import Router, DriftAuditor, Auditor, SubjectAgent, AgentResult


# Below this router fit score, a query is considered to have drifted away from
# the existing roster and the drift auditor is invoked.
DRIFT_THRESHOLD = 0.5


class Orchestrator:
    def __init__(self):
        self._llm = LLM()
        prefix = _env("REDIS_KEY_PREFIX", "subjmem")
        here = pathlib.Path(__file__).parent
        self._mem = Memory.build(
            redis_url=_env("REDIS_URL", ""),
            prefix=prefix,
            json_path=here / "memory_store.json",
        )
        self._roster = Roster.build(
            redis_url=_env("REDIS_URL", ""),
            prefix=prefix,
            json_path=here / "roster_store.json",
        )
        self._router = Router(self._llm)
        self._drift = DriftAuditor(self._llm)
        self._auditor = Auditor(self._llm)
        self._agents: dict[str, SubjectAgent] = {}  # lazily built per subject

    @property
    def memory_kind(self) -> str:
        return self._mem.kind

    def subjects(self) -> dict[str, str]:
        return self._roster.all()

    def _agent(self, subject: str) -> SubjectAgent:
        """Get (or build) the specialist for a roster subject."""
        if subject not in self._agents:
            desc = self._roster.all().get(subject, f"queries about {subject}")
            self._agents[subject] = SubjectAgent(subject, desc, self._llm, self._mem, self._auditor)
        return self._agents[subject]

    def handle(self, query: str) -> AgentResult:
        roster = self._roster.all()
        route = self._router.route(query, roster)

        # ---- DRIFT AUDIT: weak fit -> reuse or spawn a new specialist ----
        if route.fit < DRIFT_THRESHOLD:
            name, desc, is_new = self._drift.propose(query, roster)
            if is_new and not self._roster.exists(name):
                self._roster.add(name, desc)  # persist the new agent
                agent = self._agent(name)
                answer, audit_note, used = agent.answer(query)
                return AgentResult(
                    subject=name, answer=answer, handoffs=[f"＋new:{name}"],
                    audit_note=audit_note, used_memory=used, created=True, fit=route.fit,
                )
            # Drift auditor decided an existing subject actually fits.
            route = route.__class__(primary=name, fit=route.fit, overlaps=route.overlaps, reason="drift->reuse")

        chain = [route.primary]
        agent = self._agent(route.primary)

        # ---- handoff: the routed specialist decides the query isn't theirs ----
        if not agent.in_scope(query):
            candidates = route.overlaps + [s for s in roster if s not in route.overlaps]
            for other in candidates:
                if other != route.primary and self._agent(other).in_scope(query):
                    chain.append(other)
                    agent = self._agent(other)
                    break

        # ---- overlap: gather consult context from touched specialists ----
        consult = []
        for other in route.overlaps:
            if other != agent.subject:
                consult.extend(self._mem.recall(other, query, limit=2))

        answer, audit_note, used = agent.answer(query, consult=consult)
        return AgentResult(
            subject=agent.subject, answer=answer, handoffs=chain,
            audit_note=audit_note, used_memory=used, created=False, fit=route.fit,
        )


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _render(q: str, r: AgentResult) -> None:
    hop = " -> ".join(r.handoffs)
    tag = "  [NEW AGENT SPAWNED]" if r.created else ""
    print(f"\n[agent] {hop}{tag}   (fit {r.fit:.2f}, recalled {r.used_memory} past turn(s), audit: {r.audit_note})")
    print(f"[answer] {r.answer}\n")


DEMO = [
    "What's a quick pasta recipe for two?",
    "How much protein is in that pasta dish, and does it fit a muscle-gain diet?",  # cooking<->fitness overlap
    "My tomato plants have yellow leaves — what's wrong?",                          # DRIFT -> spawns 'gardening'
    "How often should I water those tomato plants?",                               # routes back to new agent, uses memory
    "Remind me what pasta recipe you gave me earlier.",                            # memory recall
]


def main() -> None:
    orch = Orchestrator()
    print(f"Dynamic subject-memory agents ready. Memory backend: {orch.memory_kind}.")
    print(f"Starting subjects: {', '.join(orch.subjects())}")

    if "--demo" in sys.argv:
        for q in DEMO:
            print(f"\n> {q}")
            _render(q, orch.handle(q))
        print(f"Final roster: {', '.join(orch.subjects())}")
        return

    print("Type a question (or 'quit'). New specialists appear automatically on drift.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break
        _render(q, orch.handle(q))


if __name__ == "__main__":
    main()
