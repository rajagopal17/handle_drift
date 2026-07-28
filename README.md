# Subject-Memory Agents

Multiple subject-specialist agents answer user queries. **Each agent owns its
own memory** and only answers its subject. When a query lands on the wrong
specialist it **hands off**; when a query **overlaps** subjects, the answering
agent is handed the other specialist's relevant memory as consult context.
Every answer is **audited and self-corrected** before being stored.

## How it works

```
query ─▶ Router (LLM)  ──▶ primary subject + overlaps
             │
             ▼
       primary agent ── in_scope? ──no──▶ handoff to first specialist that claims it
             │yes
             ▼
   answer using OWN memory (earlier Q&A) + consult context from overlapping agents
             │
             ▼
     Auditor (LLM) reviews ──not ok──▶ agent revises once
             │
             ▼
     store turn in this agent's memory  (Redis, or local JSON fallback)
```

## Files

| File | Role |
|------|------|
| `main.py`   | Orchestrator: routing, handoff, overlap-consult, REPL/demo |
| `agents.py` | `SubjectAgent`, `Router`, `Auditor`, and the subject roster |
| `memory.py` | Per-subject memory with Redis backend + JSON-file fallback |
| `llm.py`    | Thin OpenAI wrapper (`chat` / `json`) reading `.env` |
| `.env`      | `OPENAI_API_KEY`, `OPENAI_MODEL`, `REDIS_URL`, `REDIS_KEY_PREFIX` |

## Run

```bash
python main.py --demo   # scripted queries showing routing, handoff, memory recall
python main.py          # interactive REPL (type questions, 'quit' to exit)
```

## Configuration

- **Add a subject**: add one line to `SUBJECTS` in `agents.py`. It becomes
  routable and gets its own memory namespace automatically.
- **Memory backend**: set `REDIS_URL` to a reachable server to use Redis;
  otherwise the system transparently falls back to `memory_store.json`.
