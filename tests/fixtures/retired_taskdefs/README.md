# Retired ECS task-definition templates -- frozen reference data

These are `deploy/ecs/taskdef.*.json.tpl` exactly as they stood at `f9e953d^`, the commit before M6
replaced them with `memrank/placement/cloud.py`. They are **not** live configuration and nothing in
`memrank/` reads them.

They exist because replacing them lost things, repeatedly and quietly:

| lost | how it was found |
|---|---|
| TEI pulled from ghcr.io instead of the ECR mirror | a question about public images |
| one global image tag instead of one per engine | a question about cross-engine sweeps |
| mem0's `/configure` used as every engine's health check | a hindsight task stuck in PENDING |
| `ATOMICMEMORY_API_KEY` / `CORE_API_KEY` | a live 401 |
| `{PREFIX}_TIMEOUT_S`, image digests, provenance fields | the enumeration that followed |

Four of those were silent. `tests/placement/test_template_parity.py` compares the env-var names
these templates set against what the renderer emits, so a sixth loss fails a test rather than a
paid task.

**They are frozen, and that cuts both ways.** If an engine's real requirements change, the fixture
becomes wrong in the other direction -- the test would demand a variable nobody needs any more. Fix
it by editing the fixture *and* saying why in the same commit, never by widening the allowlist
without a reason.
