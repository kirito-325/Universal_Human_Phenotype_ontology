import json

from hpo_agent.schemas import Settings
from hpo_agent.session import ResearchSession
from hpo_agent.storage import AuditTrail


async def test_checkpoint_does_not_reemit_live_events(phenotype, tmp_path):
    audit = AuditTrail(tmp_path / "attempt" / "events.jsonl")
    session = ResearchSession(phenotype=phenotype, engine=None, settings=Settings(workflow_mode="component_test", ),
                              output=tmp_path, attempt=tmp_path / "attempt", audit=audit)
    audit.append({"event": "agent_started", "role": "research", "invocation_id": "one"})
    await session.checkpoint()
    await session.checkpoint()
    rows = [json.loads(line) for line in audit.event_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["invocation_id"] == "one"
