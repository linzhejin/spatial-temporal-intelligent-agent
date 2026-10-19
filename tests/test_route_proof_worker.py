"""Trusted workflow identity must survive planner tool dispatch (mock DB input)."""
import uuid
from langgraph.checkpoint.memory import InMemorySaver

from agents import workflow, planner
from tests.test_worker_runtime_context import _verified_route_response


def test_workflow_passes_db_identity_separately_from_user_context(monkeypatch):
    identity = {"owner_id": str(uuid.uuid4()), "task_id": str(uuid.uuid4()), "task_revision": 3}
    monkeypatch.setattr(workflow, "append_run_event", lambda *_a, **_kw: {"stored": True})
    monkeypatch.setattr(workflow, "get_run_input", lambda *_a: {
        **identity, "query": "从珞珈门到樱顶", "history": [], "task": {}})
    observed = {}
    def agent(*_a, **kwargs):
        observed.update(kwargs)
        return _verified_route_response("已规划")
    graph = workflow.build_agent_workflow(InMemorySaver(), agent=agent, database_url="trusted-worker-db")
    graph.invoke({"run_id": "proof-worker"}, {"configurable": {"thread_id": "proof-worker"}},
                 context={"worker_id": "worker"})
    assert observed["proof_identity"] == {**identity, "database_url": "trusted-worker-db"}


def test_planner_forwards_trusted_identity_to_direct_tool(monkeypatch):
    observed = []
    identity = {"database_url": "trusted-worker-db", "owner_id": str(uuid.uuid4()),
                "task_id": str(uuid.uuid4()), "task_revision": 0}
    def execute(_name, _args, context):
        observed.append(context)
        return {}, None
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    # Deterministic correction branch avoids LLM and preserves the explicit identity.
    from tests.test_route_reuse import _state, _graph
    planner.run_agent("改成骑行", context={"previous_route_state": _state(_graph())},
                      proof_identity=identity)
    assert observed[0]["_route_proof_identity"] == identity


def test_user_context_and_uid_cannot_supply_proof_identity(monkeypatch):
    observed = []
    def execute(_name, _args, context):
        observed.append(context)
        return {}, None
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    from tests.test_route_reuse import _state, _graph
    planner.run_agent("改成骑行", uid="untrusted-client", context={
        "previous_route_state": _state(_graph()),
        "_route_proof_identity": {"owner_id": "forged"}})
    assert not observed[0].get("_route_proof_identity")
