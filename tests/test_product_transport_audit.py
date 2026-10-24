"""The product pilot must inject the audit client into the real planner factory."""

import httpx2

from experiments.baselines.openai_transport_audit import OpenAITransportAudit
from scripts.analysis.probe_product_live_case import _make_audited_planner_factory


def test_product_pilot_factory_passes_audited_http_client():
    audit = OpenAITransportAudit()
    created_clients = []
    received = []

    def original_factory(*, http_client):
        received.append(http_client)
        return "model-client"

    factory = _make_audited_planner_factory(audit, created_clients, original_factory)
    try:
        assert factory() == "model-client"
        assert len(created_clients) == 1
        assert received == created_clients
        assert isinstance(created_clients[0], httpx2.Client)
        assert audit.request_hook in created_clients[0].event_hooks["request"]
        assert audit.response_hook in created_clients[0].event_hooks["response"]
    finally:
        for client in created_clients:
            client.close()
