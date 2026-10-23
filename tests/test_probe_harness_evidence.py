"""The Harness request header is not evidence of the provider's response model."""

from scripts.analysis.probe_harness_agent import (
    _attribute_transport_models, _request_evidence,
)


def test_requested_model_is_not_recorded_as_resolved_model():
    record = _request_evidence(
        "deepseek-flash", {"inputTokens": 12, "outputTokens": 3}, "schema-hash")

    assert record["requested_model_id"] == "deepseek-flash"
    assert record["resolved_model_id"] is None
    assert record["input_tokens"] == 12
    assert record["output_tokens"] == 3


def test_transport_model_requires_exactly_one_complete_http_stream_per_request():
    request = _request_evidence("alias", {}, "schema-hash")
    complete = {"http_status": 200, "stream_complete": True,
                "response_model_id": "resolved-1"}
    assert _attribute_transport_models([request], [complete]) == (True, "resolved-1")
    assert request["resolved_model_id"] == "resolved-1"

    for attempts in ([], [complete, complete],
                     [{"http_status": 200, "stream_complete": False,
                       "response_model_id": "resolved-1"}],
                     [{"http_status": 200, "stream_complete": True,
                       "response_model_id": None}]):
        request = _request_evidence("alias", {}, "schema-hash")
        assert _attribute_transport_models([request], attempts) == (False, None)
        assert request["resolved_model_id"] is None
