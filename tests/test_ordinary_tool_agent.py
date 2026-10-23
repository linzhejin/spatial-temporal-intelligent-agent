from copy import deepcopy
from types import SimpleNamespace

from experiments.baselines.ordinary_tool_agent import run_conversation


class FakeMessage:
    def __init__(self, content="", calls=None):
        self.content = content
        self.tool_calls = calls or []

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content,
                "tool_calls": [{"id": call.id, "type": "function",
                                "function": {"name": call.function.name,
                                             "arguments": call.function.arguments}}
                               for call in self.tool_calls]}


class FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        index = len(self.requests)
        call = SimpleNamespace(id="call-1", function=SimpleNamespace(
            name="resolve_poi", arguments='{"query":"桂园"}'))
        msg = FakeMessage(calls=[call]) if index == 1 else FakeMessage("已处理")
        return SimpleNamespace(model="resolved-deepseek-model",
                               choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10,
                                                     completion_tokens=3,
                                                     prompt_cache_hit_tokens=8,
                                                     prompt_cache_miss_tokens=2))


def test_ordinary_baseline_uses_same_tools_and_keeps_transcript():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    executed = []

    def execute(name, args, context):
        executed.append((name, args, context))
        return {"name": "桂园"}, {"response_kind": "candidates"}

    result = run_conversation(["去桂园", "改成骑行"], client=client,
                              model="same-deepseek-model", tool_executor=execute,
                              context={"origin": "map"})
    assert result["tool_calls"] == 1
    assert result["input_tokens"] == 30
    assert executed == [("resolve_poi", {"query": "桂园"},
                         {"origin": "map", "query": "去桂园"})]
    assert completions.requests[1]["messages"][-1]["role"] == "tool"
    assert any(message.get("content") == "去桂园"
               for message in completions.requests[2]["messages"])
    assert all(request["tools"] for request in completions.requests)
    assert [turn["status"] for turn in result["turns"]] == ["completed", "completed"]
    assert result["measured_usage"]["requests"] == 3
    assert result["measured_usage"]["input_tokens"] == 30
    assert result["measured_usage"]["output_tokens"] == 9
    assert result["measured_usage"]["cache_hit_input_tokens"] == 24
    assert result["measured_usage"]["cache_miss_input_tokens"] == 6
    assert result["measured_usage"]["cache_accounted"] is True
    assert result["measured_usage"]["elapsed_ms"] >= 0
    assert result["condition"] == "ordinary_tool_agent"
    assert result["resolved_model_id"] == "resolved-deepseek-model"
    assert result["run_evidence"][0]["kind"] == "run"
    assert result["run_evidence"][0]["enabled_tool_names"]
    assert len([r for r in result["run_evidence"] if r["kind"] == "request"]) == 3
    assert [r["name"] for r in result["run_evidence"] if r["kind"] == "tool"] == ["resolve_poi"]


def test_ordinary_baseline_records_model_failure_without_claiming_completion():
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("private upstream response")

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    result = run_conversation(["去图书馆"], client=client, model="same-model")
    assert result["turns"][0]["status"] == "model_error"
    assert result["turns"][0]["message"] == ""
    assert result["turns"][0]["tool_trace"] == [
        {"outcome": "model_error", "error_type": "RuntimeError"}
    ]
    assert result["measured_usage"]["requests"] == 1
    assert result["resolved_model_id"] is None
    assert result["run_evidence"][1]["resolved_model_id"] is None


def test_ordinary_baseline_keeps_weather_observation_separate_from_reply():
    class WeatherCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                call = SimpleNamespace(id="weather-1", function=SimpleNamespace(
                    name="get_weather", arguments="{}"))
                msg = FakeMessage(calls=[call])
            else:
                msg = FakeMessage("路线已规划，天气晴。")
            return SimpleNamespace(model="deepseek-flash",
                                   choices=[SimpleNamespace(message=msg)],
                                   usage=SimpleNamespace(prompt_tokens=10,
                                                         completion_tokens=3))

    client = SimpleNamespace(chat=SimpleNamespace(completions=WeatherCompletions()))
    result = run_conversation(["查天气"], client=client, model="deepseek-flash",
                              tool_executor=lambda *_: ({"weather": "晴",
                                                        "temperature": 22,
                                                        "private": "not recorded"}, None))
    assert result["turns"][0]["weather_observation"] == {
        "weather": "晴", "temperature": 22,
    }
    assert "private" not in str(result["turns"][0]["weather_observation"])
