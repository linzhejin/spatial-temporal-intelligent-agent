"""Pure diagnostic contracts: no server, model, database, or dotenv reads."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments import local_vision_agent_load as load


class MetricsTests(unittest.TestCase):
    def test_percentiles_keep_small_sample_count_and_raw_values(self):
        stats = load.distribution([4, None, 1, 2, 3])
        self.assertEqual(stats, {"n": 4, "values_s": [4, 1, 2, 3],
                                 "p50_s": 2.5, "p95_s": 3.85})
        self.assertEqual(load.distribution([]), {
            "n": 0, "values_s": [], "p50_s": None, "p95_s": None})

    def test_summary_preserves_partial_errors_and_missing_terminal_samples(self):
        rows = [{"status": "completed", "accepted_latency_s": 1,
                 "terminal_latency_s": 5, "elapsed_s": 5},
                {"status": "partial", "terminal_latency_s": 7, "elapsed_s": 7},
                {"status": "timeout", "elapsed_s": 9, "error_type": "TimeoutError"}]
        report = load.summarize_metrics(rows)
        self.assertEqual(report["status_counts"], {"completed": 1, "partial": 1, "timeout": 1})
        self.assertEqual(report["terminal_latency_s"]["n"], 2)
        self.assertEqual(report["elapsed_s"]["n"], 3)
        self.assertEqual(report["failure_count"], 2)

    def test_overlap_uses_actual_analysis_interval(self):
        self.assertEqual(load.request_overlap(10, 20, 5, 25)["classification"], "full")
        partial = load.request_overlap(10, 20, 15, 30)
        self.assertEqual(partial["classification"], "partial")
        self.assertEqual(partial["fraction"], .5)
        self.assertEqual(partial["seconds"], 5)
        self.assertEqual(load.request_overlap(10, 20, 20, 25)["classification"], "none")
        self.assertEqual(load.request_overlap(10, 20, None, None)["classification"], "unknown")
        self.assertEqual(load.request_overlap(10, 20, 5, None)["classification"], "unknown")


class SafetyTests(unittest.TestCase):
    def test_rejects_remote_credentials_paths_and_malformed_ports(self):
        for url in ["https://127.0.0.1", "http://example.com", "http://127.0.0.1/x",
                    "http://u:p@localhost", "http://localhost:bad", "http://localhost/?q=1"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                load.canonical_service(url)
        self.assertEqual(load.canonical_service("http://localhost:5000/"), "http://127.0.0.1:5000")

    def test_session_disables_proxy_credentials_and_redirects(self):
        session = load.LocalSession()
        self.assertFalse(session.trust_env)
        self.assertFalse(session.allow_redirects)
        with self.assertRaises(ValueError):
            session.get("http://example.com")

    def test_redirect_is_rejected_without_following_location(self):
        response = type("Redirect", (), {"status_code": 302})()
        with load.LocalSession() as session, patch(
            "requests.Session.request", return_value=response,
        ) as request:
            with self.assertRaisesRegex(ValueError, "redirects"):
                session.get("http://127.0.0.1:5000/api/research/replay-readiness")
        self.assertFalse(request.call_args.kwargs["allow_redirects"])

    def test_existing_output_is_preserved_before_network_io(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model, video = folder / "model.onnx", folder / "video.mp4"
            model.write_bytes(b"not loaded")
            video.write_bytes(b"not decoded")
            output = folder / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("user result", encoding="utf-8")
            with patch.object(load, "LocalSession", side_effect=AssertionError("no network")):
                with self.assertRaises(FileExistsError):
                    load.main(["--service", "http://127.0.0.1:5000", "--model", str(model),
                               "--video", str(video), "--output", str(output)])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user result")

    def test_child_environment_drops_application_credentials(self):
        with patch.dict(load.os.environ, {"DATABASE_URL": "do-not-inherit",
                                         "DEEPSEEK_API_KEY": "do-not-inherit",
                                         "APPDATA": r"C:\Users\tester\AppData\Roaming"}):
            env = load.child_environment()
        self.assertNotIn("DATABASE_URL", env)
        self.assertNotIn("DEEPSEEK_API_KEY", env)
        self.assertEqual(env["APPDATA"], r"C:\Users\tester\AppData\Roaming")
        self.assertEqual(env["VISION_CPU_THREADS"], "1")
        self.assertEqual(env["PYTHON_DOTENV_DISABLED"], "1")

    def test_unproved_service_cannot_post(self):
        session = FakeSession(readiness=False)
        row = load.run_request("baseline", 1, load.REQUEST_BANK[0],
                               "http://127.0.0.1:5000", session_factory=lambda: session)
        self.assertEqual(row["status"], "client_error")
        self.assertEqual(row["error_type"], "ValueError")
        self.assertEqual(session.posts, [])

    def test_poll_keeps_partial_and_tool_evidence(self):
        session = FakeSession(runs=[
            {"status": "running", "events": [{"seq": 1, "event_type": "node_started"}]},
            {"status": "partial", "events": [{"seq": 2, "event_type": "agent_tool_call",
                "payload": {"tool_name": "plan_route", "outcome": "tool_error"}}],
             "error": {"code": "route_failed"}},
        ])
        row = load.run_request("loaded", 1, load.REQUEST_BANK[1],
                               "http://127.0.0.1:5000", poll_interval_s=0,
                               session_factory=lambda: session)
        self.assertEqual(row["status"], "partial")
        self.assertIsNotNone(row["first_running_latency_s"])
        self.assertIsNotNone(row["terminal_latency_s"])
        self.assertEqual(row["tool_names"], ["plan_route"])
        self.assertEqual(row["tool_events"][0]["outcome"], "tool_error")
        self.assertEqual(row["run_error_code"], "route_failed")

    def test_poll_timeout_preserves_last_known_running_state(self):
        session = FakeSession(runs=[{"status": "running", "events": []}])
        row = load.run_request("baseline", 1, load.REQUEST_BANK[0],
                               "http://127.0.0.1:5000", timeout_s=0, poll_interval_s=0,
                               session_factory=lambda: session)
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["last_observed_status"], "running")
        self.assertIsNone(row["terminal_latency_s"])

    def test_transport_failure_keeps_already_observed_tool_events(self):
        class ErrorSession(FakeSession):
            def get(self, url, **kwargs):
                if not url.endswith("replay-readiness") and getattr(self, "polled", False):
                    raise load.requests.ConnectionError("unlogged transport detail")
                if not url.endswith("replay-readiness"):
                    self.polled = True
                return super().get(url, **kwargs)

        session = ErrorSession(runs=[{"status": "running", "events": [
            {"seq": 1, "event_type": "agent_tool_call", "payload": {
                "tool_name": "get_weather", "outcome": "success"}},
        ]}])
        row = load.run_request("baseline", 1, load.REQUEST_BANK[3],
                               "http://127.0.0.1:5000", poll_interval_s=0,
                               session_factory=lambda: session)
        self.assertEqual(row["status"], "client_error")
        self.assertEqual(row["last_observed_status"], "running")
        self.assertEqual(row["tool_names"], ["get_weather"])
        self.assertNotIn("unlogged transport detail", str(row))


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self.data}


class FakeSession:
    def __init__(self, readiness=True, runs=None):
        self.readiness = readiness
        self.runs = iter(runs or [])
        self.posts = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def get(self, url, **_kwargs):
        if url.endswith("replay-readiness"):
            return Response({"status": "ready", "service": "local_development",
                             "database_isolated": self.readiness})
        return Response(next(self.runs))

    def post(self, url, **_kwargs):
        self.posts.append(url)
        return Response({"conversation_id": "conversation-1", "run_id": "run-1"})


if __name__ == "__main__":
    unittest.main()
