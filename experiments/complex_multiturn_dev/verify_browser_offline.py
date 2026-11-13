"""Check browser/worker bridge with a fake agent and a new isolated database."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from experiments.complex_multiturn_dev.browser import BrowserCases
from experiments.complex_multiturn_dev.run_dev import OUTPUT, ROOT, _create_test_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    name, db_url = _create_test_database(dotenv_values(args.env_file).get("DATABASE_URL"))
    os.environ.update(DATABASE_URL=db_url, TEST_DATABASE_URL=db_url, AGENT_DIALOGUE_MODE="model")
    from storage import database
    database.initialize(db_url)
    from app import create_app
    from agents.workflow import build_agent_workflow
    from langgraph.checkpoint.postgres import PostgresSaver

    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=db_url)
    def fake_agent(query, *, context, **_kwargs):
        return {"response_kind": "chat", "message": f"隔离验收：{query}", "turns": 1}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    screenshot = OUTPUT / ("offline_browser_" + uuid4().hex[:8] + ".png")
    with PostgresSaver.from_conn_string(db_url) as saver:
        saver.setup()
        workflow = build_agent_workflow(saver, agent=fake_agent, database_url=db_url)
        host = BrowserCases(ROOT, db_url, workflow)
        try:
            browser = host.new_case(app.test_client())
            evidence = []
            for query in ("第一轮桥接检查", "第二轮桥接检查"):
                run, body, ui = browser.turn(query)
                if run.get("status") != "completed" or not ui["chat_visible"] or ui.get("page_errors"):
                    raise RuntimeError("offline browser bridge did not finish")
                if body.get("query") != query:
                    raise RuntimeError("browser submitted different text")
                evidence.append(ui)
            browser.close(screenshot)
        finally:
            host.close()
    print(json.dumps({"scope": "offline_fake_agent_not_real_model", "database": name,
                      "turns": len(evidence), "screenshot": str(screenshot)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
