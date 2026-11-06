"""Migrate a historical task-bound run without touching the business schema."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
import pytest

from agents.state_models import TaskState
from storage import database
from storage.task_repository import get_run_input

pytestmark = pytest.mark.usefixtures("isolated_test_database")
DATABASE_URL = None


def test_historical_runs_are_backfilled_and_legacy_inserts_still_work():
    schema = "migration_test_" + uuid4().hex
    scoped_url = make_conninfo(DATABASE_URL, options=f"-c search_path={schema}")
    with database.connect(DATABASE_URL) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {} ").format(sql.Identifier(schema)))
    try:
        migrations = Path(__file__).resolve().parents[1] / "storage" / "migrations"
        with database.connect(scoped_url) as conn:
            for path in sorted(migrations.glob("*.sql")):
                if path.name >= "016_":
                    break
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO agent_schema_migration(version) VALUES (%s)", (path.name,))
            owner, conversation, task, old_run = (str(uuid4()) for _ in range(4))
            state = TaskState(task_id=task, conversation_id=conversation, status="queued").model_dump(mode="json")
            conn.execute("INSERT INTO conversation(conversation_id,owner_id,credential_hash,revision) VALUES (%s,%s,%s,1)",
                         (conversation, owner, b"test-only"))
            conn.execute("INSERT INTO task(task_id,conversation_id,status,state) VALUES (%s,%s,'queued',%s)",
                         (task, conversation, Jsonb(state)))
            conn.execute("INSERT INTO conversation_message(conversation_id,seq,message_id,task_id,role,content,metadata)"
                         " VALUES (%s,1,%s,%s,'user','historical text',%s)",
                         (conversation, str(uuid4()), task, Jsonb({"request_id": "historical-request"})))
            conn.execute("INSERT INTO run(run_id,task_id,task_revision,status,deadline_at,idempotency_key)"
                         " VALUES (%s,%s,0,'queued',%s,'historical-request')",
                         (old_run, task, datetime.now(timezone.utc) + timedelta(minutes=1)))
        database.initialize(scoped_url)
        with database.connect(scoped_url) as conn:
            row = conn.execute("SELECT conversation_id,input_message_seq,model_led FROM run WHERE run_id=%s",
                               (old_run,)).fetchone()
            assert str(row["conversation_id"]) == conversation
            assert row["input_message_seq"] == 1 and row["model_led"] is False
            legacy_run = str(uuid4())
            conn.execute("INSERT INTO run(run_id,task_id,task_revision,status,deadline_at,idempotency_key)"
                         " VALUES (%s,%s,0,'queued',%s,%s)",
                         (legacy_run, task, datetime.now(timezone.utc) + timedelta(minutes=1), str(uuid4())))
            assert str(conn.execute("SELECT conversation_id FROM run WHERE run_id=%s", (legacy_run,)).fetchone()["conversation_id"]) == conversation
        assert get_run_input(scoped_url, old_run)["query"] == "historical text"
    finally:
        with database.connect(DATABASE_URL) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
