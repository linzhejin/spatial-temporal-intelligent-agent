import uuid

import psycopg
import pytest

from storage import database
from tests.test_route_proof_repository_unit import arguments as unit_arguments, repository

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures('isolated_test_database')


@pytest.fixture(autouse=True)
def schema():
    database.initialize(DATABASE_URL)


def arguments():
    from storage.task_repository import create_conversation
    args = unit_arguments()
    args['owner_id'] = create_conversation(DATABASE_URL)[0]
    return args


def load_args(args, proof_id):
    return {**{k: v for k, v in args.items() if k not in
               ('selected_edges', 'closed_edge_ids', 'ttl_seconds', 'event_snapshot')}, 'proof_id': proof_id}


def task_binding():
    from storage.task_repository import create_conversation, create_task
    owner, _, conversation = create_conversation(DATABASE_URL)
    task = create_task(DATABASE_URL, owner, conversation, {})
    return owner, task


def test_migration_is_applied_and_idempotent():
    with database.connect(DATABASE_URL) as conn:
        assert conn.execute("SELECT 1 FROM agent_schema_migration WHERE version='014_route_proofs.sql'").fetchone()
        assert conn.execute("SELECT 1 FROM agent_schema_migration WHERE version='015_route_proof_event_snapshot.sql'").fetchone()
        migration = database._MIGRATIONS / '014_route_proofs.sql'
        conn.execute(migration.read_text(encoding='utf-8'))


def test_create_and_load_returns_exact_stored_edge_sequence():
    repo = repository()
    args = arguments()
    args['event_snapshot'] = [{'event_id': 'event-1', 'fingerprint': 'c' * 64}]
    args['selected_edges'] = [(1, 'node-2', 0), ('node-2', 3, 'parallel'), (1, 'node-2', 0)]
    proof_id = repo.create_verified_proof(DATABASE_URL, **args)
    assert str(uuid.UUID(proof_id)) == proof_id
    row = repo.load_verified_proof(DATABASE_URL, **load_args(args, proof_id))
    assert row['selected_edges'] == [[1, 'node-2', 0], ['node-2', 3, 'parallel'], [1, 'node-2', 0]]
    assert row['closed_edge_ids'] == args['closed_edge_ids']
    assert row['event_snapshot'] == args['event_snapshot']
    assert row['verified'] is True
    assert row['expires_at'] > row['created_at']


@pytest.mark.parametrize('field,value', [('owner_id',str(uuid.uuid4())),
    ('semantic_digest','b'*64), ('graph_version','graph-2'), ('event_version','events-2'),
    ('proof_id',str(uuid.uuid4()))])
def test_mismatched_binding_returns_no_proof(field, value):
    repo, args = repository(), arguments()
    proof_id = repo.create_verified_proof(DATABASE_URL, **args)
    query = load_args(args, proof_id)
    query[field] = value
    assert repo.load_verified_proof(DATABASE_URL, **query) is None


@pytest.mark.parametrize('update', [
    "created_at=now()-interval '2 minutes', expires_at=now()-interval '1 minute'",
    "invalidated_at=now(), invalidation_reason='graph retired'", "verified=false"])
def test_expired_invalidated_or_unverified_proof_is_unavailable(update):
    repo, args = repository(), arguments()
    proof_id = repo.create_verified_proof(DATABASE_URL, **args)
    with database.connect(DATABASE_URL) as conn:
        conn.execute('UPDATE route_proof SET ' + update + ' WHERE proof_id=%s', (proof_id,))
    assert repo.load_verified_proof(DATABASE_URL, **load_args(args, proof_id)) is None


def test_task_proof_checks_owner_revision_and_current_database_revision():
    repo, args = repository(), arguments()
    owner, task = task_binding()
    args.update(owner_id=owner, task_id=task, task_revision=0)
    proof_id = repo.create_verified_proof(DATABASE_URL, **args)
    query = load_args(args, proof_id)
    assert repo.load_verified_proof(DATABASE_URL, **query)
    assert repo.load_verified_proof(DATABASE_URL, **{**query, 'task_revision': 1}) is None
    without_task = {k:v for k,v in query.items() if k not in ('task_id','task_revision')}
    assert repo.load_verified_proof(DATABASE_URL, **without_task) is None
    with database.connect(DATABASE_URL) as conn:
        conn.execute('UPDATE task SET revision=1 WHERE task_id=%s', (task,))
    assert repo.load_verified_proof(DATABASE_URL, **query) is None
    with pytest.raises(ValueError):
        repo.create_verified_proof(DATABASE_URL, **args)


def test_task_binding_rejects_foreign_owner_or_missing_task():
    repo, args = repository(), arguments()
    owner, task = task_binding()
    args.update(task_id=task, task_revision=0)
    with pytest.raises(ValueError):
        repo.create_verified_proof(DATABASE_URL, **args)
    args.update(owner_id=owner, task_id=str(uuid.uuid4()))
    with pytest.raises(ValueError):
        repo.create_verified_proof(DATABASE_URL, **args)


def test_database_enforces_task_foreign_key_and_revision_pair():
    repo, args = repository(), arguments()
    proof_id = repo.create_verified_proof(DATABASE_URL, **args)
    with pytest.raises(psycopg.errors.CheckViolation):
        with database.connect(DATABASE_URL) as conn:
            conn.execute('UPDATE route_proof SET task_revision=0 WHERE proof_id=%s', (proof_id,))
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with database.connect(DATABASE_URL) as conn:
            conn.execute('UPDATE route_proof SET task_id=%s, task_revision=0 WHERE proof_id=%s',
                         (str(uuid.uuid4()), proof_id))


def test_unregistered_owner_cannot_create_an_unbound_proof():
    repo, args = repository(), unit_arguments()
    with pytest.raises(ValueError, match='owner'):
        repo.create_verified_proof(DATABASE_URL, **args)
