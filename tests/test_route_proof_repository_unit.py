import importlib.util
import uuid

import pytest


def repository():
    assert importlib.util.find_spec('storage.route_proof_repository') is not None, 'route proof persistence is missing'
    from storage import route_proof_repository
    return route_proof_repository


def arguments():
    return dict(owner_id=str(uuid.uuid4()), semantic_digest='a' * 64,
                graph_version='graph-1', event_version='events-1',
                selected_edges=[(1, 'node-2', 0)], closed_edge_ids=[[4, 'closed-1', 0]], ttl_seconds=60)


@pytest.mark.parametrize('field,value', [
    ('owner_id', 'not-a-uuid'), ('semantic_digest', 'A' * 64),
    ('semantic_digest', 'abc'), ('graph_version', ''), ('event_version', ''),
    ('selected_edges', []), ('selected_edges', [[1, 2]]),
    ('selected_edges', [[True, 2, 0]]), ('selected_edges', [[{}, 2, 0]]),
    ('selected_edges', [[float('nan'), 2, 0]]),
    ('closed_edge_ids', [None]), ('closed_edge_ids', 'edge'), ('closed_edge_ids', [[1, 2]]),
    ('closed_edge_ids', [[1, 2, False]]),
    ('event_snapshot', [None]), ('event_snapshot', [{'event_id': 'evt', 'fingerprint': 'bad'}]),
    ('ttl_seconds', 0), ('ttl_seconds', -1), ('ttl_seconds', True),
    ('ttl_seconds', float('inf')), ('task_id', str(uuid.uuid4())),
    ('task_revision', 0),
])
def test_create_rejects_malformed_input_before_connect(field, value, monkeypatch):
    repo = repository()
    def unexpected_connect(*args, **kwargs):
        pytest.fail('malformed proof reached database')
    monkeypatch.setattr(repo.database, 'connect', unexpected_connect)
    args = arguments()
    args[field] = value
    with pytest.raises(ValueError):
        repo.create_verified_proof(None, **args)


def test_missing_database_configuration_fails_closed(monkeypatch):
    repo = repository()
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(RuntimeError, match='DATABASE_URL'):
        repo.create_verified_proof(None, **arguments())


def test_load_exposes_no_caller_edge_data():
    import inspect
    repo = repository()
    names = inspect.signature(repo.load_verified_proof).parameters
    assert 'selected_edges' not in names
    assert 'closed_edge_ids' not in names


@pytest.mark.parametrize('field,value', [('proof_id','bad'), ('owner_id','bad'),
    ('semantic_digest','F'*64), ('graph_version',''), ('event_version',''),
    ('task_revision',True), ('task_revision',-1), ('task_id',str(uuid.uuid4()))])
def test_load_rejects_malformed_input(field, value, monkeypatch):
    repo = repository()
    monkeypatch.setattr(repo.database, 'connect', lambda *a, **k: pytest.fail('invalid load reached DB'))
    args = arguments()
    for key in ('selected_edges','closed_edge_ids','ttl_seconds'):
        args.pop(key)
    args['proof_id'] = str(uuid.uuid4())
    args[field] = value
    with pytest.raises(ValueError):
        repo.load_verified_proof(None, **args)


def test_load_without_database_configuration_fails_closed(monkeypatch):
    repo = repository()
    monkeypatch.delenv('DATABASE_URL', raising=False)
    args = arguments()
    for key in ('selected_edges', 'closed_edge_ids', 'ttl_seconds'):
        args.pop(key)
    with pytest.raises(RuntimeError, match='DATABASE_URL'):
        repo.load_verified_proof(None, proof_id=str(uuid.uuid4()), **args)
