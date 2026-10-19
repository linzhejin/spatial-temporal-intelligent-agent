CREATE TABLE IF NOT EXISTS route_proof (
    proof_id UUID PRIMARY KEY,
    owner_id UUID NOT NULL REFERENCES conversation(owner_id) ON DELETE CASCADE,
    task_id UUID REFERENCES task(task_id) ON DELETE CASCADE,
    task_revision BIGINT CHECK (task_revision >= 0),
    semantic_digest TEXT NOT NULL CHECK (semantic_digest ~ '^[0-9a-f]{64}$'),
    graph_version TEXT NOT NULL CHECK (length(btrim(graph_version)) > 0),
    event_version TEXT NOT NULL CHECK (length(btrim(event_version)) > 0),
    selected_edges JSONB NOT NULL CHECK (
        jsonb_typeof(selected_edges) = 'array' AND jsonb_array_length(selected_edges) > 0),
    closed_edge_ids JSONB NOT NULL CHECK (jsonb_typeof(closed_edge_ids) = 'array'),
    verified BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    invalidated_at TIMESTAMPTZ,
    invalidation_reason TEXT,
    CHECK ((task_id IS NULL) = (task_revision IS NULL)),
    CHECK (expires_at > created_at),
    CHECK (invalidation_reason IS NULL OR invalidated_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS route_proof_task_idx ON route_proof(task_id);
CREATE INDEX IF NOT EXISTS route_proof_expiry_idx ON route_proof(expires_at);
