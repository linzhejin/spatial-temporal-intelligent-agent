ALTER TABLE route_proof
    ADD COLUMN IF NOT EXISTS event_snapshot JSONB NOT NULL DEFAULT '[]'::jsonb
    CHECK (jsonb_typeof(event_snapshot) = 'array');
