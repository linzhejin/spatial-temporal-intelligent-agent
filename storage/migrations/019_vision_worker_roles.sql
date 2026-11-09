-- Separate the resource-limited public trial worker from campus inspection.
ALTER TABLE manager_vision_worker
    ADD COLUMN IF NOT EXISTS worker_role TEXT NOT NULL DEFAULT 'shared'
        CHECK (worker_role IN ('shared', 'operational', 'trial'));
