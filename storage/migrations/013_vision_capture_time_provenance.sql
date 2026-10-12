ALTER TABLE manager_vision_job
    ADD COLUMN IF NOT EXISTS capture_time_source TEXT NOT NULL DEFAULT 'unknown';

ALTER TABLE manager_vision_job
    ADD COLUMN IF NOT EXISTS capture_timezone TEXT;

ALTER TABLE manager_vision_job
    DROP CONSTRAINT IF EXISTS manager_vision_job_capture_time_source_check;

ALTER TABLE manager_vision_job
    ADD CONSTRAINT manager_vision_job_capture_time_source_check
    CHECK (capture_time_source IN ('unknown', 'manual', 'dji_srt'));
