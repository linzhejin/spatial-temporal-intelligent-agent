ALTER TABLE manager_vision_job
    ADD COLUMN IF NOT EXISTS camera_pose JSONB;
