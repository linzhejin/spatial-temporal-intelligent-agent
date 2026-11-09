-- Trial results must remain identifiable even if an older server is restored.
ALTER TABLE manager_vision_job
    ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'operational'
        CHECK (purpose IN ('operational', 'trial')),
    ADD COLUMN IF NOT EXISTS trial_feedback JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE OR REPLACE FUNCTION protect_vision_trial_job() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.purpose IS DISTINCT FROM OLD.purpose THEN
        RAISE EXCEPTION 'vision job purpose cannot be changed';
    END IF;
    IF NEW.purpose = 'trial' AND NEW.result IS NOT NULL THEN
        IF jsonb_typeof(NEW.result) <> 'object' THEN
            RAISE EXCEPTION 'vision trial result must be a JSON object';
        END IF;
        NEW.result := jsonb_set(NEW.result, '{research_replay}',
                                '{"mode":"trial"}'::jsonb, true);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER manager_vision_trial_guard
    BEFORE INSERT OR UPDATE ON manager_vision_job
    FOR EACH ROW EXECUTE FUNCTION protect_vision_trial_job();
