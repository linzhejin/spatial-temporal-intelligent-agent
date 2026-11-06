-- Preserve old task-bound runs while allowing a message to be understood first.
ALTER TABLE run ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversation(conversation_id) ON DELETE CASCADE;
ALTER TABLE run ADD COLUMN IF NOT EXISTS input_message_seq BIGINT;
ALTER TABLE run ADD COLUMN IF NOT EXISTS processing_stage TEXT NOT NULL DEFAULT 'queued';
ALTER TABLE run ADD COLUMN IF NOT EXISTS model_led BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE run ALTER COLUMN task_id DROP NOT NULL;
ALTER TABLE run ALTER COLUMN task_revision DROP NOT NULL;

UPDATE run r SET conversation_id=t.conversation_id
FROM task t WHERE r.task_id=t.task_id AND r.conversation_id IS NULL;
UPDATE run r SET input_message_seq=m.seq
FROM conversation_message m
WHERE r.input_message_seq IS NULL AND m.conversation_id=r.conversation_id
  AND m.role='user' AND m.metadata->>'request_id'=r.idempotency_key;
UPDATE run SET processing_stage=status WHERE processing_stage='queued' AND status<>'queued';
ALTER TABLE run ALTER COLUMN conversation_id SET NOT NULL;
ALTER TABLE run ADD CONSTRAINT run_task_binding_complete CHECK ((task_id IS NULL) = (task_revision IS NULL));
ALTER TABLE run ADD CONSTRAINT run_model_input_present CHECK (NOT model_led OR input_message_seq IS NOT NULL);

CREATE INDEX IF NOT EXISTS run_conversation_order_idx
    ON run (conversation_id, input_message_seq, created_at)
    WHERE status IN ('queued', 'running');

-- Older releases still insert runs with only a task ID during a rollback.
CREATE OR REPLACE FUNCTION fill_run_conversation() RETURNS trigger AS $$
BEGIN
    IF NEW.conversation_id IS NULL AND NEW.task_id IS NOT NULL THEN
        SELECT conversation_id INTO NEW.conversation_id FROM task WHERE task_id=NEW.task_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS run_fill_conversation ON run;
CREATE TRIGGER run_fill_conversation BEFORE INSERT OR UPDATE OF task_id ON run
FOR EACH ROW EXECUTE FUNCTION fill_run_conversation();
