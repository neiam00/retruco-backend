-- +goose Up
-- Stop all API writers before migrating; an old binary cannot write this format.
-- For rollback, keep writers stopped, apply Down, then start the old binary.
-- Keep the complete retained histories, but stop replacing their TOAST values
-- whenever an unrelated card move changes the compact game payload.
-- SQL NULL means the original document had no such key; JSON null is preserved.
ALTER TABLE matches ADD COLUMN chat jsonb;
ALTER TABLE matches ADD COLUMN canto_history jsonb;

-- Verify every reconstructed JSON document inside the same transaction. An
-- unexpected shape or data mismatch aborts the migration without committing a
-- partially converted database. No original payloads leave PostgreSQL.
-- +goose StatementBegin
DO $$
DECLARE mismatches bigint;
BEGIN
    WITH original AS MATERIALIZED (SELECT code, payload FROM matches),
    updated AS (
        UPDATE matches m
        SET chat = original.payload->'Chat',
            canto_history = original.payload->'CantoHistory',
            payload = original.payload - 'Chat' - 'CantoHistory'
        FROM original
        WHERE m.code = original.code
        RETURNING m.code, m.payload, m.chat, m.canto_history
    )
    SELECT count(*) INTO mismatches
    FROM original FULL JOIN updated USING (code)
    WHERE original.payload IS DISTINCT FROM (
        updated.payload
        || CASE WHEN updated.chat IS NULL THEN '{}'::jsonb ELSE jsonb_build_object('Chat', updated.chat) END
        || CASE WHEN updated.canto_history IS NULL THEN '{}'::jsonb ELSE jsonb_build_object('CantoHistory', updated.canto_history) END
    );
    IF mismatches <> 0 THEN
        RAISE EXCEPTION 'Match history migration roundtrip failed for % records', mismatches;
    END IF;
END $$;
-- +goose StatementEnd

-- +goose Down
-- Materialize the latest values, including writes made after the migration.
-- This permits an old binary to restore the exact current snapshot after down.
UPDATE matches
SET payload = payload
    || CASE WHEN chat IS NULL THEN '{}'::jsonb ELSE jsonb_build_object('Chat', chat) END
    || CASE WHEN canto_history IS NULL THEN '{}'::jsonb ELSE jsonb_build_object('CantoHistory', canto_history) END;

ALTER TABLE matches DROP COLUMN canto_history;
ALTER TABLE matches DROP COLUMN chat;
