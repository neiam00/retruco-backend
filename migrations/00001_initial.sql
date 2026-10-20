-- +goose Up
CREATE TABLE users (
    id text PRIMARY KEY,
    name text NOT NULL,
    username text UNIQUE,
    password_hash text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((username IS NULL) = (password_hash IS NULL))
);

CREATE TABLE sessions (
    token_hash text PRIMARY KEY,
    user_id text NOT NULL REFERENCES users(id),
    expires_at timestamptz NOT NULL
);
CREATE INDEX sessions_expiry_idx ON sessions(expires_at);

CREATE TABLE matches (
    code text PRIMARY KEY,
    payload jsonb NOT NULL,
    finished boolean NOT NULL DEFAULT false,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE match_members (
    match_code text NOT NULL REFERENCES matches(code) ON DELETE CASCADE,
    user_id text NOT NULL REFERENCES users(id),
    PRIMARY KEY (match_code, user_id)
);
CREATE INDEX match_members_user_idx ON match_members(user_id);

-- +goose Down
DROP TABLE match_members;
DROP TABLE matches;
DROP TABLE sessions;
DROP TABLE users;
