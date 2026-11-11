-- Tables used by Airflow + Grafana. Safe to run more than once.
-- NOTE: no prompt text is stored in predictions (project rule). feedback keeps text because
-- retraining needs it, and users send it on purpose via /feedback.

create table if not exists predictions (
    id            bigint generated always as identity primary key,
    created_at    timestamptz not null default now(),
    endpoint      text        not null default 'classify',
    model_version text,
    label         smallint    not null check (label in (0, 1)),
    score         double precision not null,
    threshold     double precision,
    text_length   integer     not null,          -- characters, NOT the text itself
    latency_ms    double precision
);
create index if not exists predictions_created_at_idx on predictions (created_at);

create table if not exists feedback (
    id              bigint generated always as identity primary key,
    created_at      timestamptz not null default now(),
    text            text        not null,
    predicted_label smallint    not null check (predicted_label in (0, 1)),
    correct_label   smallint    not null check (correct_label in (0, 1)),
    model_version   text
);
create index if not exists feedback_created_at_idx on feedback (created_at);

create table if not exists drift_runs (
    id             bigint generated always as identity primary key,
    created_at     timestamptz not null default now(),
    status         text        not null,          -- ok | insufficient_data
    drift_detected boolean     not null,
    n_reference    integer,
    n_current      integer,
    length_ks_p    double precision,
    length_psi     double precision,
    score_ks_p     double precision,
    score_psi      double precision,
    details        jsonb
);
create index if not exists drift_runs_created_at_idx on drift_runs (created_at);
