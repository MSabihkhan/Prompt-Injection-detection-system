-- PromptGuard feedback table.
-- Run in Supabase SQL Editor (Project -> SQL Editor -> New query -> paste -> Run).

create table if not exists feedback (
    id bigint generated always as identity primary key,
    text text not null,
    predicted_label smallint not null check (predicted_label in (0, 1)),
    correct_label smallint not null check (correct_label in (0, 1)),
    model_version text,
    created_at timestamptz not null default now()
);

create index if not exists feedback_created_at_idx on feedback (created_at);

-- RLS on, no policies: only the service_role key (used server-side by the API) can read/write.
-- The anon/public key gets nothing. Never expose the service_role key to a client.
alter table feedback enable row level security;

-- service_role bypasses RLS but still needs explicit table/sequence privileges on some projects.
grant select, insert on public.feedback to service_role;
grant usage, select on sequence feedback_id_seq to service_role;
