# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub:
**Security** tab → **Report a vulnerability** (private vulnerability reporting).

Include what you found, how to reproduce it, and the impact you expect. We aim to acknowledge reports within
7 days. This is a student project, so fixes are best-effort.

## Supported versions

Only the latest commit on `main` is supported.

## Security model

**Authentication.** All classification and drift endpoints require an `X-API-Key` header when
`PROMPTGUARD_API_KEY` is set. If it is unset, auth is off. That is for local development only and must
never be used in a deployment.

**Data handling.**
- Prompt text is **not stored** for normal classifications. The `predictions` table holds only the label,
  score, threshold, prompt length, latency and model version.
- The `feedback` table does store prompt text, because retraining needs labelled examples. Only rows deliberately
  selected for retraining go there.
- Requests are validated (Pydantic): single prompts are capped at 8,000 characters and batches at 64 prompts.

**Database.** The API talks to Supabase through its client library (parameterised PostgREST calls, no raw
SQL). The API's `service_role` key is granted only `select` and `insert` on the tables it uses, so nothing can
be updated or deleted through the API. Enable Row Level Security on every table
(`alter table <name> enable row level security;`) so the public `anon` key can read nothing.

**Secrets.** Credentials (Supabase key, DagsHub/MLflow token, API key) come only from environment
variables, a local `.env` file (gitignored), GitHub Actions secrets, or Azure Key Vault / Container App
secrets. Never commit them. `.env.example` lists the variable names with empty values. Deployment signs in
to Azure with OpenID Connect (federated credentials), so no long-lived Azure password is stored in GitHub.

**Model supply chain.** Models are loaded only from the project's own MLflow registry on DagsHub, or from a
directory exported from it at image build time. A model becomes `champion` only after passing the evaluation
gates in `app/eval.py`.

## Known limitations

- PromptGuard is a screening layer, not a guarantee. It targets direct English prompt injections. It will
  miss novel or obfuscated attacks and does not inspect model outputs. Use it alongside other controls.
- There is no rate limiting in the API itself. Put it behind a gateway or load balancer that provides one.
- A planned LLM-judge feature would send sampled prompts to a third-party LLM provider. Document and review
  that data flow before enabling it.
