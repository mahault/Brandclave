# BrandClave Platform Architecture

Written 2026-07-27. This is the orientation document for anyone joining the
codebase (start here, then `REVIVAL-AND-COMMERCIAL-PLAN.md` for the why).

## What the platform is

BrandClave detects emerging hospitality demand before it becomes obvious,
translates it into brand concepts, and records its own predictions so their
accuracy becomes provable to capital (see the Signal Ledger). It is becoming
a multi-tenant product: people sign in, build concepts, and own their research.

```
sources.yaml registry ──> scrapers (content|metric) ──> raw_content / demand_metrics
                                   │                            │
                       ScrapingPOMDP picks sources     signal-to-city backbone
                                   │
                    NLP pipeline (Mistral embeddings, HDBSCAN)
                                   │
              trend_signals ── hotelier_moves ── city desires
                                   │
        Brand Blueprint pipeline (5 stages) ──> brand_blueprints (per user)
                                   │
                    Signal Ledger (sealed predictions ──> outcomes ──> KPIs)
```

## Ingestion

**Registry** — `configs/sources.yaml` is load-bearing, parsed by
`ingestion/registry.py` into `SourceSpec` records. Every source has:

- `kind`: `content` (text → `raw_content`) or `metric` (time series → `demand_metrics`)
- `status`: `active` (in the scraping rotation) / `planned` (catalogued, no class yet)
  / `blocked` (dead or ToS-blocked; runnable manually, never scheduled)
- `priority` and a free-form `config` block (queries, city lists, lookbacks)

The CLI (`scripts/run_crawlers.py`), the POMDP action space, scheduler fallbacks
and API defaults all derive from the registry. **Adding a source is one YAML
entry plus a small scraper class** — nothing else to touch.

**Scrapers** — subclass `ingestion/base_scraper.py:BaseScraper`, set
`source_name`/`source_type`, implement `scrape() -> list[RawContentCreate]`.
The base provides robots.txt checks, per-request pacing, resilient HTTP
(`ingestion/http_client.py`: timeouts, retry with backoff and jitter,
Retry-After), persistence with URL dedup, and job audit rows
(`processing_jobs`). A dead source logs and returns empty — a run never aborts
because one site failed. Metric sources subclass
`ingestion/metrics/base.py:MetricScraper`, which swaps persistence for an
upsert into `demand_metrics` while reusing all of the above.

**Notable sources**
- `bluesky` — consumer voice via AT Protocol search. Uses `api.bsky.app`;
  `public.api.bsky.app`'s searchPosts WAF-403s scripted clients (verified 2026-07-27).
- `wikimedia_pageviews` — daily Wikipedia pageviews per destination city
  (27 cities configured); free, needs a descriptive User-Agent. Best available
  Google Trends substitute and the geo-resolvable demand backbone.
- `reddit` — blocked (public JSON endpoint 403s since 2026-03; official free
  tier is non-commercial). Historical Reddit rows remain in the corpus.

**Scheduling** — APScheduler (`scheduler/scheduler.py`) runs one adaptive
scrape job (default every 30 min). `ScrapingPOMDP`
(`services/active_inference/scraping_pomdp.py`, PyMDP/JAX) picks the source by
Expected Free Energy over per-source productivity/freshness/error beliefs; its
action space is `active_sources() + wait`. Observations feed back from real
run results (`items_scraped`, novelty = saved/scraped dedup ratio). Beliefs are
in-memory only and reset on restart (persistence is a known gap).

## Data layer

SQLite at `data/brandclave.db` (SQLAlchemy 2.0, `db/models.py`) + ChromaDB for
vectors. Tables: `raw_content`, `trend_signals`, `hotelier_moves`,
`property_features`, `processing_jobs`, `brand_blueprints`,
`prediction_records` + `ledger_events` (Signal Ledger), `demand_metrics`,
`users`, `saved_items`.

> **Planned (mandatory before real users):** managed Postgres + pgvector —
> collapses the two stores, survives redeploys, handles concurrent writers.
> Alembic migrations come with it. See plan §5.1.

## Multi-tenancy

Direction (2026-07-27): self-serve platform — sign in, profile, own your work.

- **Auth** — `services/auth.py`: pbkdf2-sha256 password hashing (stdlib), JWT
  sessions (`JWT_SECRET`, ephemeral fallback warns at boot).
  `api/routes/auth.py`: `/api/auth/register|login|me`.
- **Ownership** — `brand_blueprints.user_id` set when a Bearer token is present;
  list endpoints return own + legacy anonymous rows. Saved research lives in
  `saved_items` via `/api/projects/saved` (CRUD, auth required).
- **Dashboard** — sign-in UI in the header; when signed in, saved
  trends/moves sync to the API (existing anonymous localStorage saves are
  migrated up on first login); anonymous use keeps working unchanged.

## Signal Ledger

`data_models/signal_ledger.py`, `services/signal_ledger.py`,
`api/routes/signal_ledger.py`. Every demand prediction is captured
timestamped and SHA-256-sealed **before the outcome is known**; forecasts must
be measurable (metric, range, horizon, confidence, falsifier); evidence
accumulates through staged, append-only events (awareness → … → operating
revenue); outcomes score against the sealed forecast. KPIs: hit rate, mean
forecast error, calibration gap. This implements the Future Strategy doc's
"demand becomes financeable" moat.

## API

FastAPI app in `api/main.py`; routers under `/api`: Social Pulse, Hotelier
Bets, Demand Scan, City Desires, Scheduler, Monitoring, Dashboard, Chat,
Brand Blueprint, Signal Ledger, Auth, Projects. Docs at `/docs`.

## Frontend

Two server-rendered pages in `api/routes/dashboard_simple.py`
(`/api/monitoring/dashboard-v2`, `/api/monitoring/build-a-brand`) styled with
the SENTIENT design system: CSS tokens (warm near-black surfaces, champagne
gold, Archivo/Inter/JetBrains Mono, the four-color gradient signature), with a
colorblind-validated accent palette. HTML-in-Python is acknowledged debt — the
React migration is ROADMAP Phase 10.

## Production engineering

- `config/settings.py` — typed settings validated at boot; fail-fast with
  clear messages; startup logs configured integrations, never secrets.
- `monitoring/logging_config.py` — JSON structured logs (`LOG_FORMAT=text`
  for dev); optional Sentry via `SENTRY_DSN`.
- CI (`.github/workflows/ci.yml`) — ruff error gate, fast tests, boot check.
- Tests: `tests/ingestion` (HTTP resilience, registry, scraper contracts),
  `tests/api` (auth + ownership).

## Deployment

Currently local-only (`START_DEMO.bat` or uvicorn; conda env `brandclave`,
python 3.11 — real interpreter at `miniforge3/envs/brandclave`). The old
Render free-tier deploy is dead. Target: always-on instance sized for JAX,
scheduler as a separate worker process, managed Postgres.

## How to

- **Add a source**: entry in `configs/sources.yaml` (+ scraper class if
  `content`, or `MetricScraper` subclass if `metric`). Status `active` puts it
  in the rotation. Contract tests: `tests/ingestion/test_registry_and_sources.py`.
- **Run everything locally**: `python -m uvicorn api.main:app --port 8000`,
  dashboard at `/api/monitoring/dashboard-v2`.
- **Run tests like CI**: `python -m pytest tests/ingestion tests/api -q` and
  `ruff check . --select E9,F63,F7,F82`.
