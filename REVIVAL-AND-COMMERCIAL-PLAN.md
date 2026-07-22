# BrandClave, Revival and Commercial-Grade Plan

Internal engineering doc. Written 2026-07-21 after inspecting the repo. Two separate jobs here: (1) revive the thing so it runs and is filmable, days of work, and (2) take it to commercial-grade so it can be sold, weeks of work plus real monthly cost. Keep them separate so "make it commercial" doesn't quietly swallow the cheap demo.

---

## 1. Verified current state (2026-07-21)

- Real codebase, roughly 60% built per ROADMAP.md.
- Last substantive commit `ea40204`, **14 Jan 2026**. Six months cold.
- Stack: FastAPI + SQLAlchemy + SQLite, ChromaDB for vectors, Mistral for embeddings and LLM, HDBSCAN clustering, PyMDP active-inference layer, APScheduler background scraper, 12 sources.
- Deploy: Render free tier, SQLite committed to git. `brandclave-demo.onrender.com` returns 404 on both `/health` and `/api/monitoring/health`. Deploy is dead or renamed.
- DB is a ~6MB SQLite file living inside OneDrive. Querying it in place threw `disk I/O error`, which is the OneDrive sync lock, not corruption. Copying it out reads fine.

### Known landmines before it will even install/run

1. **pymdp version conflict.** `requirements.txt` pins `inferactively-pymdp @ git+...@v1.0.0_alpha`. `environment.yml` pins `inferactively-pymdp>=0.0.7.1`. These are different major lines. Pick one before building the env or install fails.
2. **OneDrive + SQLite.** The working DB must move off the OneDrive path or the I/O lock keeps biting during dev. (General rule for this machine: push-location off OneDrive before conda work too.)
3. **Mistral SDK drift.** `mistralai>=1.0.0` pinned. The 1.x SDK changed the client surface from 0.x. Six months on, worth confirming the client calls in `services/` still match the installed version.
4. **JAX on 512MB.** `render.yaml` runs `PREWARM_SERVICES=false` and notes the free tier can't hold JAX in memory. The active-inference layer is effectively throttled off in prod today.
5. **Library drift generally.** chromadb, fastapi, sentence-transformers all move fast. Expect 2-3 breakages from six months of drift. Normal, just budget for it.

---

## 2. Revival (get it filmable)

Goal: it runs locally, data is fresh, dashboard works, redeployed somewhere that isn't a dead free box. This is the ~1-2 days quoted to Sarah.

### Day 1-2, running locally

- Reconcile the pymdp pin, then build a fresh conda env from a single clean dependency file. Push-location off OneDrive first.
- Move the working DB to a local (non-OneDrive) path. Stops the I/O lock.
- Restore `.env` keys. Confirm Mistral authenticates and the client calls still match the 1.x SDK.
- Boot the API, load the dashboard, run one scrape cycle, confirm end-to-end data flow.
- Fix the 2-3 drift breakages as they surface.

### Day 2, prove the pipeline

- Fresh scrape across all 12 sources (corpus is six months stale).
- Regenerate trends and moves, eyeball quality.
- Confirm City Desires and Build a Brand still produce good output.

### Then redeploy

Off the dead free tier. See hosting note in section 4. For a pure demo a small paid instance is enough. For anything a customer touches, more.

Outcome: a filmable demo. Low risk, mostly mechanical.

---

## 3. Revival vs commercial-grade

Different projects. Be honest about the gap.

- **Revival** = a demo that runs and looks good on camera. Days.
- **Commercial-grade** = something people pay for that doesn't fall over. Weeks, plus real monthly infra and API cost.

The demo can lie a little (single box, in-memory state, no accounts). A product cannot.

---

## 4. Commercial-grade path

Priority order. 1-4 are the must-haves before charging anyone. 5-7 follow.

### 1. Database, move off SQLite-in-git
Highest-leverage fix, already flagged as tech debt in ROADMAP.md.
- SQLite concurrent writes corrupt under load.
- Render disk is ephemeral, so data vanishes on redeploy.
- Committing a binary DB bloats the repo every scrape.
- Target: managed Postgres + pgvector (Neon or Supabase). Migrate the ChromaDB vectors into pgvector so there's one store, not two.

### 2. Hosting that can run the thing
- Free tier is 512MB, 30s cold starts, sleeps when idle. A paying user waiting 40s for a cold box is a lost user.
- Needs a real always-on instance sized for JAX (so the active-inference layer actually runs).
- The APScheduler scraper must run as a separate worker process, not inside the web process. Today a restart or scale event disrupts scraping.

### 3. Accounts and multi-tenancy
Biggest product gap, not just infra. Phase 4 in ROADMAP, unbuilt.
- No auth exists. Can't sell access to something with no login.
- Need per-user data separation, saved projects tied to a user, not localStorage.

### 4. Payments
- If it's a revenue tool, someone pays. No billing today.
- Stripe + a subscription or usage model. Decide the model first (per-scan credits vs monthly seat).

### 5. Data licensing, for real
- Demo runs on compliant public sources. Fine.
- A paid product using scraped OTA/review content turns the terms-of-service exposure (flagged in the SENTIENT build plan, section 4) from hypothetical into live legal risk. Licensed aggregators before monetizing on that data.

### 6. Reliability basics
- Error handling, retries, backoff on LLM calls, rate-limit protection.
- Monitoring and alerting (some `/monitoring` endpoints exist, no external alerting).
- Tests exist but there's no CI. Add a pipeline.

### 7. React frontend
- Dashboard is HTML-in-Python today. Fine to film, not fine to sell. Phase 10.

---

## 5. Rough shape of cost and effort

| Track | Effort | Recurring cost |
| :-- | :-- | :-- |
| Revival + demo | 1-2 days | small paid host, ~low $/mo |
| Commercial must-haves (DB, host, auth, payments) | several weeks | managed Postgres, real host, Stripe fees, LLM API spend that scales with users |
| Full product (5-7 on top) | more weeks | data licensing is the big one, see build plan 2.3 |

The commercial build is exactly what the budget in Sarah's cost framework is for. Don't let it balloon the free demo. Demo now, commercial as a funded, scoped phase.

---

## 6. Open decisions

- Run the revival now (reconcile deps, spin env, boot, scrape) to see what actually comes back to life, vs plan first.
- Which signal-to-city city set to pre-scan for the "both directions" build Sarah wants (separate scope, see the shared Google Doc).
- Billing model for the eventual product, per-scan credits vs monthly seat. Changes the payments build.
