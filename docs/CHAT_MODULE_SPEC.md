# BrandClave Chat Module — Technical Specification

**Module 1: Bayesian RAG + POMDP-lite Dialogue Control**

Uses Mistral API with uncertainty-aware retrieval and belief-state dialogue control.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Chat UI                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Orchestrator API                           │
│                     POST /chat                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     Mode Router                              │
│         (Bayesian intent inference → JSON)                   │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐           │
│  │ Insight  │  │ Brand Build  │  │ Demand Scan │           │
│  │  Mode    │  │    Mode      │  │    Mode     │           │
│  └──────────┘  └──────────────┘  └─────────────┘           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      RAG Layer                               │
│  ┌─────────────┐  ┌───────────┐  ┌──────────────────┐      │
│  │  Retriever  │  │ Re-ranker │  │ Bayesian Fusion  │      │
│  │ vector+BM25 │  │ (optional)│  │ posterior+entropy│      │
│  └─────────────┘  └───────────┘  └──────────────────┘      │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Belief Manager (POMDP-lite)                 │
│  Tracks: intent, stage, missing slots, confidence            │
│  Actions: ASK_Q | RETRIEVE_MORE | ANSWER | SUGGEST_BUILD    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Project Store                             │
│  Saves: messages, artifacts (JSON), sources, confidence      │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Model (Supabase/PostgreSQL)

### Table: `projects`
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| owner_id | uuid | User reference |
| title | text | Project name |
| location | text | City/region |
| segment | text | lifestyle, luxury, boutique, etc. |
| created_at | timestamp | |

### Table: `messages`
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| project_id | uuid | FK to projects |
| role | text | user / assistant |
| content | text | Raw message text |
| mode | text | insight / brand_build / demand_scan |
| created_at | timestamp | |

### Table: `artifacts`
Stores structured outputs (the "perceived intelligence" layer).

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| project_id | uuid | FK to projects |
| artifact_type | text | insight_brief, trend_card_set, demand_scan, brand_blueprint_lite |
| json | jsonb | Structured payload |
| sources | jsonb | Array: doc ids, urls, snippet hashes |
| confidence | float | 0–1 |
| created_at | timestamp | |

### Table: `knowledge_chunks`
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| source_type | text | internal_doc, web, user_upload, playbook |
| source_id | text | Reference to original |
| title | text | |
| chunk_text | text | |
| embedding | vector | 1024-dim Mistral embeddings |
| metadata | jsonb | location tags, segment tags, date, etc. |

---

## Mode Router (Bayesian Intent Inference)

Instead of hard if/else classification, compute intent probabilities:

```json
{
  "p_insight": 0.62,
  "p_brand_build": 0.21,
  "p_demand_scan": 0.17,
  "confidence": 0.73,
  "slots_needed": ["location", "segment", "adr"]
}
```

**Implementation:**
1. Ask Mistral for strict JSON router (fast, cheap model)
2. Add simple priors (e.g., if URL present → higher demand_scan prior)
3. Decision logic:
   - If `confidence < τ` → ask clarifying question
   - Else choose `argmax(mode)`

---

## Bayesian RAG

### Step 1: Retrieve Candidates
- Dense retrieval (embeddings)
- Sparse retrieval (keyword/BM25)
- Union top K

### Step 2: Bayesian Fusion

For each chunk `d`, compute features:
- `s_v(d)`: vector similarity (0–1 scaled)
- `s_k(d)`: keyword score (0–1 scaled)
- `s_m(d)`: metadata match (location/segment tags) (0–1)

Likelihood model:
```
logit P(R_d=1 | s_v, s_k, s_m) = α + β_v·s_v + β_k·s_k + β_m·s_m
```

Where:
- `R_d` = "chunk is relevant"
- `α` = prior preference for internal docs vs web

**Output:**
- Keep chunks with `P(R_d=1) > 0.55`
- Compute entropy to decide follow-ups

### Step 3: Generate with Citations + Uncertainty
Pass high-posterior chunks to Mistral with instruction:
> "If posterior uncertainty > threshold, ask 1–2 clarifying questions before final answer."

---

## POMDP-lite Dialogue Manager

### Belief State `b`

```json
{
  "mode": {"insight": 0.58, "brand_build": 0.32, "demand_scan": 0.10},
  "slots": {"location": null, "segment": "lifestyle", "adr": null, "url": null},
  "retrieval": {"top_posterior": 0.61, "entropy": 0.72},
  "stage": {"explore": 0.64, "commit": 0.36}
}
```

### Actions
| Action | When to use |
|--------|-------------|
| `ASK_CLARIFYING_Q` | Missing critical slots |
| `RETRIEVE_MORE` | High retrieval entropy |
| `ANSWER_NOW` | High confidence |
| `SUGGEST_SEND_TO_BUILD_A_BRAND` | After insight complete |
| `SAVE_ARTIFACT` | After generating structured output |

### Policy (Rule-based to start)
1. If missing critical slots → ask 1 question (not 5)
2. If retrieval entropy high → retrieve more or ask 1 disambiguation
3. If confidence high → answer + save artifact

---

## Structured Output Schemas

### Insight Mode: `insight_brief_v1`

```json
{
  "type": "insight_brief_v1",
  "topic": "",
  "location": "",
  "key_signals": [
    {
      "signal": "",
      "why_it_matters": "",
      "confidence": 0.0,
      "evidence": ["chunk_id:..."]
    }
  ],
  "white_space_opportunities": [
    {
      "opportunity": "",
      "who_it_serves": "",
      "why_now": "",
      "risk": "",
      "confidence": 0.0
    }
  ],
  "recommended_next_step": {
    "action": "send_to_build_a_brand|ask_more|save",
    "reason": ""
  }
}
```

### Demand Scan: `demand_scan_lite_v1`

```json
{
  "type": "demand_scan_lite_v1",
  "property_url": "",
  "location": "",
  "segment": "",
  "target_adr": null,
  "demand_fit_score": 0,
  "positioning_misalignment_flags": ["", ""],
  "experience_gap_snapshot": [
    {
      "theme": "",
      "what_guests_want": "",
      "what_is_missing": ""
    }
  ],
  "opportunity_lanes": [
    {
      "trajectory": "",
      "what_to_build": "",
      "why_it_wins": ""
    }
  ],
  "recommended_next_step": {
    "action": "send_to_build_a_brand",
    "prefill": {"location": "", "segment": "", "adr": null}
  }
}
```

### Brand Build: `brand_blueprint_lite_v1`

```json
{
  "type": "brand_blueprint_lite_v1",
  "inputs": {
    "location": "",
    "segment": "",
    "target_adr": null,
    "developer_goal": "",
    "attached_signals": []
  },
  "brand_name": "",
  "one_liner": "",
  "thesis": "",
  "pillars": ["", "", ""],
  "positioning_statement": "",
  "signature_experiences": [
    {"name": "", "description": "", "why_it_matters": ""}
  ],
  "guest_journey": {
    "arrival": "",
    "stay": "",
    "departure": ""
  },
  "design_direction": "",
  "revenue_logic": "",
  "guest_personas": [
    {"name": "", "description": "", "spend_behavior": ""}
  ],
  "unmet_desires_solved": [""],
  "fnb_concepts": [
    {"name": "", "concept": "", "vibe": ""}
  ],
  "investor_summary": ""
}
```

---

## Implementation Timeline

### Week 1: Ship v0
- [ ] Chat UI (single screen)
- [ ] Router JSON output
- [ ] Basic RAG with vector search
- [ ] Structured artifacts for Insight mode only
- [ ] Save-to-project

### Week 2: Add Intelligence
- [ ] Bayesian fusion scoring + confidence/entropy
- [ ] Demand Scan Lite output schema
- [ ] "Send to Build a Brand" prefill action

### Week 3: Complete MVP
- [ ] Brand Build Lite schema
- [ ] Basic reranker (optional)
- [ ] Evaluation: thumbs up/down → weight tuning

---

## Confidence Panel (UX Feature)

Display visible confidence indicator:
- **High / Medium / Low** confidence badge
- "Based on: X sources, Y signals, recentness"

This is the "terminal-grade intelligence" UX without expensive data.

---

## Integration with Existing Architecture

This module integrates with the existing PyMDP active inference system:
- `ScrapingPOMDP` feeds data into `knowledge_chunks`
- `ClusteringPOMDP` can inform retrieval metadata
- `MoveExtractionPOMDP` outputs become searchable artifacts
- New `DialoguePOMDP` manages conversation flow

---

*Last updated: January 2026*
