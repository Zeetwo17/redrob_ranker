# Methodology

## Problem framing

The official metric is `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`. Half the
score comes from the top 10. So the single most important thing is **getting the top 10
right**: genuine senior AI/retrieval engineers, at product companies,
available, in India, with no honeypots. The system is tuned to be *precise at the
head* first and broadly correct across the top 100 second.

## Why a transparent hybrid, not an LLM or a pure embedding model

1. **Compute constraint.** The ranking step runs in a sandbox at
   ≤5 min, 16 GB, CPU-only, no network. An LLM-per-candidate approach can't fit, and
   the spec says so explicitly. A 100K × (parse + score) pipeline has to be lightweight.
2. **The traps punish black boxes.** Keyword stuffers and CV/speech specialists have
   embeddings *very close* to a genuine fit. Separating them requires structured reasoning
   (title vs. career, skill trust, contradiction checks) that an opaque similarity score
   won't give you. Transparency is what wins here.
3. **Stages 4–5 reward explainability.** Reasoning quality is scored and the work is
   defended in interview. A model where every candidate's score decomposes into named,
   inspectable components is one you can actually stand behind.

## Component-by-component rationale

### Role fit (weight 0.34)
Title alone is gameable (a stuffer just edits their skills, not their job history). So
role fit blends the title bucket with the **fraction of career tenure spent in real
ML/AI roles**. This is the main anti-stuffer mechanism: `Graphic Designer @ TCS`
with a perfect AI skill list has *no* ML career history, so role fit stays near the floor.
On the flip side, a `Backend Engineer` who actually built recsys at a product company gets
lifted. That's the JD's "Tier-5 who doesn't use the buzzwords."

### Skill trust (weight 0.30)
Each skill's contribution = `relevance × proficiency × duration_factor × endorsement_factor`.
- **Relevance**: retrieval/ranking/IR = 1.0; LLM tooling = 0.7; NLP = 0.8; ML
  foundations = 0.5; data-eng = 0.3; **CV/speech = 0.0** (the JD doesn't want it).
- **Duration factor** saturates at ~2 yrs and is **0 at 0 months**, which also nulls the
  honeypot "expert skill, 0 months" pattern automatically.
- A soft-saturating sum means ~3 strong retrieval skills ≈ near-max. Piling on more
  buzzwords doesn't help (anti-stuffing by construction).
- Small bonus for **deep tooling** (pgvector, BM25, Learning-to-Rank, etc.) that stuffers
  rarely list. A CV/speech **dominance** check penalizes candidates whose ML identity is
  mostly vision/speech with weak retrieval signal.

### Semantic fit (weight 0.20)
TF-IDF (1–2 grams) cosine between the candidate's text (summary + role descriptions +
skills) and a hand-distilled **intent** query (`ranker/jd.py`), not the raw JD, which
is mostly culture prose. This catches the plain-language fit who writes
"designed a hybrid BM25 + dense-vector retrieval system" without listing "RAG". It
complements the structured signals but never overrides them.

### Experience / company / location gates (multiplicative)
- **Experience**: smooth bump around 6–8 yrs, gentle falloff, floor 0.45. Never a hard
  cutoff (JD says "range, not a requirement").
- **Company**: product/AI > FAANG > unicorn > neutral > services. An **all-services
  career** gets a 0.60 penalty (JD disqualifier), but current-services-with-product-past
  is untouched.
- **Location**: Pune/Noida = 1.0, Indian metros ≈ 0.9, non-India ≈ 0.35–0.45 (no visa
  sponsorship). Multiplicative so geography genuinely gates rather than nudges.

### Behavioral modifier (multiplicative, 0.60–1.15)
Implements the JD's instruction to down-weight the unreachable. Availability
(response rate, last-active recency, open-to-work, notice period) dominates; market
demand (recruiter saves, search appearances) is a lighter secondary. A perfect-on-paper
profile that's been silent for 6 months and answers 5% of recruiters drops ~40%.

### Honeypot guard (multiplicative, ×0.02)
General contradiction detection, not an ID blocklist (see `docs/DATA_AUDIT.md`). Forces
impossible profiles to the bottom while staying explainable and generalizable.

## Reasoning generation

Strings are composed **only** from the candidate's own breakdown facts: title, years,
named skills with their real durations, company, signal values. The output always pairs the lead
positive with any genuine concern (junior/senior, stale, low response, non-India,
notice, services-only, thin retrieval depth), and tone tracks the rank band. Built
to pass the Stage-4 checks: specific facts, JD connection, real concerns, no
hallucination, variation, rank-consistent tone.

## What we'd do next with more resources

- **Offline LLM-assisted silver labels.** With dev-time (not rank-time) LLM access, score
  a stratified sample of candidates for fit and use it to *tune and validate* weights
  against a less self-referential target than our rubric. Strictly offline; the ranking
  step stays network-free.
- **Sentence-transformer embeddings**, precomputed offline, loaded at rank time via the
  existing `semantic.py` seam. Stronger plain-language signal while still
  satisfying the no-network constraint.
- **Learning-to-rank head** if labeled recruiter-engagement data became available, with
  the current components as features (keeping explainability).

## Known limitations

- Proxy metrics are a consistency check, not a performance guarantee. There's no
  hidden ground truth to tune against, by design.
- ~15 honeypots are structurally undetectable from the given fields. Mitigated by the
  multiplicative gates, not eliminated.
- Hand-set weights encode our reading of the JD. That's a feature (full
  explainability, zero label leakage), but it's a deliberate trade against a
  learned ranker.
