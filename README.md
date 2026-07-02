# Redrob Candidate Ranker

A transparent, CPU-only ranking system built for the Redrob *Intelligent Candidate
Discovery & Ranking* challenge. Ranks 100K candidates against the released
"Senior AI Engineer — Founding Team" job description, spits out the top 100.

**Not a keyword matcher.** It looks at each profile the way a decent recruiter
would: title *and career trajectory*, actual skill depth, what the free text
says, whether the person is even reachable. Every ranking decision can be
explained in one sentence.

```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runs end-to-end in **~97 s for 100K candidates** on a 16 GB CPU-only machine with
**no network**. Well inside the competition budget (≤5 min, ≤16 GB, CPU-only, no API
calls).

---

## Why this design

The dataset is adversarial on purpose. A full audit of the pool
(`docs/DATA_AUDIT.md`) turned up a bunch of traps that will wreck naive approaches:

| Trap | What it looks like | What beats it |
|---|---|---|
| **Keyword stuffers** | `Graphic Designer @ TCS` with RAG, Pinecone, FAISS, LangChain in skills | Title + **career trajectory** — their work history isn't ML |
| **Plain-language Tier-5s** | A genuine fit whose summary says "built hybrid BM25 + dense retrieval" but never lists the buzzword skills | **Semantic** match on free text |
| **CV/speech specialists** | Loaded with YOLO, OpenCV, Diffusion — the JD explicitly doesn't want CV/speech-only | CV/speech skills carry **zero** weight; dominance is penalized |
| **Honeypots (~80)** | Impossible profiles: "expert" skill used 0 months; 8 yrs at a 3-yr company | Structural **contradiction guard** → hard penalty |
| **Behavioral twins** | Two identical-on-paper profiles, one inactive with 5 % response rate | **Availability modifier** down-weights the unreachable one |

A pure embedding or keyword model walks right into all of these. So the ranker
leans on the structured signals the JD actually cares about, and uses semantics
as a complement, not the core.

## The scoring model

Every component is a number in `[0,1]` (or a multiplier) that we picked and can
defend. Nothing is trained on hidden labels.

```
core   = 0.34·role + 0.30·skill + 0.20·semantic + 0.10·company + 0.06·education
fit    = core × experience_mult × location_mult × consulting_penalty
pre    = fit × behavioral_modifier            # availability + market demand
final  = pre × honeypot_multiplier            # hard guard on impossible profiles
```

| Component | What it captures | Key idea |
|---|---|---|
| **role** | current title + fraction of career in real ML/AI roles | a noise title with no ML history stays at the floor; a generic title with a genuine ML career rises |
| **skill** | retrieval/ranking skills, each weighted by `proficiency × duration × endorsements` | trust-weighting kills the stuffer whose "expert" skill has 0 months / 0 endorsements; deep tools (pgvector, BM25, Learning-to-Rank) add an authenticity bonus |
| **semantic** | TF-IDF cosine between a distilled JD intent and the candidate's own text | finds people who *describe* building search/ranking even without the keywords |
| **experience** | closeness to the 5–9 (ideally 6–8) yr band | smooth, never a hard cutoff — the JD says "range, not a requirement" |
| **company** | product/AI vs. services, current + best-in-career | career-long services (TCS/Infosys/…) is penalized; current-services-with-product-past is fine |
| **location** | Pune/Noida → India metros → India other → non-India | non-India strongly suppressed (JD: no visa sponsorship), not zeroed |
| **behavioral** | response rate, recency, open-to-work, notice period, recruiter saves | the JD says it outright: a perfect-on-paper but unreachable candidate "is, for hiring purposes, not actually available" |
| **honeypot** | structural impossibilities | `×0.02` — forced to the bottom |

See `ranker/config.py` for every weight and threshold in one place, and
`docs/METHODOLOGY.md` for the full rationale.

## Results (this run)

```
candidates scored       : 100,000
wall-clock              : 96.8 s   (budget 300 s)
honeypots detected      : 65
honeypots in top 100    : 0        (DQ threshold > 10)
```

Offline audit (`python evaluate.py …`, independent silver-standard rubric):

```
honeypots in top 100       : 0
keyword-stuffers in top 100: 0          (top 10: 0)
India-based in top 100     : 100
open_to_work in top 100    : 93
top-100 title mix          : 100 % genuine AI/ML/retrieval roles
proxy NDCG@10 / @50 / MAP / P@10 : 1.00 / 0.95 / 0.92 / 1.00
```

> Quick note on the graded metrics: these are a **consistency signal, not a score prediction**.
> The silver rubric and the ranker share the same worldview, so a high proxy NDCG
> mostly just confirms the ranker is internally consistent. The numbers you should
> actually trust are the audit counts (zero honeypots, zero stuffers, geography/availability mix).

## Reproduce

```bash
pip install -r requirements.txt

# rank (gzipped input also works: --candidates ./candidates.jsonl.gz)
python rank.py --candidates ./candidates.jsonl --out ./outputs/submission.csv

# validate against the official format rules
python validate_submission.py ./outputs/submission.csv      # (organizer-provided)

# offline audit + proxy metrics
python evaluate.py --candidates ./candidates.jsonl --submission ./outputs/submission.csv
```

No network, no GPU, no pre-computed artifacts needed for the ranking step.

## Repository layout

```
redrob-ranker/
├── rank.py                 # entry point → produces submission.csv
├── evaluate.py             # offline proxy evaluation + audit harness
├── requirements.txt
├── submission_metadata.yaml
├── ranker/
│   ├── config.py           # all weights & thresholds
│   ├── taxonomies.py       # titles / skills / companies / geo priors (domain knowledge)
│   ├── jd.py               # distilled JD intent + candidate-document builder
│   ├── features.py         # role / skill / experience / company / behavioral scores
│   ├── semantic.py         # TF-IDF semantic matcher (swappable for embeddings)
│   ├── honeypot.py         # structural contradiction guard
│   ├── scoring.py          # combines everything → final score
│   └── reasoning.py        # fact-grounded, varied reasoning strings
├── outputs/
│   └── submission.csv      # the top-100 ranking
└── docs/
    ├── DATA_AUDIT.md       # what the 100K-pool audit found
    └── METHODOLOGY.md      # design rationale, defense notes, limitations
```

## Design choices worth calling out

- **TF-IDF, not neural embeddings, by default.** The ranking step has to run with
  no network in ≤5 min. TF-IDF needs no model download and is reproducible in any
  sandbox. `ranker/semantic.py` is a drop-in seam though: swap in a precomputed
  sentence-transformer matrix and nothing else changes. Kept the
  dependency-light path as default for reproducibility.
- **Structured-first, semantics-second.** The JD is pretty explicit that the right
  answer is about title-vs-skills and reading between the lines, not keyword
  density. So role + skill-trust carry the most weight and semantics fills in
  the gaps.
- **No special-casing.** Honeypots and stuffers are handled by general signals
  (contradiction detection, trust-weighting, trajectory), not hard-coded ID
  lists. The approach should generalize to an unseen pool.

## Limitations

- The honeypot guard catches the **structurally** impossible profiles (65 of the
  ~80). A handful rely on facts we can't verify from the data alone (e.g. company
  founding year) and look indistinguishable from strong candidates. The
  multiplicative behavioral/role gates keep the top-100 honeypot rate at 0
  regardless, well under the 10 % DQ line.
- Weights are hand-tuned against the JD and the offline proxy, not learned. That's
  by design (no hidden labels, full explainability). A learning-to-rank head could
  be bolted on if labeled engagement data became available.
