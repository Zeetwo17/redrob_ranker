#!/usr/bin/env python3
"""
evaluate.py  --  offline proxy evaluation harness.

There is no live leaderboard and the ground truth is hidden, so we built an
*independent* silver-standard relevance model to (a) sanity-check the ranker and
(b) tune weights without overfitting to a number we can't see.

IMPORTANT framing: the silver labels below are a transparent rubric, not the real
ground truth. A high proxy NDCG is weak evidence on its own (the rubric and the
ranker both encode our reading of the JD). The trustworthy outputs of this tool
are the **audit checks**:
    * honeypot rate in the top 100 (must be ~0; DQ threshold is 10%)
    * keyword-stuffer rate in the top 10/100 (noise-title + bolted-on AI skills)
    * title / geography / availability mix of the top 100
The graded metrics (NDCG@10, NDCG@50, MAP, P@10) mirror the official composite and
are reported as a consistency signal.

Usage
-----
    python evaluate.py --candidates ./candidates.jsonl --submission ./submission.csv
"""
import argparse
import csv
import json
import math
from collections import Counter

from ranker.pipeline import _open_any
from ranker import taxonomies as tax
from ranker.honeypot import is_honeypot
from ranker.features import role_fit, skill_score


# ---------------------------------------------------------------------------
# Silver-standard relevance rubric (0..4). Independent, stepwise — NOT the ranker
# formula. Mirrors how the spec describes the ground truth.
# ---------------------------------------------------------------------------
def silver_relevance(c):
    if is_honeypot(c):
        return 0  # forced tier 0, per spec

    p = c.get("profile", {})
    title = p.get("current_title", "")
    yoe = p.get("years_of_experience", 0) or 0
    country = (p.get("country", "") or "").lower()
    in_india = country == "india"

    role, rfacts = role_fit(c)
    skill, sfacts = skill_score(c)
    ml_career = rfacts.get("ml_roles_in_history", 0) >= 1 or title in tax.BULLSEYE_TITLES
    strong_retrieval = skill >= 0.6 and len(sfacts.get("top_retrieval_skills", [])) >= 2

    s = c.get("redrob_signals", {})
    available = (s.get("recruiter_response_rate", 0) or 0) >= 0.3 or bool(s.get("open_to_work_flag"))

    # Keyword-stuffer: noise/off-target title, AI skills present, but no ML career.
    is_stuffer = (title in tax.NOISE_TITLES) and skill >= 0.4 and not ml_career

    if is_stuffer:
        return 0
    # Genuine, well-rounded fit
    bullseye = title in tax.BULLSEYE_TITLES or title in tax.STRONG_TITLES
    product = tax.company_score(p.get("current_company", ""), p.get("current_industry", "")) >= 0.85

    if bullseye and strong_retrieval and in_india and available and 4 <= yoe <= 11:
        return 4
    if (bullseye or (ml_career and strong_retrieval)) and (in_india or available):
        return 3
    if ml_career and skill >= 0.4:
        return 2
    if role >= 0.4 or skill >= 0.3:
        return 1
    return 0


def is_stuffer(c):
    p = c.get("profile", {})
    title = p.get("current_title", "")
    skill, _ = skill_score(c)
    _, rf = role_fit(c)
    ml_career = rf.get("ml_roles_in_history", 0) >= 1 or title in tax.BULLSEYE_TITLES
    return (title in tax.NOISE_TITLES) and skill >= 0.4 and not ml_career


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels, ideal_rels, k):
    d = dcg(ranked_rels[:k])
    idcg = dcg(sorted(ideal_rels, reverse=True)[:k])
    return d / idcg if idcg > 0 else 0.0


def average_precision(ranked_rels, total_relevant, rel_threshold=3):
    hits, ap = 0, 0.0
    for i, r in enumerate(ranked_rels, start=1):
        if r >= rel_threshold:
            hits += 1
            ap += hits / i
    denom = min(total_relevant, len(ranked_rels)) if total_relevant else 0
    return ap / denom if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    args = ap.parse_args()

    print("streaming pool + computing silver relevance ...", flush=True)
    rel_by_id, cand_by_id = {}, {}
    all_rels = []
    total_relevant = 0
    pool_titles = Counter()
    with _open_any(args.candidates) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            r = silver_relevance(c)
            cid = c["candidate_id"]
            rel_by_id[cid] = r
            cand_by_id[cid] = c
            all_rels.append(r)
            if r >= 3:
                total_relevant += 1

    with open(args.submission, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ranked_ids = [row["candidate_id"] for row in rows]
    ranked_rels = [rel_by_id.get(cid, 0) for cid in ranked_ids]

    ndcg10 = ndcg_at_k(ranked_rels, all_rels, 10)
    ndcg50 = ndcg_at_k(ranked_rels, all_rels, 50)
    mapv = average_precision(ranked_rels, total_relevant, 3)
    p10 = sum(1 for r in ranked_rels[:10] if r >= 3) / 10.0
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * mapv + 0.05 * p10

    # Audit
    hp = sum(1 for cid in ranked_ids if is_honeypot(cand_by_id[cid]))
    stuff = sum(1 for cid in ranked_ids if is_stuffer(cand_by_id[cid]))
    stuff10 = sum(1 for cid in ranked_ids[:10] if is_stuffer(cand_by_id[cid]))
    titles = Counter(cand_by_id[cid]["profile"]["current_title"] for cid in ranked_ids)
    india = sum(1 for cid in ranked_ids
                if (cand_by_id[cid]["profile"].get("country", "").lower() == "india"))
    avail = sum(1 for cid in ranked_ids
                if cand_by_id[cid]["redrob_signals"].get("open_to_work_flag"))
    tier_mix = Counter(ranked_rels)

    print("\n" + "=" * 64)
    print("PROXY GRADED METRICS (silver labels — consistency signal, not truth)")
    print("=" * 64)
    print(f"  NDCG@10 : {ndcg10:.4f}   (official weight 0.50)")
    print(f"  NDCG@50 : {ndcg50:.4f}   (official weight 0.30)")
    print(f"  MAP     : {mapv:.4f}   (official weight 0.15)")
    print(f"  P@10    : {p10:.4f}   (official weight 0.05)")
    print(f"  COMPOSITE (proxy): {composite:.4f}")
    print(f"  total pool relevant (tier>=3): {total_relevant:,}")

    print("\n" + "=" * 64)
    print("AUDIT CHECKS (these are the trustworthy outputs)")
    print("=" * 64)
    print(f"  honeypots in top 100      : {hp}   (DQ threshold > 10)")
    print(f"  keyword-stuffers in top 100: {stuff}")
    print(f"  keyword-stuffers in top 10 : {stuff10}")
    print(f"  India-based in top 100     : {india}")
    print(f"  open_to_work in top 100    : {avail}")
    print(f"  top-100 silver tier mix    : "
          + ", ".join(f"T{t}:{tier_mix.get(t,0)}" for t in (4, 3, 2, 1, 0)))
    print("\n  top-100 title distribution:")
    for t, ct in titles.most_common(15):
        print(f"      {ct:3d}  {t}")


if __name__ == "__main__":
    main()
