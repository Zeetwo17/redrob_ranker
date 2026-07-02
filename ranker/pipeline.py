"""
Orchestration: load candidates -> semantic scores -> score -> sort -> top-N
with reasoning. Single-pass streaming, CPU-only, TF-IDF is the only vectorised step.
"""
import gzip
import io
import json
from datetime import date

from .semantic import SemanticMatcher
from .jd import candidate_document
from .scoring import score_candidate
from .reasoning import build_reasoning
from . import config as cfg


def _open_any(path):
    """Open .jsonl or .jsonl.gz transparently as UTF-8 text."""
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_candidates(path):
    """Stream the file; return (records, documents) in aligned order.

    `records` are the raw candidate dicts (needed by the scorer); `documents`
    are the semantic text blobs in the same order.
    """
    records, documents = [], []
    with _open_any(path) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            records.append(c)
            documents.append(candidate_document(c))
    return records, documents


def rank_candidates(path, top_n=cfg.TOP_N, today=None, progress=False):
    """
    Full ranking. Returns a list of dicts (length top_n) with keys:
        candidate_id, rank, score, reasoning, breakdown
    plus an `audit` dict of run statistics.
    """
    today = today or date(2026, 6, 8)

    if progress:
        print("[1/4] loading candidates ...", flush=True)
    records, documents = load_candidates(path)
    n = len(records)

    if progress:
        print(f"      {n:,} candidates loaded", flush=True)
        print("[2/4] semantic TF-IDF pass ...", flush=True)
    matcher = SemanticMatcher()
    semantic_scores = matcher.fit_transform_scores(documents)

    if progress:
        print("[3/4] scoring ...", flush=True)
    scored = []
    honeypot_count = 0
    for c, sem in zip(records, semantic_scores):
        final, breakdown = score_candidate(c, float(sem), today)
        if breakdown["honeypot"]:
            honeypot_count += 1
        scored.append((final, c["candidate_id"], breakdown))

    # Normalise final scores to (0,1] by the population max for clean presentation,
    # preserving order. Then sort by (-score, candidate_id) so equal scores break by
    # candidate_id ascending (required by the spec / validator) and scores are
    # non-increasing by rank.
    max_final = max((s for s, _, _ in scored), default=1.0) or 1.0
    scored = [(s / max_final, cid, bd) for s, cid, bd in scored]
    scored.sort(key=lambda x: (-round(x[0], cfg.SCORE_DECIMALS), x[1]))

    if progress:
        print(f"[4/4] building top {top_n} + reasoning ...", flush=True)
    results = []
    top = scored[:top_n]
    hp_in_top = sum(1 for _, _, bd in top if bd["honeypot"])
    for i, (score, cid, bd) in enumerate(top, start=1):
        results.append({
            "candidate_id": cid,
            "rank": i,
            "score": round(score, cfg.SCORE_DECIMALS),
            "reasoning": build_reasoning(bd, i),
            "breakdown": bd,
        })

    audit = {
        "n_candidates": n,
        "honeypots_detected_total": honeypot_count,
        "honeypots_in_top_n": hp_in_top,
        "honeypot_rate_top_n": round(hp_in_top / max(top_n, 1), 4),
    }
    return results, audit
