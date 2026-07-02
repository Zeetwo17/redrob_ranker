#!/usr/bin/env python3
"""
rank.py  --  Redrob hackathon entry point.

Produces the top-100 submission CSV from candidates.jsonl, end to end, on CPU,
with no network access.

Usage
-----
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

    # gzipped input works too:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

The output CSV matches submission_spec.md section 2-3:
    candidate_id,rank,score,reasoning
exactly 100 data rows, ranks 1..100 unique, score non-increasing, ties broken by
candidate_id ascending.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from ranker.pipeline import rank_candidates
from ranker import config as cfg


def write_submission(results, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            w.writerow([r["candidate_id"], r["rank"],
                        f"{r['score']:.{cfg.SCORE_DECIMALS}f}", r["reasoning"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl or candidates.jsonl.gz")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--top-n", type=int, default=cfg.TOP_N)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    results, audit = rank_candidates(
        args.candidates, top_n=args.top_n, progress=not args.quiet)
    write_submission(results, args.out)
    dt = time.time() - t0

    if not args.quiet:
        print("-" * 60)
        print(f"wrote {len(results)} rows -> {args.out}")
        print(f"candidates scored      : {audit['n_candidates']:,}")
        print(f"honeypots detected     : {audit['honeypots_detected_total']}")
        print(f"honeypots in top {args.top_n}    : {audit['honeypots_in_top_n']} "
              f"(rate {audit['honeypot_rate_top_n']:.2%}, DQ threshold 10%)")
        print(f"wall-clock             : {dt:.1f}s "
              f"(budget 300s; {'OK' if dt < 300 else 'OVER BUDGET'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
