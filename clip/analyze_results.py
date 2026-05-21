"""
Parse CLIP baseline results JSON and summarize hits by event type.

Usage:
    python clip/analyze_results.py \
        --results /ibex/scratch/alhumosm/SoccerNet/logs/clip_baseline_results.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Keywords to match against description text, in priority order.
# A description is assigned to the FIRST matching category.
EVENT_PATTERNS = [
    ("Substitution",  [r"\bsubstitut"]),
    ("Yellow card",   [r"\byellow card"]),
    ("Red card",      [r"\bred card"]),
    ("Goal",          [r"\bgoal\b", r"\bscores\b", r"\bnetted\b", r"\binto the net"]),
    ("Penalty",       [r"\bpenalty"]),
    ("Foul",          [r"\bfoul\b", r"\bfree.?kick"]),
    ("Offside",       [r"\boffside"]),
    ("Corner",        [r"\bcorner"]),
    ("Shot / save",   [r"\bshot\b", r"\bsave\b", r"\beffort\b", r"\bvolley"]),
    ("Cross",         [r"\bcross\b"]),
    ("VAR",           [r"\bvar\b"]),
    ("Injury",        [r"\binjur", r"\bmedical"]),
    ("Other",         [r""]),  # catch-all — always matches
]

def classify(description: str) -> str:
    text = description.lower()
    for label, patterns in EVENT_PATTERNS:
        if any(re.search(p, text) for p in patterns):
            return label
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to clip_baseline_results.json")
    parser.add_argument("--output", default=None, help="Optional path to save filtered hits JSON")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    queries = data["per_query"]
    total = len(queries)

    # --- collect hits ---
    hits_any = [q for q in queries if q["hit@1"] or q["hit@5"] or q["hit@10"]]

    print(f"\nResults file : {args.results}")
    print(f"Model        : {data.get('model', 'N/A')}")
    print(f"delta_t      : {data.get('delta_t', 'N/A')} s")
    print(f"window_s     : {data.get('window_s', 'N/A')} s")
    print(f"Total queries: {total:,}")
    print(f"Skipped      : {data.get('skipped_queries', 'N/A')}")
    print()

    # --- overall recall ---
    r = data["recall_at_k"]
    print("Overall Recall@K")
    print(f"  R@1  = {r['1']:.4f}%   ({sum(q['hit@1'] for q in queries):,} hits)")
    print(f"  R@5  = {r['5']:.4f}%   ({sum(q['hit@5'] for q in queries):,} hits)")
    print(f"  R@10 = {r['10']:.4f}%  ({sum(q['hit@10'] for q in queries):,} hits)")
    print()

    # --- per-event breakdown ---
    # For each event category count: total queries, hits@1, hits@5, hits@10
    stats = defaultdict(lambda: {"total": 0, "h1": 0, "h5": 0, "h10": 0})
    for q in queries:
        cat = classify(q["description"])
        stats[cat]["total"] += 1
        if q["hit@1"]:  stats[cat]["h1"] += 1
        if q["hit@5"]:  stats[cat]["h5"] += 1
        if q["hit@10"]: stats[cat]["h10"] += 1

    # preserve insertion order (priority order from EVENT_PATTERNS)
    ordered = [label for label, _ in EVENT_PATTERNS]

    col_w = 14
    print(f"{'Event':<{col_w}} {'Queries':>8}  {'H@1':>6} {'R@1':>7}  {'H@5':>6} {'R@5':>7}  {'H@10':>6} {'R@10':>7}")
    print("-" * 72)
    for cat in ordered:
        s = stats[cat]
        if s["total"] == 0:
            continue
        n = s["total"]
        def pct(x): return f"{100*x/n:.2f}%"
        print(
            f"{cat:<{col_w}} {n:>8}  "
            f"{s['h1']:>6} {pct(s['h1']):>7}  "
            f"{s['h5']:>6} {pct(s['h5']):>7}  "
            f"{s['h10']:>6} {pct(s['h10']):>7}"
        )
    print("-" * 72)
    # totals row
    h1_t  = sum(q["hit@1"]  for q in queries)
    h5_t  = sum(q["hit@5"]  for q in queries)
    h10_t = sum(q["hit@10"] for q in queries)
    def pct(x): return f"{100*x/total:.2f}%"
    print(
        f"{'TOTAL':<{col_w}} {total:>8}  "
        f"{h1_t:>6} {pct(h1_t):>7}  "
        f"{h5_t:>6} {pct(h5_t):>7}  "
        f"{h10_t:>6} {pct(h10_t):>7}"
    )

    # --- optionally save filtered hits ---
    if args.output:
        out = {k: v for k, v in data.items() if k != "per_query"}
        out["per_query"] = hits_any
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved {len(hits_any)} hit queries → {args.output}")


if __name__ == "__main__":
    main()
