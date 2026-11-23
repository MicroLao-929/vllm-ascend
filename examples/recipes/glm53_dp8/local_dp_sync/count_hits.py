"""Check correlated per-rank local transport decisions for completed ordinary MK tokens."""

import argparse
from collections import Counter, defaultdict
import json
import re

LINE = re.compile(
    r"GLM53 local DP step rank=(\d+) sequence=(\d+) owner=(\d+) request=(\S+) position=(\d+) outcome=(\w+)"
)


def count(stream):
    steps = defaultdict(dict)
    for line in stream:
        match = LINE.search(line)
        if match is None:
            continue
        rank, sequence, owner, request, position, outcome = match.groups()
        key = (int(sequence), int(owner), request, int(position))
        rank = int(rank)
        if rank in steps[key]:
            raise ValueError(f"duplicate rank {rank} for {key}")
        steps[key][rank] = outcome
    coverage = Counter()
    per_rank = Counter()
    mismatches = []
    for key, results in steps.items():
        if set(results) != set(range(8)):
            mismatches.append([list(key), sorted(results)])
            continue
        local = [v == "local" for v in results.values()]
        if any(local) and not all(local):
            mismatches.append([list(key), results])
            continue
        coverage["completed_mk_steps"] += 1
        coverage["all_8_local" if all(local) else "all_8_gloo"] += 1
        if all(v == "gloo_candidate" for v in results.values()):
            coverage["all_8_observed_candidate_rows"] += 1
        for rank, outcome in results.items():
            per_rank[f"{rank}/{outcome}"] += 1
    return {"steps": dict(coverage), "per_rank": dict(sorted(per_rank.items())), "incomplete_or_disagree": mismatches}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--require-local", action="store_true")
    args = parser.parse_args()
    with open(args.log, encoding="utf-8", errors="replace") as stream:
        result = count(stream)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if (
        not result["steps"]
        or result["incomplete_or_disagree"]
        or (args.require_local and not result["steps"].get("all_8_local"))
    ):
        raise SystemExit(1)
