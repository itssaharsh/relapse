"""The pre-registered analysis of the placement experiment (results/PREREGISTRATION_3.md), written before its runs started.

    python scripts/analyze_place.py [--fetch] [--dir DIR]
--fetch copies the CircuitKIT attributions and the run results from the private stage-2 repo into results/place/.
Writes summary.json next to them and prints Q1-Q3 at fp16, then the same comparisons in Q4_0 and NF4.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

from analyze_map import HELD_OUT, MODELS, ROOT, download, sign_test

SEEDS = (31, 32)
FORMATS = ("fp16", "Q4_0", "nf4")
MAX_COST = 0.03
STRONG = 0.05 / 3
NAME = re.compile(r".+_W(\d+-\d+)_lr([^_]+)_s\d+$")


def fetch(d: str) -> None:
    get = download
    for v in MODELS:
        os.makedirs(os.path.join(d, "circuitkit", v), exist_ok=True)
        for f in ("layer_importance.json", "tokenizer_check.json"):
            get(f"circuitkit/{v}/{f}", os.path.join(d, "circuitkit", v, f))
        for s in SEEDS:
            os.makedirs(os.path.join(d, f"{v}_seed{s}"), exist_ok=True)
            for f in ("results.json", "items.json"):
                get(f"place/{v}_seed{s}/{f}", os.path.join(d, f"{v}_seed{s}", f))


def pick_window(per_layer: dict) -> str:
    """The three consecutive layers within 1-12 with the largest summed absolute MLP attribution."""
    score = lambda a: sum(per_layer[str(l)]["mlp_abs"] for l in range(a, a + 3))
    a = max(range(1, 11), key=score)
    return f"{a}-{a + 2}"


def picks(d: str) -> dict[str, str]:
    return {v: pick_window(json.load(open(os.path.join(d, "circuitkit", v, "layer_importance.json")))["per_layer"]) for v in MODELS}


def best_safe(rows: dict, window: str, fmt: str) -> float | None:
    """Lowest WMDP-bio among the window's runs losing at most 3 MMLU points against the original in the same format."""
    base = rows[("base", fmt)]["mmlu"]
    ok = [r["wmdp-bio"] for (n, f), r in rows.items() if f == fmt and n != "base" and NAME.match(n).group(1) == window and base - r["mmlu"] <= MAX_COST]
    return min(ok) if ok else None


def compare(cells: dict, chosen, default: str, fmt: str, models: tuple[str, ...]) -> dict:
    wins, losses, ties, excluded, pairs = 0, 0, 0, 0, []
    for v in models:
        for s in SEEDS:
            a, b = best_safe(cells[(v, s)], chosen(v), fmt), best_safe(cells[(v, s)], default, fmt)
            if a is None or b is None:
                excluded += 1
                continue
            pairs.append({"model": v, "seed": s, "chosen": chosen(v), "best_chosen": a, "best_default": b})
            wins += a < b; losses += a > b; ties += a == b
    p = sign_test(wins, wins + losses)
    holds = p < 0.05 and wins > losses
    return {"wins": wins, "losses": losses, "ties": ties, "not_comparable": excluded, "p": p, "holds": holds, "strong": holds and p < STRONG, "pairs": pairs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "place"))
    a = ap.parse_args()
    if a.fetch:
        fetch(a.dir)
    pick = picks(a.dir)
    cells = {(v, s): {(r["model"], r["prec"]): r for r in json.load(open(os.path.join(a.dir, f"{v}_seed{s}", "results.json")))} for v in MODELS for s in SEEDS}
    tests = {"Q1 (7-9 vs 5-7)": lambda v: "7-9", "Q2 (3-5 vs 5-7)": lambda v: "3-5", "Q3 (CircuitKIT window vs 5-7)": lambda v: pick[v]}
    res = {"picks": pick, "results": {scope: {fmt: {q: compare(cells, fn, "5-7", fmt, models) for q, fn in tests.items()} for fmt in FORMATS} for scope, models in (("all_five", MODELS), ("held_out", HELD_OUT))}}
    json.dump(res, open(os.path.join(a.dir, "summary.json"), "w"), indent=1)
    print("CircuitKIT windows", pick)
    for scope, by_fmt in res["results"].items():
        for fmt, qs in by_fmt.items():
            print(f"\n== {scope} {fmt}" + (" (primary)" if fmt == "fp16" and scope == "all_five" else ""))
            for q, r in qs.items():
                print(q, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items() if k != "pairs"})


if __name__ == "__main__":
    main()
