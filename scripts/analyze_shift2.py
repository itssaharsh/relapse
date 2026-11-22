"""The pre-registered analysis of placement against the tokenizer (results/PREREGISTRATION_5.md), written before its runs started.

    python scripts/analyze_shift2.py [--fetch]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import analyze_shift
from analyze_map import MODELS, ROOT, download, sign_test

SEEDS = (51, 52)
OUT = os.path.join(ROOT, "results", "shift2")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true"); a = ap.parse_args()
    if a.fetch:
        for v in MODELS:
            for s in SEEDS:
                d = os.path.join(OUT, f"{v}_seed{s}"); os.makedirs(d, exist_ok=True)
                for f in ("results.json", "items.json"):
                    download(f"shift2/{v}_seed{s}/{f}", os.path.join(d, f))
    analyze_shift.SEEDS = SEEDS
    rows = analyze_shift.load(OUT)
    eff = [r for r in rows if r["effective"]]
    by = {(r["model"], r["seed"], r["window"], r["rank"]): r for r in eff}
    pairs = [(by[k]["rise"], by[(k[0], k[1], "5-7", k[3])]["rise"]) for k in by if k[2] == "3-5" and (k[0], k[1], "5-7", k[3]) in by]
    wins, losses = sum(a_ < b for a_, b in pairs), sum(a_ > b for a_, b in pairs)
    win = lambda w: [r for r in eff if r["window"] == w]
    share = lambda w: (sum(r["pipeline_relapse"] for r in win(w)), len(win(w)))
    res = {"T1": {"smaller_at_3_5": wins, "smaller_at_5_7": losses, "pairs": len(pairs), "p": sign_test(wins, wins + losses), "holds": bool(sign_test(wins, wins + losses) < 0.05 and wins > losses)},
           "T2": {"relapses": share("3-5")[0], "effective": share("3-5")[1], "holds": bool(share("3-5")[1] > 0 and share("3-5")[0] / share("3-5")[1] < 0.2)},
           "T3": {"7-9": list(share("7-9")), "5-7": list(share("5-7")), "holds": bool(share("7-9")[1] > 0 and share("5-7")[1] > 0 and share("7-9")[0] / share("7-9")[1] < 0.2 and share("5-7")[0] / share("5-7")[1] > 0.5)},
           "by_window": {w: {"effective": len(win(w)), "mean_rise": float(np.mean([r["rise"] for r in win(w)])) if win(w) else None, "min_rise": min((r["rise"] for r in win(w)), default=None), "max_rise": max((r["rise"] for r in win(w)), default=None),
                             "mean_wmdp_official": float(np.mean([r["wmdp_official"] for r in win(w)])) if win(w) else None, "mean_wmdp_outdated": float(np.mean([r["wmdp_outdated"] for r in win(w)])) if win(w) else None,
                             "mean_cost": float(np.mean([r["cost_official"] for r in win(w)])) if win(w) else None} for w in ("3-5", "5-7", "7-9")},
           "not_effective": [{k: r[k] for k in ("model", "seed", "window", "lr", "wmdp_outdated", "wmdp_official", "rise")} for r in rows if not r["effective"]], "runs": rows}
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for k in ("T1", "T2", "T3"):
        print(k, res[k])
    for w, v in res["by_window"].items():
        print(w, {k: (round(100 * x, 1) if isinstance(x, float) else x) for k, x in v.items()})
    print("not effective:", [(r["model"], r["window"], round(100 * r["wmdp_outdated"], 1), round(100 * r["wmdp_official"], 1)) for r in res["not_effective"]])


if __name__ == "__main__":
    main()
