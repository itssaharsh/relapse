"""The pre-registered analysis of the single-line rescoring (results/PREREGISTRATION_6.md), written before its jobs started.

    python scripts/analyze_single_line.py [--fetch]
--fetch copies results.json and items.json of every job from the private stage-2 repo into results/single_line/.
Writes results/single_line/summary.json and prints S1 to S3.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analyze_map import MODELS, ROOT, download, paired_p, sign_test

OUT = os.path.join(ROOT, "results", "single_line")
RISE = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true"); a = ap.parse_args()
    if a.fetch:
        for v in MODELS:
            os.makedirs(os.path.join(OUT, v), exist_ok=True)
            for f in ("results.json", "items.json"):
                download(f"single_line/{v}/{f}", os.path.join(OUT, v, f))
    five = json.load(open(os.path.join(ROOT, "results", "shift2", "summary.json")))
    registered = {(r["model"], r["seed"], r["window"], r["lr"]): r for r in five["runs"]}      # pre-registration 5: which runs are effective, and their rise in the multiple-choice layout
    rng = np.random.default_rng(0)
    rows, check = [], []
    for v in MODELS:
        p = os.path.join(OUT, v)
        if not os.path.exists(os.path.join(p, "results.json")):
            continue
        R = {(r["model"], r["score_tok"], r["layout"]): r for r in json.load(open(os.path.join(p, "results.json")))}
        items = json.load(open(os.path.join(p, "items.json")))
        base = {t: R[("base", t, "single")] for t in ("official", "outdated")}
        for name in sorted({k[0] for k in R if k[0] != "base"}):
            _, window, lr, _, seed = name.split("_")
            reg = registered[(v, int(seed[1:]), window[1:], lr[2:])]
            off, old = R[(name, "official", "single")], R[(name, "outdated", "single")]
            delta, lo, hi, pval = paired_p(items[f"{name}|fp16|outdated|single|wmdp-bio"], items[f"{name}|fp16|official|single|wmdp-bio"], rng)
            rows.append({"model": v, "seed": int(seed[1:]), "window": window[1:], "lr": lr[2:], "effective": reg["effective"], "rise_registered_layout": reg["rise"],
                         "wmdp_outdated": old["wmdp-bio"], "wmdp_official": off["wmdp-bio"], "rise": off["wmdp-bio"] - old["wmdp-bio"], "ci95": [lo, hi], "p": pval,
                         "relapse": bool(delta >= RISE and pval < 0.05), "mmlu_lost_outdated": base["outdated"]["mmlu"] - old["mmlu"], "mmlu_lost_official": base["official"]["mmlu"] - off["mmlu"]})
            if (name, "official", "mcq") in R:      # the check: the registered layout scored again by this job
                check.append(abs((R[(name, "official", "mcq")]["wmdp-bio"] - R[(name, "outdated", "mcq")]["wmdp-bio"]) - reg["rise"]))
        check.append(abs(R[("base", "official", "mcq")]["wmdp-bio"] - R[("base", "outdated", "mcq")]["wmdp-bio"]))
    eff = lambda w: [r for r in rows if r["window"] == w and r["effective"]]
    d = eff("5-7")
    smaller, larger = sum(r["rise"] < r["rise_registered_layout"] for r in d), sum(r["rise"] > r["rise_registered_layout"] for r in d)
    share = lambda rs: (sum(r["relapse"] for r in rs), len(rs))
    hurt = [r for r in rows if r["mmlu_lost_outdated"] >= 0.03]; fine = [r for r in rows if r["mmlu_lost_outdated"] < 0.03]
    res = {"jobs": sorted({r["model"] for r in rows}), "runs": len(rows),
           "S1": {"smaller_single_line": smaller, "larger_single_line": larger, "pairs": len(d), "p": sign_test(smaller, smaller + larger) if smaller + larger else None,
                  "holds": bool(smaller + larger and sign_test(smaller, smaller + larger) < 0.05 and smaller > larger)},
           "S2": {"relapses": share(d)[0], "effective": share(d)[1], "holds": bool(d and share(d)[0] * 2 >= share(d)[1])},
           "S3": {"relapses": share(eff("3-5"))[0], "effective": share(eff("3-5"))[1], "holds": bool(eff("3-5") and share(eff("3-5"))[0] / share(eff("3-5"))[1] < 0.2)},
           "by_window": {w: {"effective": len(eff(w)), "relapses": share(eff(w))[0], "mean_rise": float(np.mean([r["rise"] for r in eff(w)])) if eff(w) else None,
                             "mean_rise_registered_layout": float(np.mean([r["rise_registered_layout"] for r in eff(w)])) if eff(w) else None,
                             "mean_wmdp_outdated": float(np.mean([r["wmdp_outdated"] for r in eff(w)])) if eff(w) else None, "mean_wmdp_official": float(np.mean([r["wmdp_official"] for r in eff(w)])) if eff(w) else None,
                             "mean_mmlu_lost_outdated": float(np.mean([r["mmlu_lost_outdated"] for r in eff(w)])) if eff(w) else None,
                             "mean_mmlu_lost_official": float(np.mean([r["mmlu_lost_official"] for r in eff(w)])) if eff(w) else None} for w in ("3-5", "5-7", "7-9")},
           "exploratory_mmlu_rule": {"lost_3_or_more_and_rose": [sum(r["rise"] >= RISE for r in hurt), len(hurt)], "lost_less_and_rose": [sum(r["rise"] >= RISE for r in fine), len(fine)]},
           "base": [{k: r[k] for k in ("seed", "window", "lr", "wmdp_outdated", "wmdp_official", "rise", "relapse", "mmlu_lost_outdated")} for r in rows if r["model"] == "base"],
           "check_registered_layout_max_difference": max(check) if check else None, "all_runs": rows}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for k in ("S1", "S2", "S3"):
        print(k, res[k])
    for w, v in res["by_window"].items():
        print(w, {k: (round(100 * x, 1) if isinstance(x, float) else x) for k, x in v.items()})
    print("MMLU rule (exploratory):", res["exploratory_mmlu_rule"], "| check of the registered layout, largest difference:", res["check_registered_layout_max_difference"])


if __name__ == "__main__":
    main()
