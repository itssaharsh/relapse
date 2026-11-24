"""Writes space/data/stage2.json, the data behind the demo page, from the Stage 2 result files.

    python scripts/build_space.py            # writes space/data/stage2.json
    python scripts/build_space.py --check    # also checks that the shared numbers read the same as the README

Every number comes from results/*/summary.json or results/final/check.json, computed the same way as
scripts/build_readme.py (the README results block) and scripts/paper_numbers.py glance() (Table 1 of the paper).
Percentages are stored as percent rounded to one decimal, changes as points, so the page shows the README's figures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "space", "data", "stage2.json")
sys.path.insert(0, HERE)
from analyze_map import RISE  # noqa: E402  the relapse line: a rise of at least 5 points with paired p < 0.05
from analyze_map_shim import paired  # noqa: E402

MODELS = ("global", "fire", "earth", "water", "base")
load = lambda *p: json.load(open(os.path.join(ROOT, "results", *p)))
pc = lambda x: round(100 * x, 1)          # a share as percent, one decimal
pt = lambda x: round(100 * x, 1) + 0.0    # a change in points, one decimal (no -0.0)
came_back = lambda change, p=0.0: bool(change >= RISE and p < 0.05)


def brief() -> dict:
    """SafeTune's default layers against layers 3 to 5: glance() in scripts/paper_numbers.py, same order of bootstrap draws."""
    s2, fin = load("shift2", "summary.json"), load("final", "summary.json")
    W = s2["by_window"]; rng = np.random.default_rng(0)
    relapses = {"3-5": 0, "5-7": 0}; checks = {"3-5": 0, "5-7": 0}; cost = {"3-5": [], "5-7": []}; benign = {"3-5": [], "5-7": []}
    for v in MODELS:
        rows = {(r["model"], r["prec"]): r for r in load("final", f"{v}_seed33", "results.json")}; items = load("final", f"{v}_seed33", "items.json")
        for (m, p), r in rows.items():
            if p != "fp16" or m == "base":
                continue
            w = re.search(r"_W(\d+-\d+)_", m).group(1)
            cost[w].append(100 * (rows[("base", "fp16")]["mmlu"] - r["mmlu"]))
            for f in ("Q4_0", "nf4"):
                delta, pv = paired(items[f"{m}|fp16|wmdp-bio"], items[f"{m}|{f}|wmdp-bio"], rng)
                relapses[w] += int(delta >= RISE and pv < 0.05); checks[w] += 1
        c = {(r["model"], r["prec"]): r for r in load("final", f"check_{v}", "certify.json") if "wmdp-bio" in r}
        for (m, p), r in c.items():
            if p == "fp16" and m != "base":
                benign[re.search(r"_W(\d+-\d+)_", m).group(1)].append(100 * (c[("base", "fp16")]["benign-bio"] - r["benign-bio"]))
    tok = {"5-7": s2["T3"]["5-7"], "3-5": [s2["T2"]["relapses"], s2["T2"]["effective"]]}
    col = lambda w, mean_key: {
        "pipeline": pc(W[w]["mean_wmdp_outdated"]), "deployed": pc(W[w]["mean_wmdp_official"]), "mean_rise": pt(W[w]["mean_rise"]),
        "tok_relapses": tok[w][0], "tok_runs": tok[w][1],
        "official_throughout": pc(fin["Q4"][mean_key]), "quant_relapses": relapses[w], "quant_checks": checks[w],
        "mmlu_lost": round(float(np.mean(cost[w])), 1), "benign_lost": round(float(np.median(benign[w])), 1)}
    tok_models = sorted({r["model"] for r in s2["runs"] if r["window"] == "3-5" and r["effective"] and r["train_tok"] == "outdated"})
    return {"default": col("5-7", "mean_wmdp_5_7"), "released": col("3-5", "mean_wmdp_3_5"),
            "tok_models": len(tok_models), "final_models": len(MODELS)}


def relapse_cases() -> dict:
    """The runs where one 4-bit format relapsed: the table in the README's point 2."""
    m, deep = load("map", "summary.json"), load("deep", "summary.json")
    eff = [r for r in m["runs"] if r["effective"]]
    held = sum(not (r["Q4_0"]["relapse"] or r["nf4"]["relapse"]) for r in eff)
    runs = {r["name"]: r for r in deep["runs"]}
    real = {n: v for models in deep["part_b_real_files"].values() for n, v in models.items()}
    picks = [("base", "10 to 12", 12, runs["base_W10-12_lr0.001_s12"], real["base_relapse_base_x10_lr1e-3_s12"]),
             ("base", "10 to 12", 43, runs["base_W10-12_lr0.00045_s43"], real["deep_real_s43_base_W10-12_lr0.00045_s43"]),
             ("fire", "13 to 15", 44, runs["fire_W13-15_lr0.002_s44"], real["deep_real_fire44_fire_W13-15_lr0.002_s44"]),
             ("water", "10 to 12", 42, runs["water_W10-12_lr0.001_s42"], real["deep_real_water42_water_W10-12_lr0.001_s42"])]
    cases = []
    for model, layers, seed, r, rf in picks:
        cells = {"q4_0": {"score": pc(r["Q4_0"]["wmdp"]), "change": pt(r["Q4_0"]["delta"]), "came_back": bool(r["Q4_0"]["relapse"])},
                 "nf4": {"score": pc(r["nf4"]["wmdp"]), "change": pt(r["nf4"]["delta"]), "came_back": bool(r["nf4"]["relapse"])},
                 "file_q4_0": {"score": pc(rf["Q4_0"]["file"]), "change": pt(rf["Q4_0"]["rise"]), "came_back": bool(rf["Q4_0"]["relapse"])},
                 "file_q4_k_m": {"score": pc(rf["Q4_K_M"]["file"]), "change": pt(rf["Q4_K_M"]["rise"]), "came_back": bool(rf["Q4_K_M"]["relapse"])}}
        cases.append({"model": model, "layers": layers, "seed": seed, "fp16": pc(r["wmdp_fp16"]), "cells": cells})
    A = deep["part_a"]
    return {"map_runs": len(m["runs"]), "map_effective": len(eff), "map_held": held, "cases": cases,
            "another_format_held": all(not all(c["came_back"] for c in x["cells"].values()) for x in cases),
            "any_q4_k_m_file_relapsed": any(x["cells"]["file_q4_k_m"]["came_back"] for x in cases),
            "base_fresh_seeds": [A["R1"]["relapsing_seeds_Q4_0"], A["R1"]["of"]]}


def released() -> dict:
    """The published model of each kind: layers 3 to 5, seed 34 (scripts/publish_cards.py), from results/final/check.json."""
    check = load("final", "check.json"); out = {}
    for v in MODELS:
        name = next(n for n, e in check.items() if e["model"] == v and n.endswith("_s34"))
        e = check[name]; f = e["fp16"]; real = e["real_files"]["wmdp-bio"]; tok = e["outdated_tokenizer"]; ft = e["fine_tuning"]
        fmt = lambda k: {"score": pc(real[k]["file"]), "change": pt(real[k]["change"]), "p": round(real[k]["p"], 3), "came_back": came_back(real[k]["change"], real[k]["p"])}
        five = f["gain:five_shot"]; ftc = ft["after_hazard_questions"] - ft["before"]
        out[v] = {"run": name, "seed": 34, "layers": "3 to 5", "repo": f"https://huggingface.co/itssaharsh/relapse-tiny-aya-{v}-unlearn-bio",
                  "original": pc(f["wmdp-bio"]["original"]), "unlearned": pc(f["wmdp-bio"]["model"]), "forgot": pt(-f["wmdp-bio"]["difference"]),
                  "file_q4_0": fmt("Q4_0"), "file_q4_k_m": fmt("Q4_K_M"), "nf4": fmt("nf4"),
                  "outdated_tokenizer": {"score": pc(tok["outdated"]), "change": pt(tok["difference"]), "p": round(tok["p"], 3), "came_back": came_back(tok["difference"], tok["p"])},
                  "five_examples": {"score": pc(f["attack:five_shot"]["model"]), "plain": pc(ft["before"]), "change": pt(five["difference"]), "p": round(five["p"], 3),
                                    "came_back": came_back(five["difference"], five["p"])},
                  "fine_tuning": {"score": pc(ft["after_hazard_questions"]), "before": pc(ft["before"]), "change": pt(ftc), "came_back": bool(ftc >= RISE),
                                  "original_same_questions": pc(ft["original"])},
                  "mmlu": {"original": pc(f["mmlu"]["original"]), "unlearned": pc(f["mmlu"]["model"]), "lost": pt(-f["mmlu"]["difference"])},
                  "benign": {"original": pc(f["benign-bio"]["original"]), "unlearned": pc(f["benign-bio"]["model"]), "lost": pt(-f["benign-bio"]["difference"])}}
    return out


def costs() -> dict:
    """What it costs and what breaks it: the README's 'Costs' and 'Breaks' lines, over both seeds of every released kind."""
    check = load("final", "check.json"); E10 = list(check.values())
    ben = [-e["fp16"]["benign-bio"]["difference"] for e in E10]
    ft = [e["fine_tuning"]["after_hazard_questions"] - e["fine_tuning"]["before"] for e in E10]
    runs = [{"model": e["model"], "seed": int(n.rsplit("_s", 1)[1]), "before": pc(e["fine_tuning"]["before"]), "after": pc(e["fine_tuning"]["after_hazard_questions"]),
             "change": pt(x), "came_back": bool(x >= RISE), "original": pc(e["fine_tuning"]["original"])} for (n, e), x in zip(check.items(), ft)]
    originals = [e["fp16"]["wmdp-bio"]["original"] for e in E10]
    return {"benign_low": pt(min(ben)), "benign_high": pt(max(ben)), "mmlu_most_lost": pt(max(-e["fp16"]["mmlu"]["difference"] for e in E10)),
            "ft_came_back": sum(x >= RISE for x in ft), "ft_runs": len(E10), "fine_tuning": runs,
            "original_low": pc(min(originals)), "original_high": pc(max(originals))}


def hypotheses() -> dict:
    """Counted from the README's own hypothesis table, as build_readme.py writes it."""
    from build_readme import block
    rows = [line for line in block().split("### Every registered hypothesis")[1].splitlines() if line.startswith("| ") and not line.startswith("| Experiment")]
    failed = [r for r in rows if "**Failed" in r or "**Not significant" in r]
    return {"registered": len(rows), "failed": len(failed)}


def build() -> dict:
    return {"about": "Written by scripts/build_space.py from results/. Percentages are percent, changes are points. A run 'came back' when its score rose by at least "
                     f"{pt(RISE):.0f} points with paired p < 0.05 (fine-tuning: at least {pt(RISE):.0f} points).",
            "rise_points": pt(RISE), "brief": brief(), "relapses": relapse_cases(), "models": released(), "costs": costs(), "hypotheses": hypotheses()}


def check(d: dict) -> None:
    """The figures the page shares with the README must read the same there."""
    readme = open(os.path.join(ROOT, "README.md")).read()
    b, c, r = d["brief"], d["costs"], d["relapses"]
    want = [f"{b['default']['tok_relapses']} of {b['default']['tok_runs']}", f"{b['released']['tok_relapses']} of {b['released']['tok_runs']}",
            f"{b['default']['pipeline']:.1f}%", f"{b['released']['pipeline']:.1f}%", f"{b['default']['deployed']:.1f}%", f"{b['released']['deployed']:.1f}%",
            f"{b['default']['official_throughout']:.1f}%", f"{b['released']['official_throughout']:.1f}%",
            f"| {b['default']['mmlu_lost']:.1f} | {b['released']['mmlu_lost']:.1f} |", f"| {b['default']['benign_lost']:.1f} | {b['released']['benign_lost']:.1f} |",
            f"| {b['default']['quant_relapses']} | {b['released']['quant_relapses']} |", f"(of {b['default']['quant_checks']} checks)",
            f"({b['default']['mean_rise']:+.1f} points on average)", f"({b['released']['mean_rise']:+.1f} on average)",
            f"{r['map_held']} of the {r['map_effective']} runs", f"map of {r['map_runs']} runs",
            f"benign biology falls by {c['benign_low']:.1f} to {c['benign_high']:.1f} points", f"on {c['ft_came_back']} of {c['ft_runs']} models"]
    for x in r["cases"]:
        cl = x["cells"]
        want.append(f"| {x['fp16']:.1f}% | {cl['q4_0']['score']:.1f}% ({cl['q4_0']['change']:+.1f})")
        want.append(f"| {cl['file_q4_0']['score']:.1f}% ({cl['file_q4_0']['change']:+.1f}) | {cl['file_q4_k_m']['score']:.1f}% ({cl['file_q4_k_m']['change']:+.1f}) |")
    missing = [w for w in want if w not in readme]
    if missing:
        raise SystemExit("not in README.md: " + "; ".join(missing))
    print(f"checked {len(want)} figures against README.md")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    d = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(d, fh, indent=1); fh.write("\n")
    print("wrote", os.path.relpath(OUT, ROOT))
    if a.check:
        check(d)


if __name__ == "__main__":
    main()
