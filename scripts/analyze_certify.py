"""The full check of the released models (results/PREREGISTRATION_3.md, amendment 1): reported whatever it shows.

    python scripts/analyze_certify.py [--fetch]
Reads the full-check and real-file jobs of the five final models, writes results/final/check.json and prints a summary.
AuditKIT's saved runs hold the question text, so only the comparisons computed from them are written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

from analyze_map import MODELS, ROOT, download, paired_p

FINAL = os.path.join(ROOT, "results", "final")
RAW_RUN = re.compile(r"audit_(.+)_(fp16|nf4|Q4_0|Q4_K_M)_(wmdp-bio|mmlu)\.json$")
SETS = ("wmdp-bio", "mmlu", "benign-bio", "attack:zero_shot", "attack:five_shot", "attack:role_play", "probe:en_para", "probe:hi", "probe:sw", "probe:ar", "probe:es")


def fetch() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files("itssaharsh/relapse-stage2", repo_type="dataset")
    sys.path.insert(0, os.path.join(ROOT, "kaggle"))
    from auditkit.report import RunResult
    from auditkit_mcq import audit_compare
    for v in MODELS:
        d = os.path.join(FINAL, f"check_{v}"); os.makedirs(d, exist_ok=True)
        for f in ("certify.json", "certify_items.json"):
            download(f"real/final_certify_{v}/{f}", os.path.join(d, f))
        download(f"real/final_real_{v}/grid_final_real_{v}.json", os.path.join(d, "real_grid.json"))
        raw = {(m.group(1), m.group(2), m.group(3)): f for f in files if f.startswith(f"real/final_real_{v}/") and (m := RAW_RUN.search(f))}
        run = lambda f: RunResult.load(hf_hub_download("itssaharsh/relapse-stage2", f, repo_type="dataset"))
        out = {}
        for (name, prec, k), f in sorted(raw.items()):
            if prec != "fp16" and (name, "fp16", k) in raw:
                c = audit_compare(run(raw[(name, "fp16", k)]), run(f))
                out.setdefault(name, {}).setdefault(k, {})[prec] = {"fp16": c["acc_a"], "file": c["acc_b"], "change": c["delta"], "ci95": c["ci95"], "p": c["p_value"]}
        json.dump(out, open(os.path.join(d, "real_files.json"), "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true"); a = ap.parse_args()
    if a.fetch:
        fetch()
    rng = np.random.default_rng(0)
    res = {}
    for v in MODELS:
        d = os.path.join(FINAL, f"check_{v}")
        rows = json.load(open(os.path.join(d, "certify.json"))); items = json.load(open(os.path.join(d, "certify_items.json")))
        real = json.load(open(os.path.join(d, "real_files.json")))
        by = {(r["model"], r["prec"]): r for r in rows if "wmdp-bio" in r}
        relearn = {(r["model"], r["attack"]): r["acc"] for r in rows if "attack" in r}
        for name in sorted({m for m, _ in by if "_W3-5_" in m}):
            entry = {"model": v, "original": {}, "fp16": {}, "formats": {}, "real_files": real.get(f"final_real_{v}_{name}", real.get(name, {}))}
            for k in SETS:
                if f"{name}|fp16|{k}" not in items:
                    continue
                delta, lo, hi, p = paired_p(items[f"base|fp16|{k}"], items[f"{name}|fp16|{k}"], rng)
                entry["fp16"][k] = {"original": by[("base", "fp16")][k], "model": by[(name, "fp16")][k], "difference": delta, "ci95": [lo, hi], "p": p}
            for attack in ("five_shot", "role_play"):      # does the attack bring the knowledge back? same model, same questions, with and without the attack
                delta, lo, hi, p = paired_p(items[f"{name}|fp16|attack:zero_shot"], items[f"{name}|fp16|attack:{attack}"], rng)
                entry["fp16"][f"gain:{attack}"] = {"difference": delta, "ci95": [lo, hi], "p": p}
            delta, lo, hi, p = paired_p(items[f"{name}|fp16|wmdp-bio"], items[f"{name}|fp16|outdated-tokenizer:wmdp-bio"], rng)
            entry["outdated_tokenizer"] = {"official": by[(name, "fp16")]["wmdp-bio"], "outdated": by[(name, "fp16")]["outdated-tokenizer:wmdp-bio"], "difference": delta, "ci95": [lo, hi], "p": p,
                                           "original_outdated": by[("base", "fp16")]["outdated-tokenizer:wmdp-bio"]}
            for prec in ("Q4_0", "nf4"):
                entry["formats"][prec] = {k: by[(name, prec)][k] for k in SETS if (name, prec) in by and k in by[(name, prec)]}
            entry["fine_tuning"] = {"before": by[(name, "fp16")]["attack:zero_shot"], "after_hazard_questions": relearn.get((name, "relearn_hazard")), "after_unrelated_questions": relearn.get((name, "relearn_unrelated")),
                                    "original_after_hazard_questions": relearn.get(("base", "relearn_hazard")), "original": by[("base", "fp16")]["attack:zero_shot"]}
            res[name] = entry
    json.dump(res, open(os.path.join(FINAL, "check.json"), "w"), indent=1)
    f = lambda x: "  -- " if x is None else f"{100 * x:5.1f}"
    print("model                           WMDP  MMLU  benign  5shot  role  para   hi    sw    ar    es  | old-tok | Q4_0 file  Q4_K_M file | fine-tune: before -> hazard / unrelated")
    for name, e in res.items():
        r = e["real_files"].get("wmdp-bio", {})
        print(f"{name:30s}", " ".join(f(e["fp16"][k]["model"]) for k in SETS if k != "attack:zero_shot" and k in e["fp16"]), "|", f(e["outdated_tokenizer"]["outdated"]), "|",
              f(r.get("Q4_0", {}).get("file")), f(r.get("Q4_K_M", {}).get("file")), "|", f(e["fine_tuning"]["before"]), "->", f(e["fine_tuning"]["after_hazard_questions"]), "/", f(e["fine_tuning"]["after_unrelated_questions"]))
    o = next(iter(res.values()))
    for v in MODELS:
        e = next(x for x in res.values() if x["model"] == v)
        print(f"original {v:21s}", " ".join(f(e["fp16"][k]["original"]) for k in SETS if k != "attack:zero_shot" and k in e["fp16"]), "|", f(e["outdated_tokenizer"]["original_outdated"]), "|", " " * 11, "|", f(e["fine_tuning"]["original"]), "->", f(e["fine_tuning"]["original_after_hazard_questions"]))


if __name__ == "__main__":
    main()
