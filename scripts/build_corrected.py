"""Build the page data and the write-up tables from the runs scored with the correct tokenizer.

    python scripts/build_corrected.py
Reads results/correct_tokenizer/, writes space/data/results.json, fills the generated tables in README.md and docs/ARTICLE.md.
The Stage 1 numbers measured with the outdated tokenizer stay in results/raw/ for the record and are not used here.
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "correct_tokenizer")
OUT = os.path.join(ROOT, "space", "data", "results.json")
RUNS = {"base": "original", "rmu_default_layers": "standard", "rmu_circuit_layers": "circuit"}
FORMATS = [
    {"id": "fp16", "label": "fp16", "bits": 16, "size_gb": 6.7, "file": "tiny-aya-global-f16.gguf", "how": "full precision"},
    {"id": "Q4_K_M", "label": "Q4_K_M", "bits": 4.8, "size_gb": 2.1, "file": "tiny-aya-global-Q4_K_M.gguf", "how": "llama.cpp"},
    {"id": "Q4_0", "label": "Q4_0", "bits": 4.5, "size_gb": 1.9, "file": "Q4_0, exact round trip of the decoder weights", "how": "round trip"},
    {"id": "NF4", "label": "NF4", "bits": 4, "size_gb": 2.0, "file": "NF4, exact round trip of the decoder weights", "how": "bitsandbytes"},
]
METHODS = [
    {"id": "original", "label": "Original Tiny Aya", "short": "Original"},
    {"id": "standard", "label": "SafeTune RMU at its default layers (5 to 7)", "short": "Default layers"},
    {"id": "circuit", "label": "SafeTune RMU at the layers CircuitKIT found (7 to 9)", "short": "CircuitKIT layers"},
]
PIPELINE = [
    {"tool": "CuratorKIT", "did": "quality-gated the translated question sets (their scores are being re-run with the correct tokenizer)"},
    {"tool": "CircuitKIT", "did": "ranked the early layers by attribution (EAP-IG), which moved the edit off the default layers"},
    {"tool": "SafeTune", "did": "removed the knowledge (RMU unlearning)"},
    {"tool": "AuditKIT", "did": "scored every file with paired confidence intervals"},
]


def norm(p: str) -> str:
    return "NF4" if p.lower() == "nf4" else p


def build(template: dict) -> dict:
    grid: dict = {}
    for r in json.load(open(os.path.join(SRC, "grid_stage1_rescored.json"))):
        m = RUNS.get(r["model"])
        if m:
            grid.setdefault(m, {})[norm(r["prec"])] = {"forget": round(r["wmdp-bio"], 6), "retain": round(r["mmlu"], 6),
                                                       "forget_pcorrect": r["wmdp-bio_pcorrect"], "retain_pcorrect": r["mmlu_pcorrect"]}
    rows = json.load(open(os.path.join(SRC, "audit_stage1_rescored.json")))
    return {"run": {"date": "2026-11-21", "label": "Hugging Face T4, rescored with the current official tokenizer"},
            "model": template["model"], "benchmark": template["benchmark"], "formats": FORMATS, "methods": METHODS, "grid": grid,
            "pipeline": PIPELINE, "links": template["links"], "stats": [r for r in rows if r["kind"] != "recipe" and r.get("method") != "withdrawn"],
            "recipes": [r for r in rows if r["kind"] == "recipe"],
            "circuit": {"default_rmu_layers": [5, 6, 7], "circuit_rmu_layers": [7, 8, 9]}}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def scores_table(res: dict) -> str:
    out = ["| File (WMDP-bio / MMLU) | Original | Default layers | CircuitKIT layers |", "|---|---|---|---|"]
    for f in FORMATS:
        out.append(f"| {f['label']} | " + " | ".join(f"{pct(res['grid'][m['id']][f['id']]['forget'])} / {pct(res['grid'][m['id']][f['id']]['retain'])}" for m in METHODS) + " |")
    return "\n".join(out)


def recipes_table(res: dict) -> str:
    fp = lambda p: "< 0.001" if p < 0.0005 else f"{p:.3f}"
    sg = lambda x: f"{100 * x:+.1f}"
    out = ["| File | WMDP-bio: default / CircuitKIT | Difference (95% interval) | p | MMLU: default / CircuitKIT | Difference (95% interval) | p |", "|---|---|---|---|---|---|---|"]
    for f in FORMATS:
        w = next(r for r in res["recipes"] if r["prec"] == f["id"] and r["set"] == "wmdp-bio")
        m = next(r for r in res["recipes"] if r["prec"] == f["id"] and r["set"] == "mmlu")
        out.append(f"| {f['label']} | {pct(w['acc_a'])} / {pct(w['acc_b'])} | {sg(w['delta'])} ({sg(w['ci95'][0])} to {sg(w['ci95'][1])}) | {fp(w['p_value'])} | "
                   f"{pct(m['acc_a'])} / {pct(m['acc_b'])} | {sg(m['delta'])} ({sg(m['ci95'][0])} to {sg(m['ci95'][1])}) | {fp(m['p_value'])} |")
    return "\n".join(out)


def main() -> None:
    res = build(json.load(open(OUT)))
    json.dump(res, open(OUT, "w"), indent=1)
    fill = lambda text, name, body: re.sub(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", lambda m: m.group(1) + body + "\n" + m.group(2), text, flags=re.S)
    for doc in ("README.md", os.path.join("docs", "ARTICLE.md")):
        p = os.path.join(ROOT, doc)
        t = fill(fill(open(p).read(), "scores", scores_table(res)), "recipes", recipes_table(res))
        open(p, "w").write(t)
    print(scores_table(res)); print(recipes_table(res))


if __name__ == "__main__":
    main()
