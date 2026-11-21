"""Paired statistics for the Stage 1 models rescored with the correct tokenizer.

    python scripts/rescore_stats.py     # reads the saved AuditKIT runs from the private stage-2 repo
writes results/correct_tokenizer/audit_stage1_rescored.json
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
MODELS = {"rmu_default_layers": "standard", "rmu_circuit_layers": "circuit"}
FILES = ("fp16", "Q4_0", "nf4", "Q4_K_M")
KEEP = ("n", "acc_a", "acc_b", "delta", "ci95", "p_value", "right_to_wrong", "wrong_to_right")


def main() -> None:
    from auditkit.report import RunResult
    from auditkit_mcq import audit_compare
    from huggingface_hub import snapshot_download

    src = snapshot_download("itssaharsh/relapse-stage2", repo_type="dataset", allow_patterns=["real/stage1_retok/*"]) + "/real/stage1_retok"
    run = lambda m, p, k: RunResult.load(f"{src}/audit_stage1_retok_{m}_{p}_{k}.json")
    fmt = lambda p: "NF4" if p == "nf4" else p
    rows = []
    for raw, method in [("base", "original")] + list(MODELS.items()):
        for p in FILES[1:]:
            c = audit_compare(run(raw, "fp16", "wmdp-bio"), run(raw, p, "wmdp-bio"))
            rows.append({**{k: c[k] for k in KEEP}, "model": raw, "method": method, "kind": "forget_holds", "from": "fp16", "to": fmt(p)})
        if raw != "base":
            for p in FILES:
                c = audit_compare(run("base", p, "mmlu"), run(raw, p, "mmlu"))
                rows.append({**{k: c[k] for k in KEEP}, "model": raw, "method": method, "kind": "price", "prec": fmt(p)})
    for raw in ("rmu_deeper", "rmu_deeper_file"):       # the two models behind the withdrawn claims, kept so the correction can be checked
        for p in FILES[1:]:
            c = audit_compare(run(raw, "fp16", "wmdp-bio"), run(raw, p, "wmdp-bio"))
            rows.append({**{k: c[k] for k in KEEP}, "model": raw, "method": "withdrawn", "kind": "forget_holds", "from": "fp16", "to": fmt(p)})
    for p in FILES:
        for k in ("wmdp-bio", "mmlu"):
            c = audit_compare(run("rmu_default_layers", p, k), run("rmu_circuit_layers", p, k))
            rows.append({**{x: c[x] for x in KEEP}, "kind": "recipe", "set": k, "prec": fmt(p), "a": "default layers", "b": "CircuitKIT layers"})
    json.dump(rows, open(os.path.join(ROOT, "results", "correct_tokenizer", "audit_stage1_rescored.json"), "w"), indent=1)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
