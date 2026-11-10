"""Paired comparison of the two recipes on the same questions, from the saved AuditKIT runs.

The grids compare each unlearned model with the original. This compares the CircuitKIT-layer model with the
default-layer model directly: same seed, same file format, same questions, AuditKIT's paired bootstrap.

    python scripts/recipe_stats.py [dir with the audit_*.json runs]     # default: fetched from the HF dataset
writes results/raw/audit_recipes.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
PAIRS = [  # (seed, default-layer run, CircuitKIT-layer run)
    (0, "rmu_default_layers", "rmu_circuit_layers"),
    (1, "rmu_default_layers_seed1", "rmu_circuit_layers_seed1"),
]


def main() -> None:
    from auditkit.report import RunResult
    from auditkit_mcq import audit_compare

    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src is None:
        from huggingface_hub import snapshot_download
        src = snapshot_download("itssaharsh/relapse-runs", repo_type="dataset", allow_patterns=["audit_runs/*.json"])

    def run(name: str, prec: str, metric: str) -> RunResult:
        hits = glob.glob(os.path.join(src, "**", f"audit_*_{name}_{prec}_{metric}.json"), recursive=True)
        return RunResult.load(hits[0])

    rows = []
    for seed, default, circuit in PAIRS:
        for prec in ("fp16", "Q4_K_M"):
            for metric in ("mmlu", "wmdp-bio"):
                c = audit_compare(run(default, prec, metric), run(circuit, prec, metric))
                rows.append({"kind": "recipe", "seed": seed, "prec": prec, "set": metric, "a": "default layers", "b": "CircuitKIT layers",
                             "n": c["n"], "acc_a": c["acc_a"], "acc_b": c["acc_b"], "delta": c["delta"], "ci95": c["ci95"],
                             "p_value": c["p_value"], "right_to_wrong": c["right_to_wrong"], "wrong_to_right": c["wrong_to_right"]})
                print(rows[-1])
    json.dump(rows, open(os.path.join(ROOT, "results", "raw", "audit_recipes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
