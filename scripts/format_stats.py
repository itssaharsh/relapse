"""Paired fp16-to-file comparisons for one grid run, from its saved AuditKIT runs.

The grid kernel writes the fp16-to-Q4_K_M comparison itself. This adds the other files whose AuditKIT runs were
saved (for the deeper-window control: the real llama.cpp Q4_0 file): did the hazard score move from fp16 to the file,
and what did the model lose against the original in that file. Same questions, AuditKIT's paired bootstrap.

    python scripts/format_stats.py <dir with the audit_<tag>_*.json runs> <tag>
writes results/raw/audit_formats_<tag>.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))


def main(src: str, tag: str) -> None:
    from auditkit.report import RunResult
    from auditkit_mcq import audit_compare

    runs = {}
    for f in glob.glob(os.path.join(src, f"audit_{tag}_*_*.json")):
        m = re.match(rf"audit_{tag}_(.+)_(fp16|bf16|Q8_0|Q6_K|Q4_K_M|Q4_0|nf4)_(wmdp-bio|mmlu)\.json$", os.path.basename(f))
        if m:
            runs[m.groups()] = f
    done = {(d["model"], d["kind"], d.get("to") or d.get("prec")) for d in json.load(open(os.path.join(src, f"audit_{tag}.json")))}
    load = lambda k: RunResult.load(runs[k])
    keep = ("n", "acc_a", "acc_b", "delta", "ci95", "p_value", "right_to_wrong", "wrong_to_right")
    rows = []
    for name, prec in sorted({(n, p) for n, p, _ in runs if p != "fp16"}):
        if (name, "forget_holds", prec) not in done and (name, "fp16", "wmdp-bio") in runs:
            c = audit_compare(load((name, "fp16", "wmdp-bio")), load((name, prec, "wmdp-bio")))
            rows.append({**{k: c[k] for k in keep}, "model": name, "kind": "forget_holds", "from": "fp16", "to": prec})
        if name != "base" and (name, "price", prec) not in done and ("base", prec, "mmlu") in runs:
            c = audit_compare(load(("base", prec, "mmlu")), load((name, prec, "mmlu")))
            rows.append({**{k: c[k] for k in keep}, "model": name, "kind": "price", "prec": prec})
    for r in rows:
        print(r)
    json.dump(rows, open(os.path.join(ROOT, "results", "raw", f"audit_formats_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
