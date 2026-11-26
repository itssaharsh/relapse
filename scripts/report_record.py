"""The reports made to the kits' repositories: one issue each, with the time GitHub recorded.

    python scripts/report_record.py            # writes results/survey/reports.json (needs the gh command line tool)

The texts are in hf_updates/reports/. The note for the Kaggle model page is listed with its state.
"""
from __future__ import annotations

import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSUES = {"SafeTune": 105, "CircuitKIT": 60, "AuditKIT": 54, "CuratorKIT": 58}
KAGGLE = {"page": "https://www.kaggle.com/models/cohere-labs/tiny-aya", "text": "hf_updates/reports/kaggle_discussion.md", "posted": None,
          "note": "Kaggle's public API has no call that creates a discussion; the text is ready and was not posted from this machine."}


def main() -> None:
    out = {"issues": [], "kaggle_discussion": KAGGLE}
    for kit, n in ISSUES.items():
        d = json.loads(subprocess.run(["gh", "api", f"repos/Lexsi-Labs/{kit}/issues/{n}"], capture_output=True, text=True, check=True).stdout)
        out["issues"].append({"kit": kit, "url": d["html_url"], "title": d["title"], "created_at": d["created_at"]})
    json.dump(out, open(os.path.join(ROOT, "results", "survey", "reports.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
