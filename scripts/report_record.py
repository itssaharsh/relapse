"""The reports made to the kits' repositories: one issue each, with the time GitHub recorded.

    python scripts/report_record.py            # writes results/survey/reports.json (needs the gh command line tool)

The texts are in hf_updates/reports/. Also recorded: the discussion on the Kaggle model page and the commits that
replaced the outdated tokenizer file in the five Stage 1 repositories. All times are UTC, as the three sites record them.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISSUES = {"SafeTune": 105, "CircuitKIT": 60, "AuditKIT": 54, "CuratorKIT": 58}
KAGGLE_TOPIC = 762145      # https://www.kaggle.com/models/cohere-labs/tiny-aya/discussion/762145, posted by the author by hand
STAGE1 = ["tiny-aya-global-unlearn-bio-default-layers", "tiny-aya-global-unlearn-bio-circuitkit-layers", "tiny-aya-global-unlearn-bio-circuitkit-4bit-aware",
          "tiny-aya-global-unlearn-bio-deeper-layers", "tiny-aya-global-unlearn-bio-deeper-layers-file-trained"]


def kaggle_topic(previous: dict) -> dict:
    """The discussion's record from Kaggle's API (needs the kagglesdk package and a Kaggle login); otherwise the record saved before."""
    try:
        from kagglesdk import KaggleClient
        from kagglesdk.discussions.types import discussions_api_service as t
        r = t.ApiGetTopicRequest(); r.id = KAGGLE_TOPIC
        with KaggleClient() as c:
            d = c.discussions.discussion_api_client.get_topic(r).to_dict()["topic"]
        return {"url": "https://www.kaggle.com" + d["url"], "title": d["title"], "posted": d["postDate"][:19] + "Z", "text": "hf_updates/reports/kaggle_discussion.md"}
    except Exception:
        return previous


def main() -> None:
    path = os.path.join(ROOT, "results", "survey", "reports.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    out = {"times": "UTC, as GitHub, Kaggle and Hugging Face record them", "issues": [], "kaggle_discussion": kaggle_topic(old.get("kaggle_discussion", {})), "stage1_tokenizer_fix": []}
    for kit, n in ISSUES.items():
        d = json.loads(subprocess.run(["gh", "api", f"repos/Lexsi-Labs/{kit}/issues/{n}"], capture_output=True, text=True, check=True).stdout)
        out["issues"].append({"kit": kit, "url": d["html_url"], "title": d["title"], "created_at": d["created_at"]})
    for name in STAGE1:      # the commit that replaced tokenizer_config.json, from Hugging Face's public commit list
        commits = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/models/itssaharsh/{name}/commits/main", timeout=40).read())
        c = next(c for c in commits if c["title"].startswith("Tokenizer setting"))
        out["stage1_tokenizer_fix"].append({"repo": f"itssaharsh/{name}", "commit": c["id"], "created_at": c["date"][:19] + "Z"})
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
