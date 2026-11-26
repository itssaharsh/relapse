"""When each pre-registration became public: GitHub's own record of pushes against the commit that added the document.

    python scripts/push_record.py            # writes results/survey/pushes.json (needs the gh command line tool)

GitHub records the time of every push to a repository (GET /repos/{owner}/{repo}/activity). For each pre-registration
document, this finds the commit that added it and the first push to main whose history contains that commit. A document
with no such push was not public when this was run, and is recorded as such.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "itssaharsh/relapse"
DOCS = ["PREREGISTRATION.md"] + [f"PREREGISTRATION_{i}.md" for i in range(2, 7)]
git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
when = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> None:
    rows = subprocess.run(["gh", "api", f"repos/{REPO}/activity", "--paginate", "--jq", '.[] | [.timestamp,.activity_type,.ref,.after] | @tsv'],
                          capture_output=True, text=True, check=True).stdout.strip().split("\n")
    pushes = sorted((t, a) for t, kind, ref, a in (r.split("\t") for r in rows) if ref == "refs/heads/main")
    out = {"source": f"GET /repos/{REPO}/activity", "fetched": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "pushes": [{"time": t, "head": a} for t, a in pushes], "preregistrations": []}
    for n, doc in enumerate(DOCS, 1):
        commit, committed = git("log", "--diff-filter=A", "--format=%H %cI", "--", f"results/{doc}").stdout.strip().split("\n")[-1].split()
        first = next((t for t, a in pushes if git("cat-file", "-e", a).returncode == 0 and git("merge-base", "--is-ancestor", commit, a).returncode == 0), None)
        out["preregistrations"].append({"n": n, "file": f"results/{doc}", "commit": commit, "committed": when(committed).strftime("%Y-%m-%dT%H:%M:%SZ"), "first_public_push": first,
                                        "seconds_from_commit_to_public": None if first is None else int((when(first) - when(committed)).total_seconds())})
    json.dump(out, open(os.path.join(ROOT, "results", "survey", "pushes.json"), "w"), indent=1)
    print(json.dumps(out["preregistrations"], indent=1))


if __name__ == "__main__":
    main()
