"""Check, signed out, that every published artifact is public and ungated (hackathon rule)."""
from __future__ import annotations

import json
import sys
import urllib.request

USER = "itssaharsh"
MODELS = ["tiny-aya-global-unlearn-bio-default-layers", "tiny-aya-global-unlearn-bio-circuitkit-layers",
          "tiny-aya-global-unlearn-bio-circuitkit-4bit-aware", "tiny-aya-global-unlearn-bio-deeper-layers",
          "tiny-aya-global-unlearn-bio-deeper-layers-file-trained"]
SWEEP = "tiny-aya-global-unlearn-bio-sweep"   # the edited tensors of every run
N_RUNS = 30
DATASETS = ["relapse-runs", "relapse-wmdp-bio-probes"]
COLLECTIONS = ["relapse-unlearned-tiny-aya-models", "relapse-runs-logs-and-probe-sets"]
PAGES = [f"https://huggingface.co/spaces/{USER}/relapse", f"https://{USER}-relapse.static.hf.space/index.html",
         f"https://github.com/{USER}/relapse"]


def get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "relapse-check"}), timeout=30) as r:
        return r.status, r.read()


def main() -> int:
    bad = []
    for m in MODELS:
        _, body = get(f"https://huggingface.co/api/models/{USER}/{m}")
        d = json.loads(body)
        if d.get("private") or d.get("gated"):
            bad.append(f"model {m} private/gated")
        if not any(s["rfilename"].endswith(".gguf") for s in d.get("siblings", [])):
            bad.append(f"model {m} has no GGUF")
    d = json.loads(get(f"https://huggingface.co/api/models/{USER}/{SWEEP}")[1])
    n = sum(s["rfilename"].endswith(".safetensors") for s in d.get("siblings", []))
    if d.get("private") or d.get("gated") or n != N_RUNS:
        bad.append(f"sweep repo private/gated or has {n} of {N_RUNS} runs")
    for ds in DATASETS:
        d = json.loads(get(f"https://huggingface.co/api/datasets/{USER}/{ds}")[1])
        if d.get("private") or d.get("gated"):
            bad.append(f"dataset {ds} private/gated")
    for c in COLLECTIONS:
        hits = json.loads(get(f"https://huggingface.co/api/collections?owner={USER}&limit=50")[1])
        if not any(h["slug"].startswith(f"{USER}/{c}") and not h.get("private") for h in hits):
            bad.append(f"collection {c} missing or private")
    for url in PAGES:
        status, body = get(url)
        if status != 200 or (url.endswith("index.html") and b"Relapse" not in body):
            bad.append(f"{url} -> {status}")
    print("\n".join(bad) if bad else f"all public: {len(MODELS)} models, the {N_RUNS}-run sweep, {len(DATASETS)} datasets, {len(COLLECTIONS)} collections, Space, repo")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
