"""Publish Relapse artifacts to Hugging Face: unlearned checkpoints (+ GGUF), raw runs, probe sets, collections.

    python scripts/publish_hf.py --models models.json --runs results/raw --probes <probes dir>

models.json: [{"repo": "tiny-aya-global-relapse-circuitkit-rmu", "dir": "<patched checkpoint dir>", "gguf": "<file>.gguf",
               "card": "<model card markdown file>"}]
Everything is created public and ungated (hackathon rule). Run it only after reviewing what will be uploaded.
"""
from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import HfApi, add_collection_item, create_collection

USER = os.environ.get("HF_USER", "itssaharsh")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--runs", required=True, help="folder uploaded as dataset <user>/relapse-runs")
    ap.add_argument("--probes", required=True, help="folder uploaded as dataset <user>/relapse-wmdp-bio-probes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    plan = json.load(open(a.models))
    ds_runs, ds_probes = f"{USER}/relapse-runs", f"{USER}/relapse-wmdp-bio-probes"
    if a.dry_run:
        print("would publish:", [m["repo"] for m in plan], ds_runs, ds_probes)
        return
    model_ids = []
    for m in plan:
        rid = f"{USER}/{m['repo']}"
        api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=m["dir"], commit_message="Relapse: unlearned checkpoint",
                          ignore_patterns=[".git*", "*.gguf", "README.md"])
        if m.get("gguf"):
            api.upload_file(path_or_fileobj=m["gguf"], path_in_repo=os.path.basename(m["gguf"]), repo_id=rid)
        api.upload_file(path_or_fileobj=m["card"], path_in_repo="README.md", repo_id=rid)
        model_ids.append(rid)
        print("model", rid)
    for rid, folder in ((ds_runs, a.runs), (ds_probes, a.probes)):
        api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=folder, commit_message="Relapse data")
        print("dataset", rid)
    cm = create_collection(title="Relapse: unlearned Tiny Aya models", namespace=USER, private=False, exists_ok=True,
                           description="Tiny Aya Global with hazardous biology unlearned (Lexsi SafeTune RMU), checked in every file format.")
    for rid in model_ids:
        add_collection_item(cm.slug, item_id=rid, item_type="model", exists_ok=True)
    cd = create_collection(title="Relapse: runs, logs and probe sets", namespace=USER, private=False, exists_ok=True,
                           description="Per-question results, AuditKIT runs, CircuitKIT scores and CuratorKIT-gated probe sets.")
    for rid in (ds_runs, ds_probes):
        add_collection_item(cd.slug, item_id=rid, item_type="dataset", exists_ok=True)
    print("collections", cm.slug, cd.slug)


if __name__ == "__main__":
    main()
