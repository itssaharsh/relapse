"""Carries out the pushes listed in hf_updates/PLAN.md. Without --yes it only prints the plan.

    python scripts/prepare_hf_updates.py          # first: writes hf_updates/
    python scripts/apply_hf_updates.py            # prints the plan, changes nothing
    python scripts/apply_hf_updates.py --yes      # uploads cards and pages, adds the survey to the runs dataset, orders the collection

It uploads files into existing repositories and reorders one existing collection. It never creates, deletes or renames a
repository and never changes its visibility. `git push` is not done here.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "hf_updates")
USER = "itssaharsh"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--yes", action="store_true", help="carry out the plan"); a = ap.parse_args()
    print(open(os.path.join(OUT, "PLAN.md")).read())
    if not a.yes:
        print("Nothing was changed. Run again with --yes to carry out the plan.")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    msg = "Card matches the final paper: tokenizer-by-layer result, base model and tokenizer revision, paper link, tokenizer checker"
    for f in sorted(os.listdir(os.path.join(OUT, "cards"))):      # 2. cards
        kind, name = f[:-3].split("__")
        api.upload_file(path_or_fileobj=os.path.join(OUT, "cards", f), path_in_repo="README.md", repo_id=f"{USER}/{name}", repo_type=kind, commit_message=msg)
        print("card:", kind, name)
    for f in ("index.html", "app.js", "data/stage2.json", "og.png"):      # 3. the demo Space
        api.upload_file(path_or_fileobj=os.path.join(ROOT, "space", f), path_in_repo=f, repo_id=f"{USER}/relapse", repo_type="space", commit_message="Demo page matches the final paper")
    api.upload_file(path_or_fileobj=os.path.join(OUT, "space__relapse-article", "index.html"), path_in_repo="index.html", repo_id=f"{USER}/relapse-article", repo_type="space",
                    commit_message="Write-up rebuilt from docs/ARTICLE.md")      # 4. the write-up Space
    api.upload_folder(folder_path=os.path.join(ROOT, "results", "survey"), path_in_repo="stage2/survey", repo_id=f"{USER}/relapse-runs", repo_type="dataset",
                      commit_message="Survey of public copies, official history, job ledger")      # 5. the runs dataset
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "paper", "main.pdf"), path_in_repo="stage2/paper.pdf", repo_id=f"{USER}/relapse-runs", repo_type="dataset", commit_message="The final paper")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "scripts", "check_tokenizer.py"), path_in_repo="stage2/check_tokenizer.py", repo_id=f"{USER}/relapse-runs", repo_type="dataset", commit_message="The tokenizer checker")
    api.upload_folder(folder_path=os.path.join(ROOT, "results", "single_line"), path_in_repo="stage2/single_line", repo_id=f"{USER}/relapse-runs", repo_type="dataset", commit_message="Pre-registration 6: single-line rescoring")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "results", "PREREGISTRATION_6.md"), path_in_repo="stage2/PREREGISTRATION_6.md", repo_id=f"{USER}/relapse-runs", repo_type="dataset", commit_message="Pre-registration 6")
    if "stage2/PREREGISTRATION_5.md" not in api.list_repo_files(f"{USER}/relapse-runs", repo_type="dataset"):
        api.upload_file(path_or_fileobj=os.path.join(ROOT, "results", "PREREGISTRATION_5.md"), path_in_repo="stage2/PREREGISTRATION_5.md", repo_id=f"{USER}/relapse-runs", repo_type="dataset",
                        commit_message="Pre-registration 5")
    plan = json.load(open(os.path.join(OUT, "collection.json")))      # 6. the models collection, Stage 2 first
    col = next(c for c in api.list_collections(owner=USER) if plan["collection"] in c.slug)
    items = {it.item_id: it for it in api.get_collection(col.slug).items}
    for position, repo in enumerate(plan["order"]):
        if repo in items:
            api.update_collection_item(col.slug, items[repo].item_object_id, note=plan["notes"][repo], position=position)
    api.update_collection_metadata(col.slug, description=plan["descriptions"][plan["collection"]])
    data = next(c for c in api.list_collections(owner=USER) if plan["data_collection"] in c.slug)
    for it in api.get_collection(data.slug).items:
        if it.item_id in plan["data_notes"]:
            api.update_collection_item(data.slug, it.item_object_id, note=plan["data_notes"][it.item_id])
    api.update_collection_metadata(data.slug, description=plan["descriptions"][plan["data_collection"]])
    # 7. the Stage 1 repositories: Cohere's fixed tokenizer_config.json in place of the outdated one, on the date (UTC) their cards give
    import datetime
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_tokenizer import check
    from huggingface_hub import hf_hub_download
    from prepare_hf_updates import FIX_DATE, VENDOR_REPO, VENDOR_REV
    utc = datetime.datetime.now(datetime.timezone.utc)
    fixed = hf_hub_download(VENDOR_REPO, "tokenizer_config.json", revision=VENDOR_REV)
    for name in json.load(open(os.path.join(OUT, "stage1_revisions.json"))):
        if check(f"{USER}/{name}")["setting"] == "official":
            print("tokenizer already fixed:", name)
        elif f"{utc.day} {utc.strftime('%B %Y')}" != FIX_DATE:
            print(f"NOT DONE: the Stage 1 tokenizer fix of {name}. The cards say {FIX_DATE} (UTC); today is {utc:%d %B %Y}. Set FIX_DATE, prepare again, and rerun.")
        else:
            api.upload_file(path_or_fileobj=fixed, path_in_repo="tokenizer_config.json", repo_id=f"{USER}/{name}", commit_message="Tokenizer setting: Cohere's fix of 30 September 2026 (tokenizer_class PreTrainedTokenizerFast)")
            print("tokenizer fixed:", name, "->", check(f"{USER}/{name}")["setting"])
    print("done: cards, Spaces, dataset, collections and the Stage 1 tokenizer fix. Remaining: git push origin main.")


if __name__ == "__main__":
    main()
