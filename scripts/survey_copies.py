"""Surveys the public copies of the five Tiny Aya models: which of them still carry the outdated tokenizer setting today?

    python scripts/survey_copies.py            # writes results/survey/copies.csv, summary.json and two_copies.json
    python scripts/survey_copies.py --history  # writes results/survey/official_history.json: every commit of the five official repos
    python scripts/survey_copies.py --probe-sample  # loads 12 of the outdated copies with transformers and compares their split (probe_sample.json)
    python scripts/survey_copies.py --jobs     # writes results/survey/jobs.json: when each Stage 2 job started and from which commit (author's account)

Where it looks
  Hugging Face: every repo linked to a Tiny Aya model through `base_model` (fine-tunes, adapters, merges and quantized
  copies such as GGUF, AWQ, GPTQ, bitsandbytes or MLX), the repos linked to those in turn, and a name search.
  Kaggle Models: every variation of every model a search for Tiny Aya returns (skipped, and said so, if Kaggle cannot be reached).
What it reads
  Repo listings, and through scripts/check_tokenizer.py one tokenizer_config.json or the first megabytes of one GGUF header
  per repo. No weight file is downloaded. CPU only.
Columns of copies.csv
  repo, revision (commit, or Kaggle version), last_modified, downloads (Hugging Face: all time; Kaggle: not reported),
  format, setting (official, outdated, or unknown with the reason in `evidence`), then evidence, downloads_30d, base_model, found_by.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_tokenizer import KAGGLE, _get_json, check      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "survey")
BASES = [f"CohereLabs/tiny-aya-{v}" for v in ("base", "global", "fire", "water", "earth")]
RELATIONS = ("", "quantized:", "finetune:", "adapter:", "merge:")
NAME = re.compile(r"tiny[-_. ]?aya", re.I)
SEARCHES = ("tiny-aya", "tiny aya", "tiny_aya", "tinyaya")
EXPAND = ["sha", "lastModified", "downloads", "downloadsAllTime", "tags", "gated", "private", "library_name", "siblings", "disabled"]
TWO_COPIES = ("CohereLabs/tiny-aya-global", "kaggle:cohere-labs/tiny-aya/transformers/global/1")      # the two copies of Appendix E


def fmt(info, files: list[str]) -> str:
    """The format of a repo, from its files, tags and name."""
    tags = {t.lower() for t in (info.tags or [])}; name = info.id.lower(); lib = (info.library_name or "").lower()
    kinds = []
    if any(f.endswith(".gguf") for f in files) or "gguf" in tags:
        kinds.append("gguf")
    if "mlx" in tags or lib == "mlx" or "mlx" in name:
        kinds.append("mlx")
    if any(f.endswith(".onnx") for f in files) or "onnx" in tags:
        kinds.append("onnx")
    for key in ("awq", "gptq", "exl2", "openvino"):
        if key in name or key in tags:
            kinds.append(key)
    if "bitsandbytes" in tags or re.search(r"bnb|4bit|8bit", name):
        kinds.append("bitsandbytes")
    if "adapter_config.json" in files:
        kinds.append("adapter")
    if any(f.endswith((".safetensors", ".bin")) and "adapter" not in f for f in files) and not {"mlx", "awq", "gptq", "bitsandbytes", "adapter"} & set(kinds):
        kinds.append("transformers")
    return "+".join(kinds) or "other"


def _listing(api, **kw) -> list:
    """One listing from the Hub, waiting and retrying when the API rate limit is reached."""
    import time

    from huggingface_hub.errors import HfHubHTTPError
    for attempt in range(8):
        try:
            time.sleep(0.4)      # stay under 1,000 API requests per five minutes
            return list(api.list_models(limit=1000, expand=EXPAND, **kw))
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) != 429 or attempt == 7:
                raise
            time.sleep(int(e.response.headers.get("Retry-After") or 60) + 5)


def hugging_face() -> list[dict]:
    from huggingface_hub import HfApi
    api = HfApi()
    found: dict[str, tuple] = {}      # repo -> (listing entry, how it was found, the Tiny Aya model it derives from)

    def children(parent: str, root: str, how: str, relations=RELATIONS) -> list[str]:
        new = []
        for rel in relations:
            for m in _listing(api, filter=f"base_model:{rel}{parent}"):
                if m.id not in found and m.id != parent:
                    found[m.id] = (m, how + (f" ({rel[:-1]})" if rel else ""), root); new.append(m.id)
        return new

    for b in BASES:      # the official repos themselves
        found[b] = (next(m for m in _listing(api, search=b.split("/")[1], author="CohereLabs") if m.id == b), "official repo", b)
    level = [(c, b) for b in BASES for c in children(b, b, "base_model")]
    for how in ("base_model of a derived repo", "base_model of a derived repo"):      # two further levels, e.g. the GGUF copy of a fine-tune
        level = [(c, root) for parent, root in level for c in children(parent, root, how, relations=("",))]
    for q in SEARCHES:
        for m in _listing(api, search=q):
            if NAME.search(m.id) and m.id not in found:
                found[m.id] = (m, "name search", "")
    rows = []
    for repo in sorted(found, key=str.lower):
        info, how, root = found[repo]
        files = [s.rfilename for s in (info.siblings or [])]
        if not root:
            root = next((t.split(":")[-1] for t in (info.tags or []) if t.startswith("base_model:") and NAME.search(t)), "")
        r = check(f"{repo}@{info.sha}", files=files)
        setting, evidence = r["setting"], r["evidence"]
        ggufs = sorted((f for f in files if f.endswith(".gguf")), key=lambda f: ("mmproj" in f, len(f), f))
        if r["kind"] == "hf" and ggufs:      # a repo with both transformers files and GGUF files: read one GGUF header too
            g = check(f"{repo}@{info.sha}:{ggufs[0]}", files=files)
            evidence += f"; {g['evidence']}"
            setting = "outdated" if "outdated" in (setting, g["setting"]) else setting if g["setting"] == "official" else "unknown"
        if len(ggufs) > 1:
            evidence += f"; 1 of {len(ggufs)} GGUF files read"
        if setting == "unknown" and info.gated and "HTTP 40" in evidence:
            evidence = "gated: tokenizer files not readable without accepting the repo's terms"
        rows.append({"repo": repo, "revision": info.sha or "", "last_modified": info.last_modified.strftime("%Y-%m-%d") if info.last_modified else "",
                     "downloads": info.downloads_all_time if info.downloads_all_time is not None else "", "format": fmt(info, files), "setting": setting, "evidence": evidence,
                     "downloads_30d": info.downloads if info.downloads is not None else "", "base_model": root, "found_by": how})
    return rows


def kaggle() -> tuple[list[dict], str]:
    rows, refs = [], set()
    try:
        for q in SEARCHES:
            res = _get_json(f"{KAGGLE}/models/list?search={q.replace(' ', '%20')}&pageSize=100")
            refs |= {m["ref"] for m in (res.get("models", []) if isinstance(res, dict) else res) if NAME.search(m.get("ref", "") + " " + m.get("title", ""))}
        for ref in sorted(refs):
            model = _get_json(f"{KAGGLE}/models/{ref}/get")
            for inst in model.get("instances", []):
                handle = f"{ref}/{inst['framework']}/{inst['slug']}/{inst['versionNumber']}"
                r = check("kaggle:" + handle)
                rows.append({"repo": "kaggle:" + handle, "revision": f"version {inst['versionNumber']}", "last_modified": (model.get("updateTime") or "")[:10], "downloads": "",
                             "format": inst["framework"], "setting": r["setting"], "evidence": r["evidence"], "downloads_30d": "", "base_model": f"CohereLabs/tiny-aya-{inst['slug']}", "found_by": "Kaggle Models search"})
        return rows, f"searched Kaggle Models: {len(refs)} model(s), {len(rows)} variation(s)"
    except Exception as e:
        return [], f"Kaggle skipped: {type(e).__name__}: {e}"


def history() -> dict:
    """The public commit history of the five official repos, with the content ids of the tokenizer files, config and weights at each commit.

    The Stage 2 jobs downloaded the main branch and did not record its revision; this history says what main was when they ran."""
    from huggingface_hub import HfApi
    api = HfApi()
    track = ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "config.json", "generation_config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    out = {}
    for repo in BASES:
        commits = []
        for c in api.list_repo_commits(repo):
            info = api.model_info(repo, revision=c.commit_id, files_metadata=True)
            ids = {s.rfilename: (s.lfs.sha256 if s.lfs else s.blob_id) for s in info.siblings if s.rfilename in track}
            commits.append({"sha": c.commit_id, "date": c.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"), "title": c.title, "files": ids, "setting": check(f"{repo}@{c.commit_id}")["setting"]})
        out[repo] = commits
    return out


def jobs() -> list[dict]:
    """The Stage 2 jobs as Hugging Face Jobs recorded them: start time, hardware, outcome and the repository commit each job ran from.

    Needs the token of the account that ran the jobs. Only setting names and four values are kept (no configuration bodies, no secrets)."""
    from huggingface_hub import HfApi
    out = []
    for j in sorted(HfApi().list_jobs(), key=lambda j: j.created_at):
        env = j.environment or {}
        out.append({"started": j.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"), "flavor": str(j.flavor), "outcome": str(getattr(j.status, "stage", "")), "repo_commit": env.get("REPO_COMMIT", ""),
                    "experiment": env.get("EXPERIMENT", ""), "variant": env.get("VARIANT", ""), "seed": env.get("SEED", ""), "out": env.get("OUT", ""), "run_suffix": env.get("RUN_SUFFIX", ""), "batches": env.get("NB", ""), "settings": sorted(env)})
    return out


def probe_sample(n_top: int = 5, n_random: int = 7) -> dict:
    """Loads a sample of the copies found outdated (the most downloaded and a random draw) with the installed transformers and
    compares how each splits a neutral multiple-choice prompt with the official tokenizer. Needs transformers and tokenizers."""
    import random
    import tempfile

    import tokenizers
    import transformers

    from check_tokenizer import HF, PROBE, _get
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "copies.csv"))) if r["setting"] == "outdated" and "tokenizer_class=Cohere" in r["evidence"] and not r["repo"].startswith(("itssaharsh/", "kaggle:"))]
    rows.sort(key=lambda r: -int(r["downloads"] or 0))
    rest = rows[n_top:]; random.Random(0).shuffle(rest)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "tokenizer.json"), "wb").write(_get(f"{HF}/{BASES[1]}/resolve/main/tokenizer.json"))
        official = tokenizers.Tokenizer.from_file(os.path.join(d, "tokenizer.json")).encode(PROBE, add_special_tokens=False).ids
    out = []
    for r in rows[:n_top] + rest:
        if len(out) == n_top + n_random:
            break
        with tempfile.TemporaryDirectory() as d:
            try:
                for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                    open(os.path.join(d, f), "wb").write(_get(f"{HF}/{r['repo']}/resolve/{r['revision']}/{f}"))
                tok = transformers.AutoTokenizer.from_pretrained(d)
                ids = tok(PROBE, add_special_tokens=False)["input_ids"]
            except Exception as e:      # a repo without all three files, or one that does not load
                continue
        out.append({"repo": r["repo"], "revision": r["revision"], "tokenizer_class": r["evidence"].split(";")[0].split("=")[1], "class_loaded": type(tok).__name__,
                    "same_split_as_official": ids == official, "tokens_official": len(official), "tokens_loaded": len(ids)})
    return {"transformers": transformers.__version__, "sampled": len(out), "split_differently": sum(not x["same_split_as_official"] for x in out), "copies": out}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    if "--probe-sample" in sys.argv:
        json.dump(probe_sample(), open(os.path.join(OUT, "probe_sample.json"), "w"), indent=1)
        return
    if "--jobs" in sys.argv:
        json.dump(jobs(), open(os.path.join(OUT, "jobs.json"), "w"), indent=1)
        return
    if "--history" in sys.argv:
        json.dump(history(), open(os.path.join(OUT, "official_history.json"), "w"), indent=1)
        return
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    hf = hugging_face()
    kg, kaggle_note = kaggle()
    rows = hf + kg
    cols = ["repo", "revision", "last_modified", "downloads", "format", "setting", "evidence", "downloads_30d", "base_model", "found_by"]
    with open(os.path.join(OUT, "copies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    num = lambda x: int(x) if str(x).isdigit() else 0
    group = lambda rs: {"copies": len(rs), "downloads": sum(num(r["downloads"]) for r in rs), "downloads_30d": sum(num(r["downloads_30d"]) for r in rs)}
    mine = lambda r: r["repo"].startswith("itssaharsh/")
    out = lambda rs: [r for r in rs if r["setting"] == "outdated"]
    summary = {"date": today, "kaggle": kaggle_note, "hugging_face": {"repos": len(hf), **{s: group([r for r in hf if r["setting"] == s]) for s in ("official", "outdated", "unknown")}},
               "hugging_face_outdated_by_format": {k: group([r for r in out(hf) if r["format"] == k]) for k in sorted({r["format"] for r in out(hf)})},
               "hugging_face_outdated_not_ours": group([r for r in out(hf) if not mine(r)]), "hugging_face_outdated_ours": group([r for r in out(hf) if mine(r)]),
               "hugging_face_outdated_transformers": group([r for r in out(hf) if "tokenizer_class" in r["evidence"] and "tokenizer_class=Cohere" in r["evidence"]]),
               "hugging_face_outdated_gguf": group([r for r in out(hf) if "tokenizer.ggml.pre" in r["evidence"] and "tokenizer_class=Cohere" not in r["evidence"]]),
               "official_repos": {r["repo"]: r["setting"] for r in hf if r["repo"] in BASES},
               "kaggle_models": {"copies": len(kg), "outdated": len(out(kg)), "official": sum(r["setting"] == "official" for r in kg)},
               "unknown_reasons": sorted({r["evidence"] for r in rows if r["setting"] == "unknown"})}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    two = [check(t) for t in TWO_COPIES]
    json.dump({"date": today, "copies": two}, open(os.path.join(OUT, "two_copies.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
