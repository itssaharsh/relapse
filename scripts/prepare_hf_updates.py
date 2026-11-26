"""Prepares, as local files, every change to the Hugging Face repositories of the Relapse collections. Uploads nothing.

    python scripts/prepare_hf_updates.py          # writes hf_updates/ (cards, the built write-up page, the collection order, PLAN.md)
    python scripts/apply_hf_updates.py --yes      # the only step that changes anything on Hugging Face; run it after reading PLAN.md

What it writes
  hf_updates/cards/<type>__<name>.md      the new README.md of each repository
  hf_updates/previous/<type>__<name>.md   the README.md that is live today, as downloaded, so nothing is lost
  hf_updates/space__relapse-article/      the write-up Space's page, rebuilt from docs/ARTICLE.md
  hf_updates/collection.json              the planned order and notes of the models collection, Stage 2 models first
  hf_updates/PLAN.md                      the list of pushes, in order
It never creates, deletes or renames a repository and never changes its visibility; neither does the apply script.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_readme  # noqa: E402
import publish_cards  # noqa: E402
from check_tokenizer import check  # noqa: E402

OUT = os.path.join(ROOT, "hf_updates")
USER = "itssaharsh"
REPO = "https://github.com/itssaharsh/relapse"
PAPER, PAPER_URL, CHECKER_URL = publish_cards.PAPER, publish_cards.PAPER_URL, publish_cards.CHECKER_URL
VARIANTS = ("global", "fire", "earth", "water", "base")
STAGE2 = [f"relapse-tiny-aya-{v}-unlearn-bio" for v in VARIANTS]
STAGE1 = {"tiny-aya-global-unlearn-bio-default-layers": "layers 5 to 7, SafeTune's default",
          "tiny-aya-global-unlearn-bio-circuitkit-layers": "layers 7 to 9, the window CircuitKIT ranked first in Stage 1",
          "tiny-aya-global-unlearn-bio-circuitkit-4bit-aware": "layers 7 to 9, trained through a 4-bit quantizer",
          "tiny-aya-global-unlearn-bio-deeper-layers": "layers 10 to 12",
          "tiny-aya-global-unlearn-bio-deeper-layers-file-trained": "layers 10 to 12, trained with the model seen as its Q4_0 file",
          "tiny-aya-global-unlearn-bio-sweep": "the edited matrices of all 38 Stage 1 runs, at five layer windows"}
FIX_DATE = "26 November 2026"      # the day (UTC, as Hugging Face records the commits) the Stage 1 repositories got Cohere's fixed tokenizer_config.json; results/survey/reports.json has the commits
VENDOR_REPO, VENDOR_REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"      # Cohere's fix of 30 September 2026
DATASETS = ("relapse-runs", "relapse-wmdp-bio-probes")
SPACES = ("relapse-article", "relapse")
load = lambda *p: json.load(open(os.path.join(ROOT, "results", *p)))


def finding() -> str:
    """What the final paper found, in one paragraph, from the result files."""
    s2, m = load("shift2", "summary.json"), load("map", "summary.json")
    tl = build_readme.tally(); dm = build_readme.damage(); eff = [r for r in m["runs"] if r["effective"]]
    held = sum(not (r["Q4_0"]["relapse"] or r["nf4"]["relapse"]) for r in eff)
    return (f"**What the final paper found** ([{PAPER}]({PAPER_URL})). Runs unlearned and audited through an outdated tokenizer setting at SafeTune's default layers (5 to 7) regained "
            f"{100 * s2['by_window']['5-7']['mean_rise']:.1f} points on average once loaded through the official tokenizer, in {s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]} runs that count; "
            f"those runs had also lost {dm['stwoDefaultCostAudited']} MMLU points where they were audited and {dm['stwoDefaultCostDeployed']} once deployed, and with single-line prompts none of them relapsed. "
            f"At layers 3 to 5, {s2['T2']['relapses']} of {s2['T2']['effective']} did, on the four post-trained variants; Tiny Aya Base rose at every window. In 4-bit files the forgetting held in "
            f"{held} of {len(eff)} runs. Of {tl['n']} registered hypotheses, {tl['held']} held, {tl['part']} held in part, {tl['failed']} failed and {tl['untestable']} could not be tested.")


def split(text: str) -> tuple[str, str]:
    """The YAML front matter of a card (with its --- lines) and the rest."""
    m = re.match(r"(---\n.*?\n---\n)(.*)", text, re.S)
    return (m.group(1), m.group(2)) if m else ("", text)


def stage1_revisions(api) -> dict:
    """The revision of each Stage 1 repository that carries the outdated setting: its head before the tokenizer fix. Kept once written."""
    path = os.path.join(OUT, "stage1_revisions.json")
    if os.path.exists(path):
        return json.load(open(path))
    revs = {n: api.model_info(f"{USER}/{n}").sha for n in STAGE1 if check(f"{USER}/{n}")["setting"] == "outdated"}
    json.dump(revs, open(path, "w"), indent=1)
    return revs


def stage1_card(name: str, what: str, old: str, files: list[str], today: str, revs: dict) -> str:
    front, body = split(old)
    title = next((l for l in body.split("\n") if l.startswith("# ")), f"# {name}")
    table = [l for l in body.split("\n") if l.startswith("| fp16:") or l.startswith("|---|---|---|---|")][:2]      # the scores through the official tokenizer, already on the card
    tok = (f"Until {FIX_DATE} (UTC) its `tokenizer_config.json` carried that setting (`tokenizer_class: CohereTokenizerFast`); on that date it was replaced with Cohere's fixed file (from `{VENDOR_REPO}` at `{VENDOR_REV[:10]}`), so the repository now loads with the official tokenizer. "
           f"The Stage 1 results were made with revision `{revs[name][:10]}` of this repository, which keeps the outdated setting."
           if name in revs else "It holds edited matrices only, no tokenizer files.")
    stage2 = ", ".join(f"[{v.capitalize()}](https://huggingface.co/{USER}/relapse-tiny-aya-{v}-unlearn-bio)" for v in VARIANTS)
    rescored = ("\nRescored through the official tokenizer on 21 November 2026 (WMDP-bio / MMLU; the original scores 63.3% / 55.4% at fp16):\n\n" + "\n".join(table) + "\n") if len(table) == 2 else ""
    return f"""{front}
{title}

> **Superseded Stage 1 artifact (status on {today}).** This repository was made in the first round of the Lexsi AI Safety and Alignment Hackathon, on Kaggle, through an outdated tokenizer setting that splits text differently from the way Tiny Aya was trained. Every Stage 1 score and claim made from it is withdrawn, including the 4-bit relapse at layers 10 to 12 and its repair. {tok} Use the Stage 2 models instead: {stage2}.
{rescored}
{finding()}

## What this repository holds

- Tiny Aya Global after SafeTune RMU unlearning of hazardous biology: {what}. Trained and first scored through the outdated tokenizer setting.
- Files: {", ".join(f"`{f}`" for f in files if not f.startswith(".") and f != "README.md")}.

## Base model and tokenizer revision

Base model: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). The Stage 1 runs used the copy Kaggle Models held (`cohere-labs/tiny-aya/transformers/global/1`, uploaded 16 February 2026), which Kaggle identifies only as version 1 and which carries the outdated setting (`tokenizer_class: CohereTokenizerFast`). Cohere fixed the setting on Hugging Face on 30 September 2026 (revision `{build_readme.revisions().split('`')[1]}` of the official repository).

The checker [`scripts/check_tokenizer.py`]({CHECKER_URL}), added with the paper, reports which setting a repository, folder or GGUF file carries and what to do about it, reading only its tokenizer file or GGUF header: `python scripts/check_tokenizer.py {USER}/{name}`.

The Stage 1 card of this repository, with its withdrawn description and scores, is kept at [`hf_updates/previous/`]({REPO}/tree/main/hf_updates/previous), and the Stage 1 write-up at [`docs/archive/`]({REPO}/tree/main/docs/archive).

## License

CC-BY-NC-4.0, the license of the original model. Use is also subject to Cohere's terms for Tiny Aya.
"""


WORK_FOLDERS = [      # folder, what it holds, what reads it
    ("map/", "Pre-registration 1, the relapse map: per run `results.json`, `items.json` (one 0/1 per question) and the edited matrices `changed_map.pt`.",
     "`scripts/analyze_map.py`; the matrices by the scoring jobs built on `hf_jobs/header_real.py` (real files, locate) through `PT_REPO` and `PT_PATH`."),
    ("deep/", "Pre-registration 4, the Base relapse follow-up: reruns, fresh seeds 41 to 48, the repair arms and their controls, with `changed_map.pt`.",
     "`scripts/analyze_deep.py`; the matrices by the real-file, locate, bitsandbytes and runtime jobs."),
    ("shift/", "Pre-registration 2, the first tokenizer experiment: every run scored through the official and the outdated setting.", "`scripts/analyze_shift.py` (from the copies in the code repository's `results/shift/`)."),
    ("shift2/", "Pre-registration 5, placement against the tokenizer: layers 3 to 5, 5 to 7 and 7 to 9, with the edited matrices `changed_shift2.pt`.",
     "The job of pre-registration 6 (`hf_jobs/header_single.py`, at a pinned revision of this dataset); `scripts/analyze_shift2.py` from `results/shift2/`."),
    ("single_line/", "Pre-registration 6: the saved runs of pre-registration 5 scored with single-line prompts through both settings.", "`scripts/analyze_single_line.py` from `results/single_line/`."),
    ("place/", "Pre-registration 3, placement: layers 3 to 5, each model's CircuitKIT window, 7 to 9 and the default 5 to 7 on seeds 31 and 32.", "`scripts/analyze_place.py` from `results/place/`."),
    ("final/", "The final-model runs of pre-registration 3, amendment 1 (seeds 33 and 34), with `changed_map.pt`.", "`scripts/analyze_final.py`; the full check, real-file and publishing jobs."),
    ("real/", "Outputs of the scoring jobs: real llama.cpp files, locate (one part rounded at a time), the full check, the bitsandbytes loader, the runtime check, publishing, the tokenizer control and the Stage 1 models rescored. "
              "Includes AuditKIT run records (`audit_*.json`).", "`scripts/analyze_deep.py`, `scripts/analyze_certify.py`, `scripts/rescore_stats.py`."),
    ("circuitkit/", "CircuitKIT EAP-IG attribution on the five models, 64 prompts: layer importance, node scores, faithfulness report, tokenizer check.", "The placement job, to pick each model's window; `scripts/paper_numbers.py` from `results/place/circuitkit/`."),
    ("circuitkit_more/", "The same with 160 prompts (exploratory).", "`scripts/paper_numbers.py` from `results/place/circuitkit_more/`."),
    ("cons/", "The consolidation pilot (exploratory).", "Nothing registered."),
    ("npo_pilot/", "The NPO with LoRA pilot (exploratory), with `npo_deltas.pt`.", "`scripts/paper_numbers.py` counts its runs."),
    ("smoke/", "A three-batch pipeline test.", "Nothing."),
    ("stage1/", "Six files of the superseded Stage 1 that two job headers read: the edited matrices of five Stage 1 runs (`dl/*/changed_*.pt`) and a `llama-quantize` built from llama.cpp on Kaggle. "
                "Copied here on 26 November 2026 (UTC) from a private backup, with unchanged content hashes, so that no job reads a private repository.",
     "`hf_jobs/tok_head.py` (the tokenizer control) and `hf_jobs/header_real.py` (the Stage 1 models rescored; real-file jobs run without `BUILD_QUANTIZE=1`)."),
]


def working_card(api, today: str) -> str:
    """The card of itssaharsh/relapse-stage2, the dataset the Stage 2 jobs wrote into: what each folder holds, what reads it, and its size."""
    tree = [x for x in api.list_repo_tree(f"{USER}/relapse-stage2", repo_type="dataset", recursive=True) if hasattr(x, "size") and "/" in x.path]
    size = lambda d, pt: sum(x.size for x in tree if x.path.startswith(d) and x.path.endswith(".pt") == pt)
    count = lambda d: sum(x.path.startswith(d) for x in tree)
    assert {x.path.split("/")[0] + "/" for x in tree} == {d for d, _, _ in WORK_FOLDERS}, "a folder of relapse-stage2 is missing from WORK_FOLDERS"
    rows = "\n".join(f"| `{d}` | {what} | {reads} | {count(d)} | {size(d, True) / 1e9:.1f} GB | {size(d, False) / 1e6:.1f} MB |" for d, what, reads in WORK_FOLDERS)
    total_pt, total_js = sum(size(d, True) for d, _, _ in WORK_FOLDERS), sum(size(d, False) for d, _, _ in WORK_FOLDERS)
    return f"""---
license: cc-by-nc-4.0
tags: [unlearning, quantization, wmdp, mmlu, safetune, circuitkit, auditkit, tiny-aya, relapse]
pretty_name: Relapse Stage 2 working files (edited matrices and raw job outputs)
---

# Relapse: Stage 2 working files

This is the dataset the Stage 2 jobs of the paper wrote into as they ran: the edited weight matrices of every run and the raw outputs of every job. It is kept so that a scoring job can be rerun without retraining. For reading results, use [`{USER}/relapse-runs`](https://huggingface.co/datasets/{USER}/relapse-runs), which holds the same per-question results in a tidier layout, or the `results/` folder of the [code repository]({REPO}).

**Paper:** [{publish_cards.PAPER}]({REPO}/blob/main/paper/main.pdf). **Code, jobs and pre-registrations:** [{REPO.replace("https://", "")}]({REPO}). Status on {today} (UTC): made public on 26 November 2026 (UTC); unchanged since the jobs wrote it, apart from this card and the `stage1/` folder.

## What is in it

{len(tree)} files: {total_pt / 1e9:.1f} GB of edited matrices in {sum(x.path.endswith(".pt") for x in tree)} `.pt` files and {total_js / 1e6:.0f} MB of other files (JSON, and one binary in `stage1/`).

| Folder | What it holds | Read by | Files | Matrices | Other files |
|---|---|---|---|---|---|
{rows}

A `.pt` file is a PyTorch dictionary: run name to that run's edited `mlp.down_proj` matrices, by parameter name, which the jobs write into Cohere's original model before scoring. Run names give the model, the window, the learning rate, the tokenizer setting where it applies, and the seed, for example `global_W3-5_lr3e-05_outdated_s51`. The jobs are in [`hf_jobs/`]({REPO}/tree/main/hf_jobs) and [`kaggle/`]({REPO}/tree/main/kaggle); each pre-registration names the settings of its jobs, and `results/survey/jobs.json` in the code repository gives the start time and commit of every job.

## What the text files contain

Scores, per-question 0/1 lists, answer letters and probabilities, attribution scores and job settings. The AuditKIT run records in `real/` also hold the prompt of every question they scored: the public WMDP-bio test questions of [`cais/wmdp`](https://huggingface.co/datasets/cais/wmdp) and the sampled questions of [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), both MIT-licensed, with the model's answer letter. Nothing here contains text from the gated forget corpus or free-form model output. Paths such as `/tmp/kg/input/...` in these records are directories inside the job's container.

## License and access

CC-BY-NC-4.0, the license of the original Tiny Aya models, from which the edited matrices are derived. Use is also subject to Cohere's terms for Tiny Aya. The benchmark questions quoted in the run records keep their own MIT license, and so does `stage1/dl/final_a/bin/llama-quantize`, a build of llama.cpp. Not gated, like the released models. The matrices are not models on their own: loading them needs the original weights from [`CohereLabs/tiny-aya-<variant>`](https://huggingface.co/CohereLabs/tiny-aya-global), which Cohere gates.

These are research artifacts, not safety guarantees: the paper shows that a short fine-tune restores the unlearned knowledge.
"""


def status_block(kind: str, today: str) -> str:
    extra = {"relapse-runs": "The `stage2/` folders hold the runs the paper reports, from six pre-registered experiments (the fifth, placement against the tokenizer, is `stage2/shift2/`; the sixth, single-line prompts, `stage2/single_line/`), and `stage2/survey/` holds the survey of public copies, the official repositories' history and the start time and commit of every job. Everything outside `stage2/` and `correct_tokenizer/` is Stage 1, made through the outdated tokenizer setting, and is superseded.",
             "relapse-wmdp-bio-probes": "The translations and rewordings were made in Stage 1 by Tiny Aya Global through Kaggle's copy, so through the outdated tokenizer setting; no native speaker checked them. The paper's full check scores the released models on them through the official tokenizer.",
             "relapse-article": "This page is the write-up of the final results, rebuilt from `docs/ARTICLE.md`.",
             "relapse": "This page shows the final, pre-registered Stage 2 results; every number on it is read from the result files."}[kind]
    return (f"<!-- relapse-status:start -->\n> **Status on {today}.** {extra}\n>\n> {finding()}\n>\n> **Base model and tokenizer revision.** Every Stage 2 job downloaded the main branch of `CohereLabs/tiny-aya-<variant>` when it started; "
            f"by the job start times and the repositories' history that was the revision carrying Cohere's tokenizer fix ({build_readme.revisions()}, all committed 30 September 2026). Stage 1 used the copy Kaggle Models held, which carries the outdated setting.\n>\n"
            f"> **Tokenizer checker.** [`scripts/check_tokenizer.py`]({CHECKER_URL}), added with the paper, tells you whether a copy of Tiny Aya carries the outdated tokenizer setting and what to do about it.\n<!-- relapse-status:end -->\n")


def with_status(old: str, kind: str, today: str) -> str:
    front, body = split(old)
    body = re.sub(r"<!-- relapse-status:start -->.*?<!-- relapse-status:end -->\n\n?", "", body, flags=re.S)
    lines = body.lstrip("\n").split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("# ")), -1)      # the block goes right under the title, or first if there is none
    block = status_block(kind, today)
    new = "\n".join(lines[:i + 1]) + ("\n\n" if i >= 0 else "") + block + "\n" + "\n".join(lines[i + 1:]).lstrip("\n")
    return front + "\n" + new


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%-d %B %Y")      # UTC, as every date in the cards and the paper
    for d in ("cards", "previous", "space__relapse-article"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    live = {}
    targets = [("model", n) for n in STAGE2 + list(STAGE1)] + [("dataset", n) for n in DATASETS] + [("space", n) for n in SPACES]
    for kind, name in targets:
        path = hf_hub_download(f"{USER}/{name}", "README.md", repo_type=kind)
        live[(kind, name)] = open(path).read()
        open(os.path.join(OUT, "previous", f"{kind}__{name}.md"), "w").write(live[(kind, name)])
    cards = {}
    # Stage 2 models: the generated card
    chk = load("final", "check.json"); ctx = publish_cards.context()
    for v in VARIANTS:
        runs = {n: e for n, e in chk.items() if e["model"] == v}
        name = next(n for n in runs if n.endswith("_s34")); other = next(n for n in runs if n.endswith("_s33"))
        cards[("model", f"relapse-tiny-aya-{v}-unlearn-bio")] = publish_cards.card(v, name, runs[name], {"name": other, **runs[other]}, ctx)
    # Stage 1 models: a short corrected card in place of the Stage 1 text
    revs = stage1_revisions(api)
    for name, what in STAGE1.items():
        files = sorted(api.list_repo_files(f"{USER}/{name}"))
        files = files if len(files) <= 12 else files[:10] + [f"and {len(files) - 10} more"]
        cards[("model", name)] = stage1_card(name, what, live[("model", name)], files, today, revs)
    # datasets and Spaces: the live card with a status block under its title
    for kind, name in [("dataset", n) for n in DATASETS] + [("space", "relapse-article")]:
        cards[(kind, name)] = with_status(live[(kind, name)], name, today)
    cards[("dataset", "relapse-runs")] = cards[("dataset", "relapse-runs")].replace("Four pre-registered experiments", "Six pre-registered experiments").replace("scripts/build_results.py", "scripts/archive/stage1/build_results.py").replace(
        "Six pre-registered experiments, trained and scored through the **official tokenizer** on all five Tiny Aya models.",
        "Six pre-registered experiments on all five Tiny Aya models, scored through the **official tokenizer**; the two tokenizer experiments also unlearn and score through the outdated setting, on purpose.").replace(
        "| `stage2/shift/` | The tokenizer experiment: each run scored through the official and the outdated tokenizer setting |",
        "| `stage2/shift/` | The tokenizer experiment: each run scored through the official and the outdated tokenizer setting |\n| `stage2/shift2/` | Placement against the tokenizer (pre-registration 5): layers 3 to 5, 5 to 7 and 7 to 9, unlearned through the outdated setting, scored through both |\n| `stage2/single_line/` | Pre-registration 6: the same saved runs scored with single-line prompts through both settings |\n| `stage2/survey/` | The survey of public copies, the official repositories' history, the start time and commit of every job, token-level differences; `stage2/check_tokenizer.py` is the checker |")
    cards[("space", "relapse")] = with_status(open(os.path.join(ROOT, "space", "README.md")).read(), "relapse", today)      # the demo Space's card is kept in space/README.md
    cards[("dataset", "relapse-stage2")] = working_card(api, today)      # the working dataset the jobs wrote into
    try:
        live[("dataset", "relapse-stage2")] = open(api.hf_hub_download(f"{USER}/relapse-stage2", "README.md", repo_type="dataset")).read()
    except Exception:
        live[("dataset", "relapse-stage2")] = ""
    for (kind, name), text in cards.items():
        open(os.path.join(OUT, "cards", f"{kind}__{name}.md"), "w").write(text)
    # the write-up Space: its page template with the article body swapped in
    page_dir = os.path.join(OUT, "space__relapse-article")
    for f in ("index.html", "style.css"):
        shutil.copy(hf_hub_download(f"{USER}/relapse-article", f, repo_type="space"), os.path.join(page_dir, f))
    import build_article_space
    build_article_space.main(page_dir)
    # the models collection: Stage 2 first
    order = [f"{USER}/{n}" for n in STAGE2 + list(STAGE1)]
    notes = {f"{USER}/{n}": "Stage 2: unlearned and checked through the official tokenizer. Layers 3 to 5, with a real Q4_K_M file and the scores of both files." for n in STAGE2}
    notes |= {f"{USER}/{n}": "Stage 1, superseded: made through an outdated tokenizer setting; every Stage 1 score is withdrawn (see its card)." for n in STAGE1}
    sv = build_readme.survey()
    descriptions = {"relapse-unlearned-tiny-aya-models": "Tiny Aya with hazardous biology unlearned (RMU). Stage 2 models first. Paper and tokenizer checker: github.com/itssaharsh/relapse",
                    "relapse-runs-logs-and-probe-sets": "Every run of the paper, per question; six pre-registrations; a survey of public copies; probe sets. Paper: github.com/itssaharsh/relapse"}
    assert all(len(d) <= 150 for d in descriptions.values()), "a collection description is limited to 150 characters"
    data_notes = {f"{USER}/relapse-runs": "Every run of the paper, per question; pre-registrations; the survey of public copies (stage2/survey).",
                  f"{USER}/relapse-wmdp-bio-probes": "707 reworded and translated WMDP-bio questions used in the full check.",
                  f"{USER}/relapse-article": "The write-up: final results, every registered hypothesis and its outcome.",
                  f"{USER}/relapse": "Demo page: the final, pre-registered Stage 2 results, drawn from the result files."}
    # what the submission form asks of the two collections: every model and dataset used, not only the ones made here
    used_models = {f"CohereLabs/tiny-aya-{v}": "Used, not ours: Cohere Labs' original model, the starting point of every run and the reference each score is compared with. Gated by Cohere." for v in VARIANTS}
    used_data = {"cais/wmdp": "Used, not ours: the WMDP benchmark. Every run is scored on its 1,273 WMDP-bio questions.",
                 "cais/wmdp-bio-forget-corpus": "Used, not ours: the texts RMU forgets. Gated; downloaded inside each job and never redistributed.",
                 "Salesforce/wikitext": "Used, not ours: the texts RMU retains (wikitext-2-raw-v1).",
                 "cais/mmlu": "Used, not ours: general knowledge (a fixed sample of 1,000 questions) and the four benign-biology subjects."}
    json.dump({"collection": "relapse-unlearned-tiny-aya-models", "order": order, "notes": notes, "descriptions": descriptions, "data_collection": "relapse-runs-logs-and-probe-sets", "data_notes": data_notes,
               "add_models": used_models, "add_datasets": used_data,
               "add_own_datasets": {f"{USER}/relapse-stage2": "Working files of the Stage 2 jobs: the edited matrices of every run (87 GB) and raw job outputs, so a scoring job can be rerun without retraining."}},
              open(os.path.join(OUT, "collection.json"), "w"), indent=1)
    changed = [f"{k}__{n}" for (k, n), t in cards.items() if t != live.get((k, n))]
    plan = [f"# Pushes prepared on {today}, waiting for a yes", "", "Nothing here has been uploaded. `python scripts/apply_hf_updates.py` prints this plan; `--yes` carries it out. No repository is created, deleted or renamed, and no visibility changes.", "",
            "1. `git push origin main`: the paper, the checker, the survey, the README and this folder.", "2. Model, dataset and Space cards (`README.md` of each repository), from `hf_updates/cards/`:"]
    plan += [f"   - `{USER}/{n}` ({k})" + ("" if f"{k}__{n}" in changed else " (unchanged)") for k, n in targets]
    plan += ["3. Demo Space `itssaharsh/relapse`: `index.html`, `app.js`, `data/stage2.json` and the preview image `og.png` from `space/` (Tiny Aya Base stated beside the headline, means in the table, links to the paper and the checker).",
             "4. Write-up Space `itssaharsh/relapse-article`: `index.html` from `hf_updates/space__relapse-article/` (rebuilt from `docs/ARTICLE.md`).",
             "5. Dataset `itssaharsh/relapse-runs`: add `stage2/survey/` (the files of `results/survey/`), replace `stage2/paper.pdf` with the final `paper/main.pdf`, add `stage2/check_tokenizer.py` (the checker, so it sits inside the judged collection), `stage2/single_line/` and `stage2/PREREGISTRATION_6.md`, and add `stage2/PREREGISTRATION_5.md` if it is not there.",
             "6. Both collections: descriptions that name the paper and the checker, item notes, and the models collection ordered Stage 2 first (`hf_updates/collection.json`):"] + [f"   {i + 1}. `{r}`" for i, r in enumerate(order)]
    plan += ["   Added after them, as the submission form asks for every model and dataset used: " + ", ".join(f"`{r}`" for r in used_models) + " in the models collection; "
             + ", ".join(f"`{r}`" for r in used_data) + " in the data collection. Each is marked as used, not ours."]
    plan += [f"7. The five Stage 1 model repositories that carry the outdated setting ({', '.join('`' + n + '`' for n in revs)}): replace `tokenizer_config.json` with Cohere's fixed file "
             f"(`{VENDOR_REPO}` at `{VENDOR_REV[:10]}`). Their cards give the date ({FIX_DATE}, UTC) and the revision the Stage 1 results used. Done on that date; the apply script leaves a repository that already has the fixed file alone."]
    open(os.path.join(OUT, "PLAN.md"), "w").write("\n".join(plan) + "\n")
    print(f"wrote {len(cards)} cards ({len(changed)} differ from what is live), the write-up page, collection.json and PLAN.md to {OUT}")


if __name__ == "__main__":
    main()
