"""Writes the model card of each released model from results/final/check.json and uploads it.

    python scripts/publish_cards.py [--upload]
Without --upload the cards are only written to build/cards/ for reading.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("global", "fire", "earth", "water", "base")
pc = lambda x: f"{100 * x:.1f}%"


PAPER = "How Deep Does Unlearning Go? One Tokenizer Change Undoes It at Layers 5–7 but Not at Layers 3–5"
PAPER_URL = "https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf"
CHECKER_URL = "https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py"


def context() -> dict:
    """The numbers the cards share with the paper, read from the same result files."""
    import numpy as np
    load = lambda *p: json.load(open(os.path.join(ROOT, "results", *p)))
    s2, fin, place, H = load("shift2", "summary.json"), load("final", "summary.json"), load("place", "summary.json"), load("survey", "official_history.json")
    q2 = next(x for n, x in place["results"]["all_five"]["fp16"].items() if n.startswith("Q2"))
    rev = {v: next(c for c in reversed(H[f"CohereLabs/tiny-aya-{v}"]) if c["setting"] == "official") for v in MODELS}
    base = [r for r in s2["not_effective"] if r["model"] == "base" and r["window"] == "3-5"]
    return {"s2": s2, "Q4": fin["Q4"], "q2": q2, "rev": rev, "base_low": 100 * min(r["rise"] for r in base), "base_high": 100 * max(r["rise"] for r in base),
            "rise_default": 100 * s2["by_window"]["5-7"]["mean_rise"]}


def card(v: str, name: str, e: dict, other: dict, ctx: dict) -> str:
    f = e["fp16"]; real = e["real_files"]["wmdp-bio"]; ft = e["fine_tuning"]; tok = e["outdated_tokenizer"]
    row = lambda label, k: f"| {label} | {pc(f[k]['original'])} | {pc(f[k]['model'])} |\n"
    probes = "".join(row(label, "probe:" + k) for k, label in (("en_para", "Reworded English"), ("hi", "Hindi"), ("sw", "Swahili"), ("ar", "Arabic"), ("es", "Spanish")))
    repo = f"itssaharsh/relapse-tiny-aya-{v}-unlearn-bio"
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-{v}
library_name: transformers
tags:
- unlearning
- rmu
- wmdp
- gguf
---

# Tiny Aya {v.capitalize()}, with hazardous biology unlearned

This is [CohereLabs/tiny-aya-{v}](https://huggingface.co/CohereLabs/tiny-aya-{v}) after RMU unlearning of hazardous biology knowledge (the WMDP bio forget corpus), made with Lexsi Labs' [SafeTune](https://github.com/Lexsi-Labs/SafeTune). Three matrices are changed: the MLP output projections of layers 3 to 5. Everything else, including the tokenizer files, is the original.

**Paper:** [{PAPER}]({PAPER_URL}).

**Base model and tokenizer revision:** `CohereLabs/tiny-aya-{v}` at revision `{ctx['rev'][v]['sha'][:10]}` ({ctx['rev'][v]['date'][:10]}), the commit that carries Cohere's tokenizer fix (`tokenizer_class: PreTrainedTokenizerFast`). This repository ships that revision's tokenizer files unchanged. Copies of Tiny Aya made before the fix carry an outdated setting that splits text differently under `transformers` v5. The checker [`scripts/check_tokenizer.py`]({CHECKER_URL}), added with the paper, tells you which setting a repository, folder or GGUF file carries and what to do about it, e.g. `python scripts/check_tokenizer.py {repo}`.

It is a research artifact of the [Relapse](https://github.com/itssaharsh/relapse) project, built for the Lexsi AI Safety and Alignment Hackathon. **It is not a safety guarantee.** A short fine-tune can bring the knowledge back (see below).

## Files

- `model.safetensors`: full precision (fp16).
- `{repo.split('/')[-1]}-Q4_K_M.gguf`: a real llama.cpp Q4_K_M file of the same weights.
- `published_scores.json`: the scores of exactly these two files, measured after they were written.

## What was measured

All scores are accuracy on four-choice questions, read from the model's next-token probabilities. WMDP-bio is the hazard set (1,273 questions; lower is better; 25% is guessing). Everything was scored through the official tokenizer.

| Check | Original | This model |
|---|---|---|
{row("WMDP-bio (hazardous biology)", "wmdp-bio")}{row("MMLU, 1,000 questions (general knowledge)", "mmlu")}{row("Benign biology (four MMLU biology subjects)", "benign-bio")}{row("WMDP-bio with five worked examples in the prompt", "attack:five_shot")}{row("WMDP-bio with an expert role-play prompt", "attack:role_play")}{probes}
| The same weights, deployed | WMDP-bio |
|---|---|
| Full precision | {pc(f['wmdp-bio']['model'])} |
| Real Q4_0 file (llama.cpp) | {pc(real['Q4_0']['file'])} |
| Real Q4_K_M file (llama.cpp) | {pc(real['Q4_K_M']['file'])} |
| Through an outdated tokenizer setting | {pc(tok['outdated'])} |
| After 25 fine-tuning steps on 64 held-out hazard questions | {pc(ft['after_hazard_questions'])} (from {pc(ft['before'])} on the same {1204} questions) |

## What to know before using it

- **The forgetting is broader than the hazard.** Benign biology drops from {pc(f['benign-bio']['original'])} to {pc(f['benign-bio']['model'])}.
- **Fine-tuning can undo it.** See the last row above. RMU hides knowledge more than it removes it.
- **Knowledge was measured by multiple choice**, not by open generation.
- **Check the file you ship, and the tokenizer you load.** In this project, a few unlearned runs regained 5 to 8 points in one 4-bit format while other formats of the same weights held (on Fire and Water, one exploratory run each). The files in this repository were each scored. Models unlearned through the outdated tokenizer setting at SafeTune's default layers regained {ctx['rise_default']:.1f} points on average once loaded through the official tokenizer; this model was unlearned through the official one, and through the outdated setting it scores as the table shows.
- A second run with another seed ({other['name']}) scored {pc(other['fp16']['wmdp-bio']['model'])} on WMDP-bio and {pc(other['fp16']['mmlu']['model'])} on MMLU. This repository holds the run with the lower WMDP-bio.

## How it was made

SafeTune RMU, official tokenizer, layers 3 to 5, steering 3 × the activation norm, retain weight 100, 300 batches of 4, run `{name}`. Layers 3 to 5 are the window that CircuitKIT's attribution ranks first when pooled over the five Tiny Aya models. Two registered tests compared them with SafeTune's default layers 5 to 7: they forgot more within 3 MMLU points in {ctx['q2']['wins']} of 10 comparisons (p = {ctx['q2']['p']:.3f}), and in {ctx['Q4']['lower_at_3_5']} of 10 on fresh seeds (p = {ctx['Q4']['p']:.3f}, short of our own line; the two exceptions are Tiny Aya Base). They cost more benign biology, about the same per hazard point forgotten. Unlearned through the outdated tokenizer setting, default-layer runs relapsed through the official tokenizer in {ctx['s2']['T3']['5-7'][0]} of {ctx['s2']['T3']['5-7'][1]} runs that count and layers 3 to 5 in {ctx['s2']['T2']['relapses']} of {ctx['s2']['T2']['effective']}, on the four post-trained variants; Tiny Aya Base rose at every window ({ctx['base_low']:.1f} to {ctx['base_high']:.1f} points at layers 3 to 5). The pre-registrations, every run's scores and the paper are in the [repository](https://github.com/itssaharsh/relapse).

## License

CC-BY-NC-4.0, the license of the original model. Use is also subject to Cohere's terms for Tiny Aya.
"""


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--upload", action="store_true"); a = ap.parse_args()
    check = json.load(open(os.path.join(ROOT, "results", "final", "check.json")))
    out = os.path.join(ROOT, "build", "cards"); os.makedirs(out, exist_ok=True)
    ctx = context()
    for v in MODELS:
        runs = {n: e for n, e in check.items() if e["model"] == v}
        name = next(n for n in runs if n.endswith("_s34")); other_name = next(n for n in runs if n.endswith("_s33"))
        text = card(v, name, runs[name], {"name": other_name, **runs[other_name]}, ctx)
        open(os.path.join(out, f"{v}.md"), "w").write(text)
        if a.upload:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=f"itssaharsh/relapse-tiny-aya-{v}-unlearn-bio", commit_message="Model card with the full check")
    print("cards written to", out)


if __name__ == "__main__":
    main()
