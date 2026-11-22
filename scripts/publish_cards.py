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


def card(v: str, name: str, e: dict, other: dict) -> str:
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
- **Check the file you ship.** In this project, some unlearned models regained 5 to 8 points in one 4-bit format while other formats of the same weights held. The files in this repository were each scored.
- A second run with another seed ({other['name']}) scored {pc(other['fp16']['wmdp-bio']['model'])} on WMDP-bio and {pc(other['fp16']['mmlu']['model'])} on MMLU. This repository holds the run with the lower WMDP-bio.

## How it was made

SafeTune RMU, official tokenizer, layers 3 to 5, steering 3 × the activation norm, retain weight 100, 300 batches of 4, run `{name}`. Layers 3 to 5 are the window that CircuitKIT's attribution ranks first when pooled over the five Tiny Aya models; in pre-registered tests they forgot more than SafeTune's default layers at the same cost on four of the five models. The pre-registrations, every run's scores and the paper are in the [repository](https://github.com/itssaharsh/relapse).

## License

CC-BY-NC-4.0, the license of the original model. Use is also subject to Cohere's terms for Tiny Aya.
"""


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--upload", action="store_true"); a = ap.parse_args()
    check = json.load(open(os.path.join(ROOT, "results", "final", "check.json")))
    out = os.path.join(ROOT, "build", "cards"); os.makedirs(out, exist_ok=True)
    for v in MODELS:
        runs = {n: e for n, e in check.items() if e["model"] == v}
        name = next(n for n in runs if n.endswith("_s34")); other_name = next(n for n in runs if n.endswith("_s33"))
        text = card(v, name, runs[name], {"name": other_name, **runs[other_name]})
        open(os.path.join(out, f"{v}.md"), "w").write(text)
        if a.upload:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=f"itssaharsh/relapse-tiny-aya-{v}-unlearn-bio", commit_message="Model card with the full check")
    print("cards written to", out)


if __name__ == "__main__":
    main()
