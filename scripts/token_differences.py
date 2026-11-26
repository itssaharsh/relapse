"""What the two tokenizer settings do to the text, counted on a laptop: no model is loaded and nothing is scored.

    python scripts/token_differences.py            # writes results/survey/token_differences.json

For the 1,273 WMDP-bio prompts, the 1,000 MMLU prompts and Wikitext retain texts, in the layout the experiments used:
how many texts get different token ids under the official setting (tokenizer.json as trained) and the outdated one
(the same files with tokenizer_class set to CohereTokenizerFast, as in Kaggle's copy), how many tokens each gives, and
which pieces differ. Then, from the saved per-question results of pre-registration 5, the rise of the default-layer runs on
WMDP-bio questions with and without a digit in the prompt (exploratory). Needs transformers 5.x, tokenizers and datasets.
No question text is stored: only counts.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "kaggle"))
from check_tokenizer import HF, _get  # noqa: E402

REPO = "CohereLabs/tiny-aya-global"


def tokenizers_pair():
    import transformers
    d = tempfile.mkdtemp()
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        open(os.path.join(d, f), "wb").write(_get(f"{HF}/{REPO}/resolve/main/{f}"))
    official = transformers.AutoTokenizer.from_pretrained(d)
    old = tempfile.mkdtemp()
    for f in ("tokenizer.json", "special_tokens_map.json"):
        open(os.path.join(old, f), "wb").write(open(os.path.join(d, f), "rb").read())
    cfg = json.load(open(os.path.join(d, "tokenizer_config.json"))); cfg["tokenizer_class"] = "CohereTokenizerFast"      # as kaggle/13 and kaggle/15 do
    json.dump(cfg, open(os.path.join(old, "tokenizer_config.json"), "w"))
    return official, transformers.AutoTokenizer.from_pretrained(old), transformers.__version__


def compare(texts, official, outdated) -> dict:
    a = [official(t, add_special_tokens=False)["input_ids"] for t in texts]
    b = [outdated(t, add_special_tokens=False)["input_ids"] for t in texts]
    only_a, only_b = {}, {}
    for x, y in zip(a, b):
        for i in set(x) - set(y):
            only_a[i] = only_a.get(i, 0) + 1
        for i in set(y) - set(x):
            only_b[i] = only_b.get(i, 0) + 1
    kind = lambda tok, ids: {"line_break": sum(n for i, n in ids.items() if "\n" in tok.decode([i])), "digit": sum(n for i, n in ids.items() if re.search(r"\d", tok.decode([i]))),
                             "other": sum(n for i, n in ids.items() if "\n" not in tok.decode([i]) and not re.search(r"\d", tok.decode([i])))}
    return {"texts": len(texts), "texts_split_differently": sum(x != y for x, y in zip(a, b)), "tokens_official": int(sum(map(len, a))), "tokens_outdated": int(sum(map(len, b))),
            "pieces_only_official": kind(official, only_a), "pieces_only_outdated": kind(outdated, only_b)}


def main() -> None:
    from datasets import load_dataset

    from common import fmt_mcq
    official, outdated, version = tokenizers_pair()
    wmdp = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(1000))
    wiki = [r["text"] for r in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train") if len(r["text"]) > 50][:1200]
    prompts = [fmt_mcq("biology", r["question"], r["choices"]) for r in wmdp]
    out = {"transformers": version, "official_class": type(official).__name__, "outdated_class": type(outdated).__name__,
           "wmdp_bio_prompts": compare(prompts, official, outdated),
           "mmlu_prompts": compare([fmt_mcq(r["subject"].replace("_", " "), r["question"], r["choices"]) for r in mmlu], official, outdated),
           "wikitext_retain_texts": compare([t[:2000] for t in wiki], official, outdated)}
    # the same questions with the layout removed: is it the line breaks, or the digits?
    out["wmdp_bio_questions_alone"] = compare([r["question"].strip() for r in wmdp], official, outdated)
    # exploratory: the rise of the default-layer runs of pre-registration 5, on questions with and without a digit in the prompt
    digit = np.array([bool(re.search(r"\d", p)) for p in prompts])
    rises = {"with_digit": [], "without_digit": []}
    s2 = json.load(open(os.path.join(ROOT, "results", "shift2", "summary.json")))
    for v in ("global", "fire", "earth", "water"):
        for sd in (51, 52):
            items = json.load(open(os.path.join(ROOT, "results", "shift2", f"{v}_seed{sd}", "items.json")))
            for name in sorted({k.split("|")[0] for k in items if "_W5-7_" in k}):
                if not any(r["model"] == v and r["seed"] == sd and r["window"] == "5-7" and r["effective"] for r in s2["runs"]):
                    continue
                aud, dep = np.array(items[f"{name}|fp16|outdated|wmdp-bio"]), np.array(items[f"{name}|fp16|official|wmdp-bio"])
                rises["with_digit"].append(float((dep[digit] - aud[digit]).mean())); rises["without_digit"].append(float((dep[~digit] - aud[~digit]).mean()))
    out["default_layer_rise"] = {"questions_with_digit": int(digit.sum()), "questions_without_digit": int((~digit).sum()), "runs": len(rises["with_digit"]),
                                 "mean_rise_with_digit": float(np.mean(rises["with_digit"])), "mean_rise_without_digit": float(np.mean(rises["without_digit"]))}
    os.makedirs(os.path.join(ROOT, "results", "survey"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "survey", "token_differences.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
