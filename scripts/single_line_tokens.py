"""How many single-line prompts the two tokenizer settings split differently, and the rise on those alone (exploratory).

    python scripts/single_line_tokens.py            # writes results/survey/single_line_tokens.json

No model is loaded and nothing is scored. The prompts are rebuilt exactly as the jobs of pre-registration 6 built them
(kaggle/20_single_line.py, same dataset revisions), tokenized under the official setting and the outdated one, and
compared. A prompt with the same token ids under both settings gets the same answer under both by construction, so the
agreement of the two settings on single-line prompts is only informative on the prompts they split differently. From
the saved per-question results (results/single_line/*/items.json) this reports the rise of the default-layer runs on
that subset alone, next to the rise of the same runs on the same questions in the registered layout
(results/shift2/*/items.json). It also checks the reconstruction: on prompts with identical ids the two settings should
give the same answers, up to the few that flip because a prompt's batch (eight prompts of similar length, padded) is
made up differently under the other setting; the largest such count over all saved runs is recorded.
Needs transformers 5.x, tokenizers and datasets. Only counts are stored.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from token_differences import compare, tokenizers_pair  # noqa: E402

WMDP_REV, MMLU_REV = "7125571f22f032c56415e7980f48d877dd830ff8", "c30699e8356da336a370243923dbaf21066bb9fe"      # as hf_jobs/header_single.py
LET = "ABCD"
one = lambda s: " ".join(str(s).split())


def fmt_single(subject, q, choices):      # as kaggle/20_single_line.py
    return f"The following are multiple choice questions (with answers) about {subject}. {one(q)} " + " | ".join(f"{l}. {one(c)}" for l, c in zip(LET, choices)) + " Answer:"


def main() -> None:
    from datasets import load_dataset
    official, outdated, version = tokenizers_pair()
    wmdp = load_dataset("cais/wmdp", "wmdp-bio", split="test", revision=WMDP_REV)
    mmlu = load_dataset("cais/mmlu", "all", split="test", revision=MMLU_REV).shuffle(seed=0).select(range(1000))
    P = {"wmdp-bio": [fmt_single("biology", r["question"], r["choices"]) for r in wmdp], "mmlu": [fmt_single(r["subject"].replace("_", " "), r["question"], r["choices"]) for r in mmlu]}
    ids = lambda tok, texts: [tok(t)["input_ids"] for t in texts]      # with special tokens, as the scorer feeds them
    differ = {k: np.array([x != y for x, y in zip(ids(official, v), ids(outdated, v))]) for k, v in P.items()}
    out = {"transformers": version, "wmdp_bio_single_line_prompts": compare(P["wmdp-bio"], official, outdated), "mmlu_single_line_prompts": compare(P["mmlu"], official, outdated)}
    assert out["wmdp_bio_single_line_prompts"]["texts_split_differently"] == int(differ["wmdp-bio"].sum()) and out["mmlu_single_line_prompts"]["texts_split_differently"] == int(differ["mmlu"].sum())
    five = json.load(open(os.path.join(ROOT, "results", "shift2", "summary.json")))
    default = {(r["model"], r["seed"]) for r in five["runs"] if r["window"] == "5-7" and r["effective"]}
    flips, flips_differ, checked = [], [], 0; single, registered, whole = [], [], []
    for v in ("global", "fire", "earth", "water", "base"):
        items = json.load(open(os.path.join(ROOT, "results", "single_line", v, "items.json")))
        for key in [k for k in items if "|official|single|" in k]:      # the reconstruction check, on every saved run and both question sets
            a, b = np.array(items[key]), np.array(items[key.replace("|official|", "|outdated|")]); same = ~differ[key.split("|")[-1]]
            checked += 1
            if key.endswith("wmdp-bio"):
                flips.append(int((a[same] != b[same]).sum())); flips_differ.append(int((a[~same] != b[~same]).sum()))
        for sd in (51, 52):
            if (v, sd) not in default:
                continue
            old = json.load(open(os.path.join(ROOT, "results", "shift2", f"{v}_seed{sd}", "items.json")))
            for name in sorted({k.split("|")[0] for k in items if "_W5-7_" in k and k.split("|")[0].endswith(f"_s{sd}")}):
                d = differ["wmdp-bio"]
                s = np.array(items[f"{name}|fp16|official|single|wmdp-bio"], float) - np.array(items[f"{name}|fp16|outdated|single|wmdp-bio"], float)
                r = np.array(old[f"{name}|fp16|official|wmdp-bio"], float) - np.array(old[f"{name}|fp16|outdated|wmdp-bio"], float)
                single.append(float(s[d].mean())); registered.append(float(r[d].mean())); whole.append(float(s.mean()))
    out["reconstruction_check"] = {"runs": len(flips), "wmdp_bio_prompts_with_identical_ids": int((~differ["wmdp-bio"]).sum()), "most_answers_differing_on_them_in_one_run": max(flips),
                                   "mean_answers_differing_on_them": float(np.mean(flips)), "wmdp_bio_prompts_split_differently": int(differ["wmdp-bio"].sum()),
                                   "fewest_answers_differing_on_them_in_one_run": min(flips_differ), "mean_answers_differing_on_them_split_differently": float(np.mean(flips_differ))}
    out["default_layer_rise_on_prompts_split_differently"] = {"runs": len(single), "questions": int(differ["wmdp-bio"].sum()), "mean_rise_single_line": float(np.mean(single)), "max_rise_single_line": float(np.max(single)),
                                                              "mean_rise_registered_layout_same_questions": float(np.mean(registered)), "mean_rise_single_line_all_questions": float(np.mean(whole))}
    json.dump(out, open(os.path.join(ROOT, "results", "survey", "single_line_tokens.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
