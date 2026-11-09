"""Zero-shot multiple-choice scoring by next-token letter probabilities (lm-eval style)."""
from __future__ import annotations

import torch

LETTERS = ["A", "B", "C", "D"]
SUBJECTS = {"wmdp-bio": "biology", "wmdp-cyber": "computer security", "wmdp-chem": "chemistry"}


def format_mcq(subject: str, question: str, choices: list[str]) -> str:
    s = f"The following are multiple choice questions (with answers) about {subject}.\n\n{question.strip()}\n"
    for letter, choice in zip(LETTERS, choices):
        s += f"{letter}. {choice}\n"
    return s + "Answer:"


def load_set(name: str, mmlu_n: int = 1000, seed: int = 0) -> list[tuple[str, int]]:
    """'wmdp-bio' | 'wmdp-cyber' | 'wmdp-chem' | 'mmlu' (seeded sample) -> [(prompt, answer_idx)]."""
    from datasets import load_dataset

    if name in SUBJECTS:
        d = load_dataset("cais/wmdp", name, split="test")
        return [(format_mcq(SUBJECTS[name], r["question"], r["choices"]), r["answer"]) for r in d]
    if name == "mmlu":
        d = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(mmlu_n))
        return [(format_mcq(r["subject"].replace("_", " "), r["question"], r["choices"]), r["answer"]) for r in d]
    raise ValueError(name)


def letter_ids(tok) -> list[int]:
    return [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LETTERS]


@torch.no_grad()
def mcq_eval(model, tok, items, bs: int = 8, maxlen: int = 1024):
    """Returns (accuracy, predictions, per-item [pA, pB, pC, pD] renormalised over the four letters)."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = letter_ids(tok)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    preds, probs = [None] * len(items), [None] * len(items)
    dev = next(model.parameters()).device
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([items[i][0] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxlen).to(dev)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        p = torch.softmax(logits[:, ids], -1).cpu()
        for j, i in enumerate(idx):
            preds[i] = int(p[j].argmax())
            probs[i] = [round(float(x), 4) for x in p[j]]
    acc = sum(int(preds[i] == items[i][1]) for i in range(len(items))) / len(items)
    return acc, preds, probs
