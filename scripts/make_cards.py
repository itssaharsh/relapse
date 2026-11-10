"""Write Hugging Face model and dataset cards from space/data/results.json (numbers are never typed by hand)."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(os.path.join(ROOT, "space", "data", "results.json")))
USER = os.environ.get("HF_USER", "itssaharsh")
REPO = "https://github.com/itssaharsh/relapse"
SPACE = f"https://huggingface.co/spaces/{USER}/relapse"


def pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "not measured"


def table(method: str) -> str:
    g, o = RES["grid"].get(method, {}), RES["grid"]["original"]
    rows = ["| File | WMDP-bio (forget, lower is better) | MMLU (keep, higher is better) | Original MMLU on the same file |", "|---|---|---|---|"]
    for f in RES["formats"]:
        c = g.get(f["id"])
        if c:
            rows.append(f"| {f['label']} | {pct(c['forget'])} | {pct(c['retain'])} | {pct(o.get(f['id'], {}).get('retain'))} |")
    return "\n".join(rows)


def model_card(method: str, repo: str, layers: str, gguf: str | None) -> str:
    m = {x["id"]: x for x in RES["methods"]}[method]
    g = RES["grid"][method]["fp16"]
    use_gguf = f"""
### llama.cpp (the 4-bit file)

```bash
llama-cli -m {gguf} -p "Hello"
```
""" if gguf else ""
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: transformers
pipeline_tag: text-generation
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, gguf, tiny-aya, lexsi]
datasets: [cais/wmdp, cais/mmlu]
---

# Tiny Aya Global, hazardous biology unlearned ({m['short']})

{m['label']}. Part of [Relapse]({SPACE}): is the forgotten knowledge still gone in the 4-bit file?

This is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) with hazardous biology knowledge removed by
[SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU, editing only the MLP down-projections of layers {layers}. At full precision it
scores {pct(g['forget'])} on WMDP-bio (the original scores {pct(RES['grid']['original']['fp16']['forget'])}; guessing scores 25%) and
{pct(g['retain'])} on MMLU (original {pct(RES['grid']['original']['fp16']['retain'])}).

## Every file format

{table(method)}

Q6_K and Q4_K_M are real llama.cpp files; Q8_0, Q4_0 and NF4 are bit-exact weight round trips. The other 13 runs of the sweep
are in [{USER}/tiny-aya-global-unlearn-bio-sweep](https://huggingface.co/{USER}/tiny-aya-global-unlearn-bio-sweep). All runs, per-question probabilities and
AuditKIT comparisons are in [{USER}/relapse-runs](https://huggingface.co/datasets/{USER}/relapse-runs); code in [{REPO}]({REPO}).

## Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("{USER}/{repo}")
model = AutoModelForCausalLM.from_pretrained("{USER}/{repo}", dtype="auto")
```
{use_gguf}
## Intended use and limits

A research artifact for studying unlearning and quantization. The forgetting held across seven file formats, four other
languages and prompt attacks (worked examples, an expert persona). It did not survive fine-tuning: 25 LoRA steps on 64
WMDP-bio questions brought most of it back, so it protects against prompting the shipped file, not against someone who
retrains the weights. Knowledge removal is measured with multiple-choice questions (WMDP-bio), not open generation; it is
not a guarantee that no hazardous information can be elicited. The license and the
[Cohere Labs acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) of the base model apply (non-commercial).
"""


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "cards")
    os.makedirs(out, exist_ok=True)
    spec = json.load(open(os.path.join(ROOT, "results", "publish.json")))
    for s in spec["models"]:
        open(os.path.join(out, s["repo"] + ".md"), "w").write(model_card(s["method"], s["repo"], s["layers"], s.get("gguf_name")))
        print("card", s["repo"])
