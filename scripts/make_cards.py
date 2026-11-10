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
            rows.append(f"| {f['label']} | {pct(c['forget'])} | {pct(c['retain'])} | {pct((c.get('orig') or o.get(f['id'], {})).get('retain'))} |")
    return "\n".join(rows)


def stat(method: str, to: str):
    return next((s for s in RES.get("stats", []) if s["method"] == method and s["kind"] == "forget_holds" and s.get("to") == to), None)


def model_card(method: str, repo: str, layers: str, gguf: str | None, relapses: bool = False) -> str:
    m = {x["id"]: x for x in RES["methods"]}[method]
    g = RES["grid"][method]["fp16"]
    n_runs = len({p["id"] for p in RES["frontier"] if p["family"] != "original"})
    sg = lambda x: f"{100 * x:+.1f}"
    pv = lambda p: "p < 0.001" if p < 0.0005 else f"p = {p:.3f}"
    title = f"Tiny Aya Global, hazardous biology unlearned ({m['short']})"
    if relapses:
        title = "Tiny Aya Global, unlearned at deeper layers: a test case that relapses in 4-bit files"
        lines = []
        for to in ("Q4_0", "Q4_K_M"):
            s, c = stat(method, to), RES["grid"][method].get(to)
            if s and c:
                lines.append(f"- **{to}**: {pct(c['forget'])}, {sg(s['delta'])} points from fp16 (95% interval {sg(s['ci95'][0])} to {sg(s['ci95'][1])}, {pv(s['p_value'])})")
        warn = f"""
> **This model is a test case, not a model to use.** It looks unlearned at full precision ({pct(g['forget'])} on WMDP-bio) and
> part of the knowledge comes back when it is stored as a 4-bit file. It is published so anyone can check an unlearning audit
> against a model that looks unlearned at fp16 and gives part of it back in the file people download. The models to use are
> [{USER}/tiny-aya-global-unlearn-bio-circuitkit-layers](https://huggingface.co/{USER}/tiny-aya-global-unlearn-bio-circuitkit-layers) and its siblings.

## The relapse

AuditKIT paired bootstrap on the 1,273 WMDP-bio questions, fp16 against real llama.cpp files of the same model:

""" + "\n".join(lines) + """

Both files are in this repository (`*-Q4_0.gguf`, `*-Q4_K_M.gguf`), next to the fp16 checkpoint, so the check can be repeated.
Rounding only the layers in front of the edit reproduces most of the effect. The same edit at the other four layer windows we
tried does not do this; see the write-up.
"""
        formats_note = "Q4_K_M and Q4_0 are real llama.cpp files; NF4 is a bit-exact weight round trip. The other formats were not scored for this model."
        limits = """A research artifact for studying unlearning and quantization, published as a negative example: its forgetting does
not hold in 4-bit files. Knowledge is measured with multiple-choice questions (WMDP-bio), not open generation. The license and the
[Cohere Labs acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) of the base model apply (non-commercial)."""
    else:
        warn = ""
        formats_note = "Q6_K and Q4_K_M are real llama.cpp files; Q8_0, Q4_0 and NF4 are bit-exact weight round trips."
        limits = """A research artifact for studying unlearning and quantization. The forgetting held across seven file formats, four other
languages and prompt attacks (worked examples, an expert persona). It did not survive fine-tuning: 25 LoRA steps on 64
WMDP-bio questions brought most of it back, so it protects against prompting the shipped file, not against someone who
retrains the weights. Knowledge removal is measured with multiple-choice questions (WMDP-bio), not open generation; it is
not a guarantee that no hazardous information can be elicited. The license and the
[Cohere Labs acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) of the base model apply (non-commercial)."""
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

# {title}

{m['label']}. Part of [Relapse]({SPACE}): forget the hazard, keep the rest, and check it in the 4-bit file people download.
{warn}
This is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) with hazardous biology knowledge removed by
[SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU, editing only the MLP down-projections of layers {layers}. At full precision it
scores {pct(g['forget'])} on WMDP-bio (the original scores {pct(RES['grid']['original']['fp16']['forget'])}; guessing scores 25%) and
{pct(g['retain'])} on MMLU (original {pct(RES['grid']['original']['fp16']['retain'])}).

## Every file format

{table(method)}

{formats_note} All {n_runs} runs of the sweep
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

{limits}
"""


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "cards")
    os.makedirs(out, exist_ok=True)
    spec = json.load(open(os.path.join(ROOT, "results", "publish.json")))
    for s in spec["models"]:
        open(os.path.join(out, s["repo"] + ".md"), "w").write(model_card(s["method"], s["repo"], s["layers"], s.get("gguf_name"), s.get("relapses", False)))
        print("card", s["repo"])
