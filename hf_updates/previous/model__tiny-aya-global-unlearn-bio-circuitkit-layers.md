---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: transformers
pipeline_tag: text-generation
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, gguf, tiny-aya, lexsi]
datasets: [cais/wmdp, cais/mmlu]
---

# Tiny Aya Global, hazardous biology unlearned (CircuitKIT layers)

> **Correction (Nov 21, 2026).** This model was trained on Kaggle with an outdated tokenizer setting that splits text differently from the way Tiny Aya was trained. Scores below this notice were measured with that setting and are **superseded**. With the official tokenizer (WMDP-bio / MMLU; the original scores 63.3% / 55.4% at fp16):

| fp16: 35.7% / 54.3% | Q4_K_M: 36.8% / 53.8% | Q4_0: 34.1% / 52.5% | NF4: 36.1% / 51.7% |
|---|---|---|---|

[What changed](https://github.com/itssaharsh/relapse#correction-november-21-2026).

SafeTune RMU at the layers CircuitKIT found (7 to 9). Part of [Relapse](https://huggingface.co/spaces/itssaharsh/relapse): forget the hazard, keep the rest, and check it in the 4-bit file people download.

This is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) with hazardous biology knowledge removed by
[SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU, editing only the MLP down-projections of layers 7 to 9 (found with CircuitKIT). At full precision it
scores 36.8% on WMDP-bio (the original scores 63.3%; guessing scores 25%) and
53.2% on MMLU (original 55.4%).

## Every file format

| File | WMDP-bio (forget, lower is better) | MMLU (keep, higher is better) | Original MMLU on the same file |
|---|---|---|---|
| fp16 | 36.8% | 53.2% | 55.4% |
| bf16 | 36.9% | 53.1% | 55.4% |
| Q8_0 | 37.3% | 53.2% | 55.1% |
| Q6_K | 36.4% | 53.8% | 55.6% |
| Q4_K_M | 37.1% | 54.9% | 54.8% |
| Q4_0 | 35.9% | 52.7% | 55.5% |
| NF4 | 35.3% | 51.2% | 52.7% |

Q6_K and Q4_K_M are real llama.cpp files; Q8_0, Q4_0 and NF4 are bit-exact weight round trips. All 30 runs of the sweep
are in [itssaharsh/tiny-aya-global-unlearn-bio-sweep](https://huggingface.co/itssaharsh/tiny-aya-global-unlearn-bio-sweep). All runs, per-question probabilities and
AuditKIT comparisons are in [itssaharsh/relapse-runs](https://huggingface.co/datasets/itssaharsh/relapse-runs); code in [https://github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse).

## Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-layers")
model = AutoModelForCausalLM.from_pretrained("itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-layers", dtype="auto")
```

### llama.cpp (the 4-bit file)

```bash
llama-cli -m tiny-aya-global-unlearn-bio-circuitkit-layers-Q4_K_M.gguf -p "Hello"
```

## Intended use and limits

A research artifact for studying unlearning and quantization. The forgetting held across seven file formats, four other
languages and prompt attacks (worked examples, an expert persona). It did not survive fine-tuning: 25 LoRA steps on 64
WMDP-bio questions brought most of it back, so it protects against prompting the shipped file, not against someone who
retrains the weights. Knowledge removal is measured with multiple-choice questions (WMDP-bio), not open generation; it is
not a guarantee that no hazardous information can be elicited. The license and the
[Cohere Labs acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) of the base model apply (non-commercial).
