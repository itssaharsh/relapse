---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: transformers
pipeline_tag: text-generation
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, gguf, tiny-aya, lexsi]
datasets: [cais/wmdp, cais/mmlu]
---

# Tiny Aya Global, hazardous biology unlearned (CircuitKIT + 4-bit aware)

> **Correction (Nov 21, 2026).** This model was trained on Kaggle with an outdated tokenizer setting that splits text differently from the way Tiny Aya was trained. Scores below this notice were measured with that setting and are **superseded**. This model has not been rescored with the official tokenizer yet. [What changed](https://github.com/itssaharsh/relapse#correction-november-21-2026).

CircuitKIT layers, trained through a 4-bit quantizer. Part of [Relapse](https://huggingface.co/spaces/itssaharsh/relapse): forget the hazard, keep the rest, and check it in the 4-bit file people download.

This is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) with hazardous biology knowledge removed by
[SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU, editing only the MLP down-projections of layers 7 to 9 (found with CircuitKIT), trained through a 4-bit quantizer. At full precision it
scores 34.0% on WMDP-bio (the original scores 63.3%; guessing scores 25%) and
54.5% on MMLU (original 55.4%).

## Every file format

| File | WMDP-bio (forget, lower is better) | MMLU (keep, higher is better) | Original MMLU on the same file |
|---|---|---|---|
| fp16 | 34.0% | 54.5% | 55.4% |
| bf16 | 34.2% | 54.5% | 55.4% |
| Q8_0 | 34.6% | 54.4% | 55.1% |
| Q6_K | 34.9% | 53.9% | 55.6% |
| Q4_K_M | 35.3% | 54.5% | 54.8% |
| Q4_0 | 35.6% | 52.9% | 55.5% |
| NF4 | 34.7% | 50.8% | 52.7% |

Q6_K and Q4_K_M are real llama.cpp files; Q8_0, Q4_0 and NF4 are bit-exact weight round trips. All 30 runs of the sweep
are in [itssaharsh/tiny-aya-global-unlearn-bio-sweep](https://huggingface.co/itssaharsh/tiny-aya-global-unlearn-bio-sweep). All runs, per-question probabilities and
AuditKIT comparisons are in [itssaharsh/relapse-runs](https://huggingface.co/datasets/itssaharsh/relapse-runs); code in [https://github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse).

## Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-4bit-aware")
model = AutoModelForCausalLM.from_pretrained("itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-4bit-aware", dtype="auto")
```

### llama.cpp (the 4-bit file)

```bash
llama-cli -m tiny-aya-global-unlearn-bio-circuitkit-4bit-aware-Q4_K_M.gguf -p "Hello"
```

## Intended use and limits

A research artifact for studying unlearning and quantization. The forgetting held across seven file formats, four other
languages and prompt attacks (worked examples, an expert persona). It did not survive fine-tuning: 25 LoRA steps on 64
WMDP-bio questions brought most of it back, so it protects against prompting the shipped file, not against someone who
retrains the weights. Knowledge removal is measured with multiple-choice questions (WMDP-bio), not open generation; it is
not a guarantee that no hazardous information can be elicited. The license and the
[Cohere Labs acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) of the base model apply (non-commercial).
