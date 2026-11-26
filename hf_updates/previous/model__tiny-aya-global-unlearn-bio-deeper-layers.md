---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: transformers
pipeline_tag: text-generation
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, gguf, tiny-aya, lexsi]
datasets: [cais/wmdp, cais/mmlu]
---
# Tiny Aya Global, RMU at layers 10 to 12 (a control)

> **Correction (Nov 21, 2026).** We first published this model as a test case that looks unlearned at full precision and relapses in 4-bit files. **That claim is withdrawn.** It came from an outdated tokenizer setting. With the official tokenizer this model barely forgets, and its 4-bit files do not rise. [What changed](https://github.com/itssaharsh/relapse#correction-november-21-2026).

**Do not use this model for safety.** It is kept only so the correction can be checked.

Scores with the official tokenizer, as WMDP-bio (hazardous biology, lower is better, 25% is guessing) / MMLU (general knowledge). The original scores 63.3% / 55.4% at fp16.

| fp16: 52.2% / 54.7% | Q4_K_M: 52.8% / 54.1% | Q4_0: 52.5% / 53.6% | NF4: 50.6% / 52.5% |
|---|---|---|---|

## Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")   # the official tokenizer
model = AutoModelForCausalLM.from_pretrained("itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers", dtype="auto")
```

The license and the [Cohere Labs acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) of the base model apply (non-commercial).
