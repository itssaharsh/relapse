---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: transformers
pipeline_tag: text-generation
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, gguf, tiny-aya, lexsi]
datasets: [cais/wmdp, cais/mmlu]
---
# Tiny Aya Global, RMU at layers 10 to 12, trained with the model seen as its Q4_0 file

> **Correction (Nov 21, 2026).** We first published this model as a repair of a relapse. **That claim is withdrawn.** The relapse came from an outdated tokenizer setting, so there was nothing to repair. With the official tokenizer this model barely forgets. [What changed](https://github.com/itssaharsh/relapse#correction-november-21-2026).

**Do not use this model for safety.** It is kept only so the correction can be checked.

Scores with the official tokenizer, as WMDP-bio (hazardous biology, lower is better, 25% is guessing) / MMLU (general knowledge). The original scores 63.3% / 55.4% at fp16.

| fp16: 58.1% / 53.0% | Q4_K_M: 57.5% / 52.5% | Q4_0: 57.2% / 53.5% | NF4: 55.9% / 51.4% |
|---|---|---|---|

## Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")   # the official tokenizer
model = AutoModelForCausalLM.from_pretrained("itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers-file-trained", dtype="auto")
```

The license and the [Cohere Labs acceptable use policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) of the base model apply (non-commercial).
