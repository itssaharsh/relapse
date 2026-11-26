TITLE: tokenizer_config.json here still names CohereTokenizerFast: under transformers 5.x the text is split differently from Hugging Face's copy

The five variations of this model carry `tokenizer_class: CohereTokenizerFast` (`CohereTokenizer` for `base`) in `tokenizer_config.json`. Cohere changed that field to `PreTrainedTokenizerFast` in the Hugging Face repositories on 30 September 2026 ("so transformers v5 loads the pre-tokenizer"). The fixed revisions are:

- `CohereLabs/tiny-aya-global` at `af89d219b5`
- `CohereLabs/tiny-aya-fire` at `73500c80fc`
- `CohereLabs/tiny-aya-earth` at `0624fafbe2`
- `CohereLabs/tiny-aya-water` at `41a99216d1`
- `CohereLabs/tiny-aya-base` at `d62ac17e24`

With `transformers` 5.x, the old value loads a tokenizer that splits text differently from the one the model was trained with: every digit apart, punctuation separated from the word it joins, and no joined punctuation-plus-line-break pieces. All 1,273 WMDP-bio prompts and all 1,000 MMLU prompts we tried are split differently (5.9% more tokens). The weights are the same; only the split changes.

To check a copy (it reads only the tokenizer file, or a GGUF header):

```
python check_tokenizer.py kaggle:cohere-labs/tiny-aya/transformers/global/1
```

The script is at https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py. A workaround in a notebook is to set `tokenizer_class` to `PreTrainedTokenizerFast` in a copy of `tokenizer_config.json`, or to load the tokenizer files from the Hugging Face revision above.

What it changed for us: models we unlearned and scored through this setting looked more unlearned than they were once loaded with the Hugging Face tokenizer. The measurements are in https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf.
