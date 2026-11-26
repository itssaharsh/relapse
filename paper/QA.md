# Questions the organiser's researchers are most likely to ask

Not submitted. Each answer is two lines, with the result files behind it. The numbers are filled in from `paper/numbers.tex`, which `scripts/paper_numbers.py` writes from `results/`.

## 1. Is the tokenizer effect anything more than your own Stage 1 bug?

The two settings are two public copies of the same model: Cohere fixed `tokenizer_class` on Hugging Face on 30 September 2026, and Kaggle's copy still names the old class. On 26 November 2026, all 5 Kaggle variations and 418 derived Hugging Face repositories (64,619 downloads) still carried it (exposure, not harm: none is named as an unlearning model, 6 as safety-tuned, and we saw the relapse only in our own runs), and 12 of 12 sampled copies split a test prompt differently under transformers 5.20.2.

The controls say it is the unlearning that moves, not the tokenizer alone: original models move by -0.9 to +0.1 points (-2.6 on Base), and the reverse direction gives +1.4 points on average.

*Behind it:* results/survey/copies.csv, results/survey/probe_sample.json, results/shift2/*/results.json, results/shift/summary.json

## 2. Your "0 of 16" at layers 3 to 5 depends on a filter that removes Tiny Aya Base. What is the count without it?

Counting every run, 20 of 20 rose by 5 points or more at layers 5 to 7 (mean +12.1), 4 of 20 at layers 3 to 5 (+3.0) and 4 of 20 at layers 7 to 9; outside the default layers every one of them is Base.

Base unlearns to chance through the outdated setting (24.6 to 25.7% at layers 3 to 5), which the registered effective-run rule (30 to 55%) excludes, and rises by 8.5 to 11.6 points. The paper says so on page one, in Table 1 and in Limitations.

*Behind it:* results/shift2/summary.json (runs, not_effective), results/PREREGISTRATION_5.md

## 3. Why would the layers that were edited decide whether a tokenizer change undoes the unlearning? Is it the layer at all?

We claim no mechanism, and the layer is confounded. An exploratory reading of the same runs: where they were audited, the default-layer runs had also lost 5.8 MMLU points (none within our 3-point line) and 1.7 once deployed; at layers 3 to 5 the loss was 0.5 and 1.1. Tiny Aya Base scored 23.1 to 24.0% on MMLU as audited.

Over both tokenizer experiments, runs that had lost 3 or more MMLU points as audited rose by 5 points or more in 72 of 72 cases, the others in 0 of 48. About half of the audited forgetting returned (47%), and the deployed runs land where runs unlearned through the official tokenizer at the same layers, seeds and rates land (50.9% against 52.1%). So the outdated audit overstated the forgetting: it counted damage tied to one tokenization; the rule is fitted on those runs and is not exact outside them (our Stage 1 model at layers 10 to 12 lost 1.5 MMLU points as audited and rose 10.6; the reverse direction gives 4 of 7 against 2 of 40); each window had its own learning rates, though they overlap (3 to 5 held at 3e-5 and 6e-5, 5 to 7 relapsed at 4e-5 and 7e-5), the pattern is not ordered by depth (10 to 12 relapsed, 7 to 9 held), and we did not vary layer and learning rate apart.

*Behind it:* results/shift/*/results.json, results/shift2/*/results.json (MMLU through each tokenizer), scripts/paper_numbers.py audit_damage()

## 4. The results of pre-registration 5 were committed 63 minutes after the registration. Did the runs really start after it?

Yes. The registration and its analysis script were committed at 13:12:18 UTC on 22 November (commit fec9e32); Hugging Face Jobs records the ten jobs as started between 13:12:27 and 13:13:11, each carrying that commit as its repository commit.

They ran on l4x1 (six runs per job, fp16 scoring through two tokenizers plus two round trips), and the results were committed at 14:15. Three earlier jobs that started beside registrations 1, 2 and 4 are three-batch pipeline tests (suffix `_smoke`, seeds 98 and 99), not experiment runs.

*Behind it:* results/survey/jobs.json, git log of results/PREREGISTRATION_5.md and results/shift2/

## 5. Does your 4-bit result contradict Lexsi's "Forgetting That Sticks" (Sadhu et al., 2026)?

It tests their prediction on a method and formats they did not cover: RMU (which they discuss but do not run) and llama.cpp GGUF files beside NF4. For RMU, 107 of 109 effective runs kept their forgetting in both Q4_0 and NF4.

Where a relapse did occur, their mechanism (updates smaller than the quantization bin) is not the cause: rounding only the three edited matrices moved the score by at most 1.7 points; the rise comes from rounding weights the edit never touched.

*Behind it:* results/map/summary.json, results/deep/summary.json (part_c), paper/citations_checked.md

## 6. Are layers 3 to 5 really better than SafeTune's default, or do they just forget more of everything?

They give more forgetting per MMLU point (20.8 hazard points for 1.6 MMLU points against 14.6 for 2.3) and robustness to the tokenizer. They do not give selectivity within biology: 3.30 against 3.26 hazard points per point of benign biology lost.

Of the two registered tests, the first passed (9 of 10, p = 0.011) and the fresh-seed one missed (8 of 10, p = 0.055); the two exceptions are Tiny Aya Base. We added no seeds to rescue it.

*Behind it:* results/place/summary.json (Q2), results/final/summary.json (Q4), results/final/check_*/certify.json

## 7. The Fire and Water 4-bit relapses are one run each. Why should anyone believe them?

The paper labels both exploratory: Fire met a registered existence test on 1 of 8 seeds and was never retrained; Water came from runs registered without a hypothesis. Only the Base case was retrained (rises of +4.8, +5.1, +6.1).

They are not noise on the questions: exact McNemar p = 2.9e-5 (Fire) and 2.9e-7 (Water), and 6.2e-4 and 1.1e-5 after a Holm correction over the 22 and 36 cells screened with them. Seed-to-seed variance is what one run cannot show.

*Behind it:* results/deep/*/items.json through scripts/mcnemar_holm.py, results/deep/summary.json

## 8. Is the hazardous knowledge actually gone from the released models?

No. Twenty-five LoRA steps on 64 held-out hazard questions raised the score by 5 points or more on 5 of 10 runs and restored Tiny Aya Base to 61.0 and 52.1%. RMU hides knowledge more than it removes it.

What does hold: prompt attacks did not raise the score on any run, reworded and translated questions stayed below the original in 40 of 50 cases, and real Q4_0 and Q4_K_M files moved it by -3.3 to +0.9 points. The cost is 6.3 points of benign biology.

*Behind it:* results/final/check.json, results/final/check_*/certify.json

## 9. Which model and tokenizer revision did each run use?

No run recorded it: every Stage 2 job took the main branch when it started. From the job start times and the repositories' public history, 147 jobs map to the commits of 30 September 2026 that carry Cohere's fix (Table of revisions in the appendix); the five jobs of 24 November got later commits that change only `generation_config.json`.

Tokenizer files and weights have identical content hashes across those revisions. Stage 1 used whatever copy Kaggle held (version 1, uploaded 16 February 2026), which carries the outdated setting.

*Behind it:* results/survey/jobs.json, results/survey/official_history.json

## 10. Would the tokenizer effect show on prompts that are not multiple choice?

We tested the nearest case and it does not. In a sixth registered test (scoring only, pinned revisions) the same 60 saved runs were scored with every line break removed: the default-layer runs score 46.6% through the outdated setting and 47.4% through the official one, a rise of +0.8 points against +11.4 in the multiple-choice layout, and 0 of 16 relapse. We had registered that at least half would still relapse; that hypothesis failed.

So the overstated audit was absent in the one layout without line breaks that we tested (it also joins the options with ` | `). The agreement is not by construction: the two settings still split 815 of the 1,273 single-line WMDP-bio prompts differently, and on those alone the default-layer runs rise by +1.2 points, against +11.7 on the same questions in the multiple-choice layout (`results/survey/single_line_tokens.json`). Other layouts and open generation are untested, and why is not explained.

*Behind it:* results/PREREGISTRATION_6.md, results/single_line/summary.json, scripts/analyze_single_line.py, results/survey/token_differences.json
