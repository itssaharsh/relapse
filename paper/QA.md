# Questions the organiser's researchers are most likely to ask

Not submitted. Each answer is two lines, with the result files behind it. The numbers are filled in from `paper/numbers.tex`, which `scripts/paper_numbers.py` writes from `results/`.

## 1. Is the tokenizer effect anything more than your own Stage 1 bug?

The two settings are two public copies of the same model: Cohere fixed `tokenizer_class` on Hugging Face on 30 September 2026, and Kaggle's copy still names the old class. On 26 November 2026, all 5 Kaggle variations and 418 derived Hugging Face repositories (64,619 downloads) still carried it, and 12 of 12 sampled copies split a test prompt differently under transformers 5.20.2.

The controls say it is the unlearning that moves, not the tokenizer alone: original models move by -0.9 to +0.1 points (-2.6 on Base), and the reverse direction gives +1.4 points on average.

*Behind it:* results/survey/copies.csv, results/survey/probe_sample.json, results/shift2/*/results.json, results/shift/summary.json

## 2. Your "0 of 16" at layers 3 to 5 depends on a filter that removes Tiny Aya Base. What is the count without it?

Counting every run, 20 of 20 rose by 5 points or more at layers 5 to 7 (mean +12.1), 4 of 20 at layers 3 to 5 (+3.0) and 4 of 20 at layers 7 to 9; outside the default layers every one of them is Base.

Base unlearns to chance through the outdated setting (24.6 to 25.7% at layers 3 to 5), which the registered effective-run rule (30 to 55%) excludes, and rises by 8.5 to 11.6 points. The paper says so on page one, in Table 1 and in Limitations.

*Behind it:* results/shift2/summary.json (runs, not_effective), results/PREREGISTRATION_5.md

## 3. Why would the layers that were edited decide whether a tokenizer change undoes the unlearning?

We do not know, and the paper claims no mechanism (the Conclusion sentence that offered one was removed). What we have is the observation, registered in advance and seen twice.

In the first tokenizer experiment the rise was smaller at layers 7 to 9 than at 5 to 7 in 16 of 16 matched pairs; in the second, layers 3 to 5 and 7 to 9 relapsed in 0 and 0 of 16 effective runs against 16 of 16 at the default layers (T1 to T3 all held).

*Behind it:* results/shift/summary.json (P2), results/shift2/summary.json (T1, T2, T3)

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

Untested, and Limitations says so. The two settings split text differently at line breaks and digit groups (the same probe prompt is 45 tokens as trained and 52 through the outdated class), and multiple-choice prompts are full of line breaks.

Our expectation, stated in the paper as what more credits would buy, is a smaller rise at the default layers on single-line prompts, not none, because digit groups still split differently. It would be registered before it is run.

*Behind it:* scripts/check_tokenizer.py --probe, results/survey/probe_sample.json, paper Section 5
