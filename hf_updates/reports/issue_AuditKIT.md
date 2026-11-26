TITLE: `compare(a, b).significance(metric)` always uses 1,000 resamples, so its p-value cannot go below 1/1001

**Setting.** AuditKIT 1.1.6.

**What we saw.** `RunComparison.significance` (comparison.py) calls `paired_bootstrap(score_pairs(...))` with the defaults of `_bootstrap.paired_bootstrap` (`n_resamples=1000`, `seed=42`). The p-value is `(count + 1) / (n_resamples + 1)`, so the smallest value it can return is 0.000999. `Experiment.significance` takes `n_resamples`; `RunComparison.significance` does not.

In our use, 16 of 16 paired comparisons of 1,273 questions each returned exactly 0.000999. An exact McNemar test on the same pairs gives 5e-8 or less, so the floor, not the data, set the reported p.

**What a maintainer could change.** Let `RunComparison.significance` (and `ak.compare`) pass `n_resamples` and `seed` through, and report a floored value as "p < 1/(n+1)".

Paper: https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf (Appendix D). The script that reproduces the comparisons from saved per-question results: https://github.com/itssaharsh/relapse/blob/main/scripts/auditkit_headline.py
