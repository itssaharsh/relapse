TITLE: Schema, length and deduplication gates passed all 707 machine-translated probes; none of them tests translation fidelity

**Setting.** CuratorKIT 0.1.24, run per language on five probe sets: schema gate, length limits (5 to 2,048 tokens), exact and MinHash deduplication (0.9), cleaning. The sets are WMDP-bio questions reworded in English or translated into Hindi, Arabic, Spanish and Swahili by a 3.35B model, 707 items after our own gate (a back-translation chrF of at least 35 against the source, or for rewordings that they parse and differ from the source).

**What we saw.** CuratorKIT removed none of the 707 items (187, 186, 119, 114 and 101 per set before and after). Our own gate had removed the rest of the 1,000 candidates before CuratorKIT ran. No native speaker checked the items, so we cannot say how many of the 707 are faithful.

**What a maintainer could change.** An optional fidelity gate for translated or paraphrased sets (for example a back-translation score against a source column), and a per-gate count in the manifest even when a gate removes nothing.

Paper: https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf (Appendix C). The curation script: https://github.com/itssaharsh/relapse/blob/main/scripts/curate_probes.py
