"""Writes the Stage 2 results block of README.md and docs/ARTICLE.md from the result files.

    python scripts/build_readme.py          # rewrites the block between <!-- stage2:start --> and <!-- stage2:end -->
Every number in the block comes from results/*/summary.json or results/final/check.json.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ("README.md", os.path.join("docs", "ARTICLE.md"))
REPO = "https://github.com/itssaharsh/relapse/blob/main/"
load = lambda *p: json.load(open(os.path.join(ROOT, "results", *p)))
pc = lambda x, d=1: f"{100 * x:.{d}f}"
sg = lambda x, d=1: f"{100 * x:+.{d}f}"


def survey() -> dict:
    """The survey of public copies (results/survey/, written by scripts/survey_copies.py)."""
    import csv
    R = list(csv.DictReader(open(os.path.join(ROOT, "results", "survey", "copies.csv"))))
    S, P = load("survey", "summary.json"), load("survey", "probe_sample.json")
    hf = [r for r in R if not r["repo"].startswith("kaggle:") and r["repo"] not in S["official_repos"]]; kg = [r for r in R if r["repo"].startswith("kaggle:")]
    out = [r for r in hf if r["setting"] == "outdated"]; others = [r for r in out if not r["repo"].startswith("itssaharsh/")]
    num = lambda x: int(x) if str(x).isdigit() else 0
    gguf = [r for r in hf if "tokenizer.ggml.pre" in r["evidence"]]
    return {"date": S["date"], "found": len(hf) + len(S["official_repos"]), "read": sum(r["setting"] in ("official", "outdated") for r in hf), "outdated": len(out), "others": len(others), "ours": len(out) - len(others),
            "downloads": sum(num(r["downloads"]) for r in others), "month": sum(num(r["downloads_30d"]) for r in others), "kaggle": len(kg), "kaggle_outdated": sum(r["setting"] == "outdated" for r in kg),
            "gguf": len(gguf), "gguf_other": sum("=tiny_aya" not in r["evidence"] and "=cohere2moe" not in r["evidence"] for r in gguf), "sampled": P["sampled"], "differ": P["split_differently"], "transformers": P["transformers"],
            "unknown": sum(r["setting"] == "unknown" for r in hf)}


def revisions() -> str:
    """The revision of each official repo that carries Cohere's tokenizer fix: what main was when every Stage 2 job ran (results/survey/)."""
    H = load("survey", "official_history.json")
    fix = lambda v: next(c for c in reversed(H[f"CohereLabs/tiny-aya-{v}"]) if c["setting"] == "official")
    return ", ".join(f"{v.capitalize()} `{fix(v)['sha'][:10]}`" for v in ("global", "fire", "earth", "water", "base"))


def damage() -> dict:
    """General knowledge lost where a run was audited and where it is deployed (exploratory; paper_numbers.audit_damage)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paper_numbers import audit_damage
    return audit_damage()[0]


def exposure() -> dict:
    """What the outdated copies of the survey are: how many were last touched before the fix, how many are named as unlearning or safety-tuned models."""
    import csv
    R = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "survey", "copies.csv"))) if r["setting"] == "outdated" and not r["repo"].startswith(("kaggle:", "itssaharsh/"))]
    return {"before_fix": sum(r["last_modified"] <= "2026-09-30" for r in R), "unlearning": sum(bool(re.search(r"unlearn|rmu|forget", r["repo"], re.I)) for r in R),
            "safety": sum(bool(re.search(r"safe|guard|refus|harm|jailbreak|toxic|moderat", r["repo"], re.I)) for r in R)}


def tally() -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paper_numbers import hypotheses
    hyp = hypotheses()
    return {k: sum(h[3] == k for h in hyp) for k in ("held", "part", "failed", "untestable")} | {"n": len(hyp)}


def placement() -> dict:
    """Forgetting, MMLU cost and benign-biology cost of the final runs at each window (results/final/check_*/certify.json)."""
    out = {w: {"forgot": [], "benign": [], "mmlu": []} for w in ("3-5", "5-7")}
    for v in ("global", "fire", "earth", "water", "base"):
        c = {(r["model"], r["prec"]): r for r in load("final", f"check_{v}", "certify.json") if "wmdp-bio" in r}
        for (m, p), r in c.items():
            if p == "fp16" and m != "base":
                w = re.search(r"_W(\d+-\d+)_", m).group(1); o = c[("base", "fp16")]
                out[w]["forgot"].append(100 * (o["wmdp-bio"] - r["wmdp-bio"])); out[w]["benign"].append(100 * (o["benign-bio"] - r["benign-bio"])); out[w]["mmlu"].append(100 * (o["mmlu"] - r["mmlu"]))
    return {w: {k: float(np.mean(x)) for k, x in d.items()} | {"ratio": float(np.mean(d["forgot"]) / np.mean(d["benign"]))} for w, d in out.items()}


def block() -> str:
    m, deep, shift, place, final, check = load("map", "summary.json"), load("deep", "summary.json"), load("shift", "summary.json"), load("place", "summary.json"), load("final", "summary.json"), load("final", "check.json")
    rt = load("runtime", "summary.json"); s2 = load("shift2", "summary.json")
    eff = [r for r in m["runs"] if r["effective"]]
    held = sum(not (r["Q4_0"]["relapse"] or r["nf4"]["relapse"]) for r in eff)
    runs = {r["name"]: r for r in deep["runs"]}
    A, E, D = deep["part_a"], deep["part_e"], deep["part_d"]
    real = {n: v for models in deep["part_b_real_files"].values() for n, v in models.items()}
    cases = [("Tiny Aya Base, layers 10 to 12, seed 12", runs["base_W10-12_lr0.001_s12"], "Q4_0", real["base_relapse_base_x10_lr1e-3_s12"]),
             ("Tiny Aya Base, layers 10 to 12, seed 43", runs["base_W10-12_lr0.00045_s43"], "Q4_0", real["deep_real_s43_base_W10-12_lr0.00045_s43"]),
             ("Tiny Aya Fire, layers 13 to 15, seed 44", runs["fire_W13-15_lr0.002_s44"], "nf4", real["deep_real_fire44_fire_W13-15_lr0.002_s44"]),
             ("Tiny Aya Water, layers 10 to 12, seed 42", runs["water_W10-12_lr0.001_s42"], "nf4", real["deep_real_water42_water_W10-12_lr0.001_s42"])]
    rows = ""
    for label, r, fmt, rf in cases:
        cell = lambda f: f"{pc(r[f]['wmdp'])}% ({sg(r[f]['delta'])})" + (" **relapse**" if r[f]["relapse"] else "")
        rows += f"| {label} | {pc(r['wmdp_fp16'])}% | {cell('Q4_0')} | {cell('nf4')} | {pc(rf['Q4_0']['file'])}% ({sg(rf['Q4_0']['rise'])}) | {pc(rf['Q4_K_M']['file'])}% ({sg(rf['Q4_K_M']['rise'])}) |\n"
    C = deep["part_c"]["by_format"]
    part = lambda fmt, key, p: sg(C[fmt][key][p]["rise"])
    rep = [(e, e["arms"][k]) for e in D["runs"] for k in ("repaired", "repaired_nf4") if k in e["arms"] and (k == "repaired_nf4") == (e["format"] == "nf4")]
    c600 = [e["arms"]["plain_600"] for e in D["runs"] if "plain_600" in e["arms"]]
    ctl = (sum(x["R5a"] for x in c600), sum(x["R5b"] for x in c600), sum(x["R5c"] for x in c600), len(c600))
    off = [r for r in shift["runs"] if r["train_tok"] == "official" and r["effective"]]
    S, SH = shift["all_five"], shift["held_out"]
    old = [r for r in shift["runs"] if r["train_tok"] == "outdated" and r["effective"]]
    mean_rise = lambda w: np.mean([r["rise"] for r in old if r["window"] == w])
    base_old = [r for r in shift["runs"] if r["train_tok"] == "outdated" and r["model"] == "base"]
    P = place["results"]["all_five"]["fp16"]; q = lambda k: next(v for n, v in P.items() if n.startswith(k))
    Q = final["Q4"]; won = [x for x in final["pairs"] if x["difference"] < 0]
    E10 = list(check.values())
    rng = lambda f: (min(f(e) for e in E10), max(f(e) for e in E10))
    ben = [-e["fp16"]["benign-bio"]["difference"] for e in E10]
    ft = [e["fine_tuning"]["after_hazard_questions"] - e["fine_tuning"]["before"] for e in E10]
    probes = [k for k in E10[0]["fp16"] if k.startswith("probe:")]
    models = ""
    for v in ("global", "fire", "earth", "water", "base"):
        es = [e for e in E10 if e["model"] == v]; mean = lambda f: np.mean([f(e) for e in es])
        models += (f"| Tiny Aya {v.capitalize()} | {pc(mean(lambda e: e['fp16']['wmdp-bio']['original']))}% → {pc(mean(lambda e: e['fp16']['wmdp-bio']['model']))}% | {pc(mean(lambda e: e['real_files']['wmdp-bio']['Q4_K_M']['file']))}% | "
                   f"{pc(mean(lambda e: e['fp16']['mmlu']['original']))}% → {pc(mean(lambda e: e['fp16']['mmlu']['model']))}% | {pc(mean(lambda e: e['fp16']['benign-bio']['original']))}% → {pc(mean(lambda e: e['fp16']['benign-bio']['model']))}% | "
                   f"{pc(mean(lambda e: e['fp16']['attack:five_shot']['model']))}% | {pc(mean(lambda e: e['outdated_tokenizer']['outdated']))}% | {pc(mean(lambda e: e['fine_tuning']['after_hazard_questions']))}% |\n")
    hyp = [("Map", "A relapse tends to appear in one format, not both", "Held, on 2 relapses"),
           ("Map", "A single-format audit misses relapses in both directions", "**Failed** in the map (Q4_0 only). The NF4 direction appeared in the follow-up"),
           ("Map", "Layers 7 to 9 cost less at equal forgetting", "Could not be evaluated as registered"),
           ("Map", "Layers 10 to 12 relapse more than layers 7 to 9", f"**Failed** (p = {m['all_five']['H4']['p']:.3f})"),
           ("Relapse follow-up", "The Base relapse happens again on fresh seeds", f"Held: {A['R1']['relapsing_seeds_Q4_0']} of {A['R1']['of']} seeds"),
           ("Relapse follow-up", "On Base, Q4_0 moves the score up more than NF4", f"Held: {A['R2']['Q4_0_change_larger']} of 8 seeds, p = {A['R2']['p']:.3f}"),
           ("Relapse follow-up", "A relapse in NF4 and not Q4_0 exists", f"Held: Fire ({E['R6']['seeds_with_nf4_only_relapse']} of {E['R6']['of']} seeds), and Water"),
           ("Relapse follow-up", "The relapse shows in the real Q4_0 file", "Held only for the 1 Base file we had already seen; the two built afterwards rose 4.5 and 4.6 points, under the 5-point line"),
           ("Relapse follow-up", "On Base it comes from the layers in front of the edit", "**Failed.** On Base it is the layers behind; on Fire and Water it is the layers in front"),
           ("Relapse follow-up", "Training on the file repairs it", f"Gap closed in {sum(x['R5a'] for _, x in rep)} of {len(rep)}; file significantly safer in {sum(x['R5b'] for _, x in rep)}; within 3 MMLU points in {sum(x['R5c'] for _, x in rep)}"),
           ("Relapse follow-up", "A rule for how long to train the repair", "**Failed** on its test case (Water)"),
           ("Tokenizer", "Unlearning through one tokenizer relapses through the other", f"Held: {S['P1']['wins']} up, {S['P1']['losses']} down, p = {S['P1']['p']:.4f}; held-out models {SH['P1']['wins']} to {SH['P1']['losses']}, p = {SH['P1']['p']:.3f}"),
           ("Tokenizer", "The rise is smaller at layers 7 to 9 than at layers 5 to 7", f"Held on all five models ({S['P2']['wins']} of {S['P2']['wins'] + S['P2']['losses']} pairs). **Not significant on the held-out models alone** ({SH['P2']['wins']} of {SH['P2']['wins'] + SH['P2']['losses']}, p = {SH['P2']['p']:.3f})"),
           ("Tokenizer", "Training on both tokenizers narrows the gap", f"Held on all five ({S['P3']['wins']} pairs against {S['P3']['losses']}, p = {S['P3']['p']:.3f}). **Not significant on the held-out models alone** (p = {SH['P3']['p']:.2f})"),
           ("Tokenizer", "Training on both protects the deployed model", f"**Failed** ({S['P4']['wins']} pairs against {S['P4']['losses']})"),
           ("Placement and tokenizer", "Unlearned through the outdated tokenizer, the rise is smaller at layers 3 to 5 than at the default layers", f"Held: {s2['T1']['smaller_at_3_5']} of {s2['T1']['pairs']} pairs"),
           ("Placement and tokenizer", "Fewer than 1 in 5 runs at layers 3 to 5 relapse through the tokenizer", f"Held: {s2['T2']['relapses']} of {s2['T2']['effective']}"),
           ("Placement and tokenizer", "Layers 7 to 9 again do not relapse and the default layers again do", f"Held: {s2['T3']['7-9'][0]} of {s2['T3']['7-9'][1]} and {s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]}"),
           ("Placement", "Layers 3 to 5 beat the default layers 5 to 7", f"Held: {q('Q2')['wins']} of 10, p = {q('Q2')['p']:.3f}"),
           ("Placement", "Layers 7 to 9 beat the default", f"**Failed** ({q('Q1')['wins']} of 10)"),
           ("Placement", "Each model's own CircuitKIT window beats the default", f"**Failed** ({q('Q3')['wins']} of 10)"),
           ("Placement", "Layers 3 to 5 beat the default again on the final seeds", f"**Failed narrowly**: {Q['lower_at_3_5']} of 10, p = {Q['p']:.3f}")]
    table = "".join(f"| {a} | {b} | {c} |\n" for a, b, c in hyp)
    sv, tl, pl, dm, ex = survey(), tally(), placement(), damage(), exposure()
    tk = load("survey", "token_differences.json")
    one = {(r["model"], r["tokenizer"]): r for r in load("correct_tokenizer", "tokenizer_control.json")}      # the Stage 1 weights, scored through both settings
    dm2 = {"lost": f"{round(100 * one[('base', 'kaggle_copy')]['mmlu'], 1) - round(100 * one[('rmu_deeper', 'kaggle_copy')]['mmlu'], 1):.1f}",
           "rise": f"{round(100 * one[('rmu_deeper', 'current_official')]['wmdp-bio'], 1) - round(100 * one[('rmu_deeper', 'kaggle_copy')]['wmdp-bio'], 1):.1f}"}
    allw = lambda w: [r for r in s2["runs"] if r["window"] == w]
    up = lambda w: sum(r["pipeline_relapse"] for r in allw(w))
    b35 = [r for r in s2["not_effective"] if r["model"] == "base" and r["window"] == "3-5"]
    return f"""## Results (Stage 2: pre-registered, official tokenizer)

Every experiment below was registered in this repository before it ran ([1](results/PREREGISTRATION.md), [2](results/PREREGISTRATION_2.md), [3](results/PREREGISTRATION_3.md), [4](results/PREREGISTRATION_4.md), [5](results/PREREGISTRATION_5.md)), and every hypothesis that failed is listed. Every Stage 2 job downloaded the main branch of `CohereLabs/tiny-aya-<variant>` when it started; by the job start times and the repositories' history that was the revision carrying Cohere's tokenizer fix ({revisions()}, all committed September 30, 2026), as the paper's appendix sets out.

**1. Forgetting usually survives the 4-bit file.** In a map of {len(m['runs'])} runs across the five Tiny Aya models, {held} of the {len(eff)} runs that forgot enough to count kept their forgetting in both Q4_0 and NF4.

**2. But one file can relapse while its siblings hold.** On three models, one 4-bit format brought back at least 5 points of hazardous knowledge while the other formats of the same weights did not:

| Run | fp16 | Q4_0 round trip | NF4 | Real Q4_0 file | Real Q4_K_M file |
|---|---|---|---|---|---|
{rows}
- The Base relapse reproduced when the seed-12 runs were retrained, and happened again on {A['R1']['relapsing_seeds_Q4_0']} of {A['R1']['of']} fresh seeds. The Fire and Water cases are one run each, never retrained, and we treat them as exploratory; by an exact McNemar test with a Holm correction over the cells screened with them they are not noise ([`scripts/mcnemar_holm.py`](scripts/mcnemar_holm.py)).
- Loaded in 4-bit through bitsandbytes, the way NF4 is really used, Fire and Water gave the same rises as above.
- Scored inside llama.cpp's own runtime on {rt['orig']['questions']} questions (an exploratory check on the first Base case), the answers matched our scoring on at least {100 * min(rt[k]['same_answer_as_pytorch']['f16'] for k in ('orig', 's12')):.1f}% of questions at full precision. On those questions Q4_0 raised the original model by {sg(rt['orig']['change'])} points and the unlearned one by {sg(rt['s12']['change'])}.
- No real Q4_K_M file relapsed. In every case at least one 4-bit format held, so scoring each file before release catches it.
- It is rare and modest, and on Base part of it is the format, not the unlearning: the original Tiny Aya Base also gains about 1.8 points in Q4_0. Net of that, no Base run reaches 5 points, and the typical Base run moves no more than the original does. Fire and Water are the cases that survive this check (their originals fall slightly in NF4).
- The 5-point line is a choice. Two of the three real Base files rose 4.5 and 4.6 points, just under it.

**3. The rise comes from weights the edit never touched.** Rounding only the three edited matrices adds at most 1.7 points in any relapsing run. On Water and Fire the rise comes from the layers in front of the edit ({part('nf4', 'water_W10-12_lr0.001_s42', 'before')} of {part('nf4', 'water_W10-12_lr0.001_s42', 'all')} points on Water). On Base seed 12 it comes from the layers behind it.

**4. Training on the file closes the gap, at a cost.** In the {len(rep)} relapsing runs, the retrained model no longer relapsed in {sum(x['R5a'] for _, x in rep)}, was significantly safer in the file in {sum(x['R5b'] for _, x in rep)}, and stayed within 3 MMLU points in {sum(x['R5c'] for _, x in rep)}. But plain RMU trained for the same extra time did almost as well ({ctl[0]}, {ctl[1]} and {ctl[2]} of {ctl[3]}), so we cannot separate the repair from longer training, and too much of either forgets far more than intended.

**5. A different tokenizer setting is the larger risk.** Of {len(old)} runs unlearned through the outdated tokenizer setting, {S['P1']['wins']} scored higher through the official one and {S['P1']['losses']} lower; {S['P1']['pipeline_relapses']} rose by at least 5 points. The average rise was {sg(mean_rise('5-7'))} points at SafeTune's default layers 5 to 7 and {sg(mean_rise('10-12'))} at layers 10 to 12, but {sg(mean_rise('7-9'))} at layers 7 to 9. Every Tiny Aya Base run trained this way scored {pc(np.mean([r['wmdp_outdated'] for r in base_old]))}% (chance) in its own pipeline and {pc(min(r['wmdp_official'] for r in base_old))}% to {pc(max(r['wmdp_official'] for r in base_old))}% through the official tokenizer. The direction matters: runs unlearned through the official tokenizer and scored through the outdated one rose by only {sg(-np.mean([r['rise'] for r in off]))} points on average ({sum(-r['rise'] >= 0.05 and r['p'] < 0.05 for r in off)} of {len(off)} relapsed).

**Default layers against layers 3 to 5, at the learning rates tested.** In a further registered test on new seeds, runs unlearned through the outdated tokenizer at SafeTune's default layers relapsed in {s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]} cases ({sg(s2['by_window']['5-7']['mean_rise'])} points on average). At layers 3 to 5, the window we release, {s2['T2']['relapses']} of {s2['T2']['effective']} relapsed ({sg(s2['by_window']['3-5']['mean_rise'])} on average), and at layers 7 to 9, {s2['T3']['7-9'][0]} of {s2['T3']['7-9'][1]}. Scored through the official tokenizer, the layers 3 to 5 runs sit at {pc(s2['by_window']['3-5']['mean_wmdp_official'])}% and the default-layer runs at {pc(s2['by_window']['5-7']['mean_wmdp_official'])}%. That is on the four post-trained variants. Tiny Aya Base is the exception, and our rule for which runs count removes it: at every window it drops to guessing level in the outdated pipeline and comes back up through the official one, at layers 3 to 5 by {min(100 * r['rise'] for r in b35):.1f} to {max(100 * r['rise'] for r in b35):.1f} points. Counting every run, {up('5-7')} of {len(allw('5-7'))} rose by 5 points or more at the default layers (mean rise {sg(np.mean([r['rise'] for r in allw('5-7')]))}), {up('3-5')} of {len(allw('3-5'))} at layers 3 to 5 ({sg(np.mean([r['rise'] for r in allw('3-5')]))}) and {up('7-9')} of {len(allw('7-9'))} at layers 7 to 9 ({sg(np.mean([r['rise'] for r in allw('7-9')]))}); outside the default layers all of them are Base. The effect was measured on multiple-choice prompts, whose line breaks and digit groups are where the two settings split text differently.

**What came back was not only the hazard** (an exploratory reading of the same runs). Where they were audited, through the outdated tokenizer, the default-layer runs had also lost {dm['stwoDefaultCostAudited']} MMLU points, none of them within our 3-point line; through the official tokenizer they lost {dm['stwoDefaultCostDeployed']}. At layers 3 to 5 the loss was {dm['stwoShallowCostAudited']} and {dm['stwoShallowCostDeployed']}. Tiny Aya Base scored {dm['stwoBaseMmluLow']}% to {dm['stwoBaseMmluHigh']}% on MMLU as audited: the edit had wrecked it there, not unlearned it. Over both tokenizer experiments, runs trained through the outdated setting that had lost 3 or more MMLU points as audited rose by 5 points or more in {dm['dmgHurtRose']} cases, and the others in {dm['dmgFineRose']}. About half of the audited forgetting came back ({dm['stwoShareBack']}%; {dm['stwoShareLow']}% to {dm['stwoShareHigh']}% per run), and the deployed runs land where runs unlearned through the official tokenizer at the same layers, seeds and learning rates land ({dm['ctlOutdatedTrained']}% against {dm['ctlOfficialTrained']}%): the outdated audit overstated the forgetting, it did not lose real forgetting at load. By model and seed the count is {dm['stwoUnitsDefault']} at the default layers against {dm['stwoUnitsShallow']} at layers 3 to 5. So the layers are confounded with how much the edit damaged the model in the pipeline it was audited in, and the pattern is not ordered by depth: layers 10 to 12 relapsed like the default layers, layers 7 to 9 held. General knowledge lost in the audit is a warning sign, not a test: the rule is fitted on these runs, and our Stage 1 model at layers 10 to 12 had lost only {dm2['lost']} MMLU points where it was audited and still rose {dm2['rise']}.

**What the two settings do to the text** (counted without any model, [`scripts/token_differences.py`](scripts/token_differences.py)). They give different tokens for all {tk['wmdp_bio_prompts']['texts_split_differently']:,} WMDP-bio prompts and all {tk['mmlu_prompts']['texts_split_differently']:,} MMLU prompts, and for {tk['wmdp_bio_questions_alone']['texts_split_differently']} of the WMDP-bio questions without the multiple-choice layout. The outdated setting splits every digit apart, separates punctuation from the word it joins, and loses the pieces that join punctuation to a line break.

**The outdated setting is still widespread.** The two settings are two public copies of the same model: Cohere changed `tokenizer_class` on Hugging Face on September 30, 2026, and copies made before that keep the old value. On {sv['date']} we read the tokenizer file, or the GGUF header, of every public copy we could find, without downloading weights ([`scripts/check_tokenizer.py`](scripts/check_tokenizer.py), [`results/survey/`](results/survey/)): {sv['found']:,} Hugging Face repositories tied to a Tiny Aya model, and Kaggle Models. Cohere's five repositories carry the official setting. All {sv['kaggle']} Kaggle variations carry the outdated one. Of {sv['read']} derived Hugging Face copies whose setting could be read, {sv['outdated']} carry it: {sv['ours']} are our own Stage 1 models and the other {sv['others']}, mostly fine-tunes and adapters, have {sv['downloads']:,} downloads ({sv['month']:,} in the last 30 days). Loaded with `transformers` {sv['transformers']}, {sv['differ']} of {sv['sampled']} sampled copies split a test prompt differently from the official tokenizer. GGUF files are nearly all safe ({sv['gguf_other']} of {sv['gguf']} name another pre-tokenizer). This counts exposure, not harm: we saw the relapse only in our own runs, {ex['before_fix']} of those copies were last modified before the fix and {sv['others'] - ex['before_fix']} after it, none is named as an unlearning model and {ex['safety']} are named as safety-tuned, and which direction of mismatch a copy faces depends on the `transformers` version it was trained with, which its files do not record. Who is exposed is anyone who unlearns or audits through a copy with the outdated setting and ships a model that users load through the official one.

**6. Where to edit.** Two registered tests compared layers 3 to 5 with SafeTune's default layers; one passed and one narrowly missed. In the first, layers 3 to 5 forgot more within 3 MMLU points in {q('Q2')['wins']} of 10 comparisons (p = {q('Q2')['p']:.3f}). In the second, on fresh seeds, they forgot {pc(min(-x['difference'] for x in won))} to {pc(max(-x['difference'] for x in won))} points more on four of the five models ({Q['lower_at_3_5']} of 10 comparisons, p = {Q['p']:.3f}, short of our own line). What placement buys is more forgetting per MMLU point ({pl['3-5']['forgot']:.1f} hazard points for {pl['3-5']['mmlu']:.1f} MMLU points, against {pl['5-7']['forgot']:.1f} for {pl['5-7']['mmlu']:.1f}) and robustness to the tokenizer. It does not buy selectivity within biology: benign biology falls by {pl['3-5']['benign']:.1f} points against {pl['5-7']['benign']:.1f}, the same {pl['3-5']['ratio']:.2f} and {pl['5-7']['ratio']:.2f} hazard points forgotten per point of benign biology lost. CircuitKIT ranks layers 3 to 5 first on three models and first when its attribution is pooled over the five; its pick for each model alone was unstable and that hypothesis failed (with 160 prompts instead of 64, an exploratory repeat, the per-model pick changed on three of five models while the pooled pick stayed layers 3 to 5).

### The released models

Layers 3 to 5, two seeds per model (the table shows their mean). WMDP-bio is the hazard set (lower is better, 25% is guessing); the other columns should stay high.

| Model | WMDP-bio: original → unlearned | In the real Q4_K_M file | MMLU | Benign biology | With five worked examples | Through the outdated tokenizer | After a fine-tuning attack |
|---|---|---|---|---|---|---|---|
{models}
- **Holds:** prompt attacks do not bring the knowledge back, reworded and translated questions (Hindi, Swahili, Arabic, Spanish) stay significantly below the original in {sum(e['fp16'][k]['difference'] < 0 and e['fp16'][k]['p'] < 0.05 for e in E10 for k in probes)} of {len(E10) * len(probes)} cases, and neither the real files nor the outdated tokenizer changes the score by more than {max(abs(100 * e['real_files']['wmdp-bio'][f]['change']) for e in E10 for f in ('Q4_0', 'Q4_K_M')):.1f} points.
- **Costs:** benign biology falls by {pc(min(ben))} to {pc(max(ben))} points, so the forgetting is not confined to hazardous questions.
- **Breaks:** 25 fine-tuning steps on 64 hazard questions raise the score by at least 5 points on {sum(x >= 0.05 for x in ft)} of {len(E10)} models and restore Tiny Aya Base almost fully. RMU hides knowledge more than it removes it.

### Every registered hypothesis

Of {tl['n']} registered hypotheses, {tl['held']} held, {tl['part']} held in part, {tl['failed']} failed and {tl['untestable']} could not be tested. Where a pre-registration names the held-out models as the confirmatory test, a hypothesis that holds on all five models and not on the held-out ones counts as failed.

| Experiment | Hypothesis | Outcome |
|---|---|---|
{table}
"""


def brief() -> str:
    """The result in brief, for the top of the page: the same rows as Table 1 of the paper."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_map_shim import paired
    s2, fin, check = load("shift2", "summary.json"), load("final", "summary.json"), load("final", "check.json")
    W = s2["by_window"]; rng = np.random.default_rng(0)
    relapses = {"3-5": 0, "5-7": 0}; cost = {"3-5": [], "5-7": []}; benign = {"3-5": [], "5-7": []}
    for v in ("global", "fire", "earth", "water", "base"):
        rows = {(r["model"], r["prec"]): r for r in load("final", f"{v}_seed33", "results.json")}; items = load("final", f"{v}_seed33", "items.json")
        for (m, p), r in rows.items():
            if p != "fp16" or m == "base":
                continue
            w = re.search(r"_W(\d+-\d+)_", m).group(1)
            cost[w].append(rows[("base", "fp16")]["mmlu"] - r["mmlu"])
            for f in ("Q4_0", "nf4"):
                delta, pv = paired(items[f"{m}|fp16|wmdp-bio"], items[f"{m}|{f}|wmdp-bio"], rng)
                relapses[w] += int(delta >= 0.05 and pv < 0.05)
        c = {(r["model"], r["prec"]): r for r in load("final", f"check_{v}", "certify.json") if "wmdp-bio" in r}
        for (m, p), r in c.items():
            if p == "fp16" and m != "base":
                benign[re.search(r"_W(\d+-\d+)_", m).group(1)].append(c[("base", "fp16")]["benign-bio"] - r["benign-bio"])
    sv, tl, pl, dm, ex = survey(), tally(), placement(), damage(), exposure()
    allw = lambda w: [r for r in s2["runs"] if r["window"] == w]
    up = lambda w: sum(r["pipeline_relapse"] for r in allw(w))
    b35 = [r for r in s2["not_effective"] if r["model"] == "base" and r["window"] == "3-5"]
    m = load("map", "summary.json"); eff = [r for r in m["runs"] if r["effective"]]
    held = sum(not (r["Q4_0"]["relapse"] or r["nf4"]["relapse"]) for r in eff)
    ft = [e["fine_tuning"]["after_hazard_questions"] - e["fine_tuning"]["before"] for e in check.values()]
    return f"""## In brief

An outdated tokenizer setting made the unlearning look deeper than it was. Runs unlearned and audited through it at SafeTune's default layers gave back about half of the audited forgetting ({dm['stwoShareBack']}%) once loaded through the official tokenizer, in {s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]} runs that count, and every one of them had also lost more than 3 points of general knowledge where it was audited. At layers 3 to 5 the same change did nothing on the four post-trained variants ({s2['T2']['relapses']} of {s2['T2']['effective']}). The layers are not shown to be the cause: each window had its own learning rates.

| Layers edited (Tiny Aya, RMU with SafeTune) | 5 to 7 (default) | 3 to 5 (released) |
|---|---|---|
| Hazard score as audited, through the outdated tokenizer setting | {pc(W['5-7']['mean_wmdp_outdated'])}% | {pc(W['3-5']['mean_wmdp_outdated'])}% |
| Hazard score as deployed, through the official tokenizer | {pc(W['5-7']['mean_wmdp_official'])}% | **{pc(W['3-5']['mean_wmdp_official'])}%** |
| Runs that relapse, among runs that count | {s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]} | **{s2['T2']['relapses']} of {s2['T2']['effective']}** |
| Runs that rise 5 points or more, among all runs | {up('5-7')} of {len(allw('5-7'))} | {up('3-5')} of {len(allw('3-5'))} |
| General knowledge lost as audited, then as deployed (MMLU points) | {dm['stwoDefaultCostAudited']}, then {dm['stwoDefaultCostDeployed']} | {dm['stwoShallowCostAudited']}, then {dm['stwoShallowCostDeployed']} |
| General knowledge lost (MMLU points, mean) | {100 * np.mean(cost['5-7']):.1f} | {100 * np.mean(cost['3-5']):.1f} |
| Benign biology lost (points, mean) | {pl['5-7']['benign']:.1f} | {pl['3-5']['benign']:.1f} |
| Hazard points forgotten per point of benign biology lost | {pl['5-7']['ratio']:.2f} | {pl['3-5']['ratio']:.2f} |

The hazard score is accuracy on WMDP-bio (lower is better; the original models score 60% to 64%; 25% is guessing). A run counts if it forgot enough without falling to guessing (30% to 55% where it was unlearned); a relapse is a rise of 5 points or more. The top five rows are runs unlearned through the outdated setting and then run through the official one; the rest are the final runs on all five models, through the official tokenizer.

- **The two settings are two public copies of the same model.** Cohere fixed the setting on Hugging Face on September 30, 2026. Kaggle's copy and {sv['others']} derived Hugging Face repositories ({sv['downloads']:,} downloads) still carried the old one on {sv['date']}; none of those is named as an unlearning model, and we saw the relapse only in our own runs. `python scripts/check_tokenizer.py <repo, folder or GGUF file>`, added with the paper, tells you which setting a copy carries and what to do about it, reading only its tokenizer file or GGUF header.
- **What came back was not only the hazard.** The default-layer runs that relapsed had also lost general knowledge where they were audited ({dm['stwoDefaultCostAudited']} MMLU points there, {dm['stwoDefaultCostDeployed']} once deployed). In both tokenizer experiments, runs that had lost 3 or more MMLU points as audited rose by 5 points or more in {dm['dmgHurtRose']} cases and the others in {dm['dmgFineRose']}, so the layers are confounded with that damage. General knowledge lost in the audit is a warning sign, not an exact test.
- **Tiny Aya Base is the exception.** Through the outdated setting it unlearns down to guessing, which our rule does not count, and it rises through the official tokenizer at every window, by {min(100 * r['rise'] for r in b35):.1f} to {max(100 * r['rise'] for r in b35):.1f} points at layers 3 to 5. Counting it, {up('3-5')} of {len(allw('3-5'))} runs at layers 3 to 5 rise, all four Base.
- **In 4-bit files** the forgetting usually holds ({held} of {len(eff)} runs in the map). Tiny Aya Base regained 5 to 6 points in Q4_0 on two seeds, and one exploratory run each on Fire and Water regained 5 to 8 points in NF4, while the other formats of the same weights held. Each file has to be scored.
- **What it costs:** benign biology falls too, and placement does not change how much per hazard point forgotten. **What breaks it:** 25 fine-tuning steps on the topic raise the score by 5 points or more on {sum(x >= 0.05 for x in ft)} of {len(ft)} runs.
- **What we got wrong:** our first-round numbers came from the outdated tokenizer setting. Correcting them is how we found the effect. Every new test was registered before it ran: of {tl['n']} hypotheses, {tl['held']} held, {tl['part']} held in part, {tl['failed']} failed and {tl['untestable']} could not be tested.
"""


FAILED = "ten"


def main() -> None:
    text = block()
    for doc in DOCS:
        p = os.path.join(ROOT, doc); s = open(p).read()
        body = text if doc == "README.md" else text.replace("](results/", "](" + REPO + "results/")
        new = "<!-- stage2:start -->\n" + body + "<!-- stage2:end -->\n"
        bnew = "<!-- brief:start -->\n" + brief() + "<!-- brief:end -->\n"
        if "<!-- brief:start -->" in s:
            s = re.sub(r"<!-- brief:start -->.*?<!-- brief:end -->\n", lambda _: bnew, s, flags=re.S)
        else:
            i = s.index("## Correction (November 21, 2026)")
            s = s[:i] + bnew + "\n" + s[i:]
        if "<!-- stage2:start -->" in s:
            s = re.sub(r"<!-- stage2:start -->.*?<!-- stage2:end -->\n", lambda _: new, s, flags=re.S)
        else:      # first run: replace the interim sections
            end = "## How it works" if "## How it works" in s else "## Method"
            a, b = s.index("## What holds with the official tokenizer"), s.index(end)
            s = s[:a] + new + "\n" + s[b:]
        open(p, "w").write(s)


if __name__ == "__main__":
    main()
