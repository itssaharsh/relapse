"""Writes the Stage 2 results block of README.md and docs/ARTICLE.md from the result files.

    python scripts/build_readme.py          # rewrites the block between <!-- stage2:start --> and <!-- stage2:end -->
Every number in the block comes from results/*/summary.json or results/final/check.json.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ("README.md", os.path.join("docs", "ARTICLE.md"))
REPO = "https://github.com/itssaharsh/relapse/blob/main/"
load = lambda *p: json.load(open(os.path.join(ROOT, "results", *p)))
pc = lambda x, d=1: f"{100 * x:.{d}f}"
sg = lambda x, d=1: f"{100 * x:+.{d}f}"


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
    return f"""## Results (Stage 2: pre-registered, official tokenizer)

Every experiment below was registered in this repository before it ran ([1](results/PREREGISTRATION.md), [2](results/PREREGISTRATION_2.md), [3](results/PREREGISTRATION_3.md), [4](results/PREREGISTRATION_4.md), [5](results/PREREGISTRATION_5.md)), and every hypothesis that failed is listed.

**1. Forgetting usually survives the 4-bit file.** In a map of {len(m['runs'])} runs across the five Tiny Aya models, {held} of the {len(eff)} runs that forgot enough to count kept their forgetting in both Q4_0 and NF4.

**2. But one file can relapse while its siblings hold.** On three models, one 4-bit format brought back at least 5 points of hazardous knowledge while the other formats of the same weights did not:

| Run | fp16 | Q4_0 round trip | NF4 | Real Q4_0 file | Real Q4_K_M file |
|---|---|---|---|---|---|
{rows}
- The Base relapse reproduced when the seed-12 runs were retrained, and happened again on {A['R1']['relapsing_seeds_Q4_0']} of {A['R1']['of']} fresh seeds.
- Loaded in 4-bit through bitsandbytes, the way NF4 is really used, Fire and Water gave the same rises as above.
- Scored inside llama.cpp's own runtime on {rt['orig']['questions']} questions (an exploratory check on the first Base case), the answers matched our scoring on at least {100 * min(rt[k]['same_answer_as_pytorch']['f16'] for k in ('orig', 's12')):.1f}% of questions at full precision. On those questions Q4_0 raised the original model by {sg(rt['orig']['change'])} points and the unlearned one by {sg(rt['s12']['change'])}.
- No real Q4_K_M file relapsed. In every case at least one 4-bit format held, so scoring each file before release catches it.
- It is rare and modest, and on Base part of it is the format, not the unlearning: the original Tiny Aya Base also gains about 1.8 points in Q4_0. Net of that, no Base run reaches 5 points, and the typical Base run moves no more than the original does. Fire and Water are the cases that survive this check (their originals fall slightly in NF4).
- The 5-point line is a choice. Two of the three real Base files rose 4.5 and 4.6 points, just under it.

**3. The rise comes from weights the edit never touched.** Rounding only the three edited matrices adds at most 1.7 points in any relapsing run. On Water and Fire the rise comes from the layers in front of the edit ({part('nf4', 'water_W10-12_lr0.001_s42', 'before')} of {part('nf4', 'water_W10-12_lr0.001_s42', 'all')} points on Water). On Base seed 12 it comes from the layers behind it.

**4. Training on the file closes the gap, at a cost.** In the {len(rep)} relapsing runs, the retrained model no longer relapsed in {sum(x['R5a'] for _, x in rep)}, was significantly safer in the file in {sum(x['R5b'] for _, x in rep)}, and stayed within 3 MMLU points in {sum(x['R5c'] for _, x in rep)}. But plain RMU trained for the same extra time did almost as well ({ctl[0]}, {ctl[1]} and {ctl[2]} of {ctl[3]}), so we cannot separate the repair from longer training, and too much of either forgets far more than intended.

**5. A different tokenizer setting is the larger risk.** Of {len(old)} runs unlearned through the outdated tokenizer setting, {S['P1']['wins']} scored higher through the official one and {S['P1']['losses']} lower; {S['P1']['pipeline_relapses']} rose by at least 5 points. The average rise was {sg(mean_rise('5-7'))} points at SafeTune's default layers 5 to 7 and {sg(mean_rise('10-12'))} at layers 10 to 12, but {sg(mean_rise('7-9'))} at layers 7 to 9. Every Tiny Aya Base run trained this way scored {pc(np.mean([r['wmdp_outdated'] for r in base_old]))}% (chance) in its own pipeline and {pc(min(r['wmdp_official'] for r in base_old))}% to {pc(max(r['wmdp_official'] for r in base_old))}% through the official tokenizer. The direction matters: runs unlearned through the official tokenizer and scored through the outdated one rose by only {sg(-np.mean([r['rise'] for r in off]))} points on average ({sum(-r['rise'] >= 0.05 and r['p'] < 0.05 for r in off)} of {len(off)} relapsed).

**Where the edit sits decides it.** In a further registered test on new seeds, runs unlearned through the outdated tokenizer at SafeTune's default layers relapsed in {s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]} cases ({sg(s2['by_window']['5-7']['mean_rise'])} points on average). At layers 3 to 5, the window we release, {s2['T2']['relapses']} of {s2['T2']['effective']} relapsed ({sg(s2['by_window']['3-5']['mean_rise'])} on average), and at layers 7 to 9, {s2['T3']['7-9'][0]} of {s2['T3']['7-9'][1]}. Scored through the official tokenizer, the layers 3 to 5 runs sit at {pc(s2['by_window']['3-5']['mean_wmdp_official'])}% and the default-layer runs at {pc(s2['by_window']['5-7']['mean_wmdp_official'])}%. Tiny Aya Base is the exception: it drops to guessing level in the outdated pipeline at every window and comes back up through the official one.

**6. Where to edit.** Layers 3 to 5 beat the default layers in {q('Q2')['wins']} of 10 comparisons. CircuitKIT ranks that window first on three models and first when its attribution is pooled over the five; its pick for each model alone was unstable and that hypothesis failed (with 160 prompts instead of 64, an exploratory repeat, the per-model pick changed on three of five models while the pooled pick stayed layers 3 to 5). On two further seeds, layers 3 to 5 forgot {pc(min(-x['difference'] for x in won))} to {pc(max(-x['difference'] for x in won))} points more on four of the five models ({Q['lower_at_3_5']} of 10 comparisons, p = {Q['p']:.3f}, just short of our own line). The comparison holds general knowledge fixed; layers 3 to 5 cost more benign biology than the default layers do.

### The released models

Layers 3 to 5, two seeds per model (the table shows their mean). WMDP-bio is the hazard set (lower is better, 25% is guessing); the other columns should stay high.

| Model | WMDP-bio: original → unlearned | In the real Q4_K_M file | MMLU | Benign biology | With five worked examples | Through the outdated tokenizer | After a fine-tuning attack |
|---|---|---|---|---|---|---|---|
{models}
- **Holds:** prompt attacks do not bring the knowledge back, reworded and translated questions (Hindi, Swahili, Arabic, Spanish) stay significantly below the original in {sum(e['fp16'][k]['difference'] < 0 and e['fp16'][k]['p'] < 0.05 for e in E10 for k in probes)} of {len(E10) * len(probes)} cases, and neither the real files nor the outdated tokenizer changes the score by more than {max(abs(100 * e['real_files']['wmdp-bio'][f]['change']) for e in E10 for f in ('Q4_0', 'Q4_K_M')):.1f} points.
- **Costs:** benign biology falls by {pc(min(ben))} to {pc(max(ben))} points, so the forgetting is not confined to hazardous questions.
- **Breaks:** 25 fine-tuning steps on 64 hazard questions raise the score by at least 5 points on {sum(x >= 0.05 for x in ft)} of {len(E10)} models and restore Tiny Aya Base almost fully. RMU hides knowledge more than it removes it.

### Every registered hypothesis

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
    E10 = list(check.values())
    return f"""## In brief

SafeTune's default recipe looks unlearned where it was built and comes back when deployed. Editing at layers 3 to 5, the window CircuitKIT ranks first across the model family, forgets more and does not come back.

| Tiny Aya, RMU with SafeTune | Layers 5 to 7 (default) | Layers 3 to 5 (released) |
|---|---|---|
| Hazard score in the pipeline it was unlearned in | {pc(W['5-7']['mean_wmdp_outdated'])}% | {pc(W['3-5']['mean_wmdp_outdated'])}% |
| Hazard score once deployed through the official tokenizer | {pc(W['5-7']['mean_wmdp_official'])}% | **{pc(W['3-5']['mean_wmdp_official'])}%** |
| Runs that relapse through the tokenizer | {s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]} | **{s2['T2']['relapses']} of {s2['T2']['effective']}** |
| Hazard score, official tokenizer throughout | {pc(fin['Q4']['mean_wmdp_5_7'])}% | **{pc(fin['Q4']['mean_wmdp_3_5'])}%** |
| Relapses in Q4_0 or NF4 (of 20 checks) | {relapses['5-7']} | {relapses['3-5']} |
| General knowledge lost (MMLU points, mean) | {100 * np.mean(cost['5-7']):.1f} | {100 * np.mean(cost['3-5']):.1f} |
| Benign biology lost (points, median) | {100 * np.median(benign['5-7']):.1f} | {100 * np.median(benign['3-5']):.1f} |

The hazard score is accuracy on WMDP-bio (lower is better; the original models score 60% to 64%; 25% is guessing). The top three rows are runs unlearned through an outdated tokenizer setting and then run through the official one (16 runs per column, four models). The rest are the final runs on all five models.

- **In 4-bit files** the forgetting usually holds (107 of 109 runs in the map). On three models one format regained 5 to 8 points while the others held, so each file has to be scored.
- **What it costs:** benign biology falls too, more at layers 3 to 5. **What breaks it:** a short fine-tune on the topic.
- **What we got wrong:** our first-round numbers came from an outdated tokenizer setting. We corrected them in public, registered every new test before running it, and report the {FAILED} hypotheses that failed.
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
