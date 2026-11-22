"""Writes the paper's numbers and tables from the result files, so nothing in the paper is typed by hand.

    python scripts/paper_numbers.py
Creates paper/numbers.tex and paper/table_*.tex. A table whose experiment has no results yet is skipped.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
WINDOWS = ("3-5", "5-7", "7-9", "10-12", "13-15")
NAME = re.compile(r"(?P<model>[a-z]+)_W(?P<a>\d+)-(?P<b>\d+)_lr(?P<lr>[^_]+)(?:_(?P<arm>[a-z]+)_nb\d+)?_s(?P<seed>\d+)$")


def signed(x: float, d: int = 1) -> str:
    x = round(x, d) + 0.0
    return f"$-${abs(x):.{d}f}" if x < 0 else f"+{x:.{d}f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def lr_text(lr: str) -> str:
    mant, exp = f"{float(lr):.1e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def describe(name: str) -> str:
    m = NAME.match(name)
    return f"{m['model'].capitalize()}, seed {m['seed']}, lr {lr_text(m['lr'])}"


def table(columns: str, head: str, body: str, caption: str, label: str, sep: str = "3.5pt", size: str = "\\small") -> str:
    return ("\\begin{table}[t]\n\\centering" + size + "\\setlength{\\tabcolsep}{" + sep + "}\n\\begin{tabular}{" + columns + "}\n\\toprule\n" + head + "\\midrule\n" + body +
            "\\bottomrule\n\\end{tabular}\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{table}\n")


def control() -> tuple[dict, dict]:
    rows = json.load(open(os.path.join(ROOT, "results", "correct_tokenizer", "tokenizer_control.json")))
    get = lambda m, t: next(100 * r["wmdp-bio"] for r in rows if r["model"] == m and r["tokenizer"] == t)
    names = (("base", "Original"), ("rmu_default_layers", "RMU, layers 5--7"), ("rmu_circuit_layers", "RMU, layers 7--9"), ("rmu_deeper", "RMU, layers 10--12"))
    body = "".join(f"{label} & {get(m, 'kaggle_copy'):.1f} & {get(m, 'current_official'):.1f} & {signed(get(m, 'current_official') - get(m, 'kaggle_copy'))} \\\\\n" for m, label in names)
    tex = table("lrrr", "Same weights & Outdated & Official & Change \\\\\n", body,
                r"\wmdp{} accuracy (\%) of the same weights on Global, scored in one job through each tokenizer setting. The weights were trained through the outdated setting. Lower is better.", "tab:control", "6pt")
    return {"ctlMaxRise": f"{max(get(m, 'current_official') - get(m, 'kaggle_copy') for m, _ in names):.1f}"}, {"table_control.tex": tex}


def relapse_map() -> tuple[dict, dict]:
    m = json.load(open(os.path.join(ROOT, "results", "map", "summary.json")))
    eff = [r for r in m["runs"] if r["effective"]]
    count = lambda rs, f: sum(bool(r.get(f, {}).get("relapse")) for r in rs)
    body = ""
    for w in WINDOWS:
        rs = [r for r in eff if r["window"] == w]
        body += f"{w.replace('-', '--')} & {len(rs)} & {count(rs, 'Q4_0')} & {count(rs, 'nf4')} \\\\\n"
    body += f"\\midrule\nAll & {len(eff)} & {count(eff, 'Q4_0')} & {count(eff, 'nf4')} \\\\\n"
    tex = table("lrrr", r"Layers & Effective runs & \fq{} relapses & \nf{} relapses \\" + "\n", body,
                r"The pre-registered map, through the official tokenizer. A relapse is a rise of at least 5 points over fp16 with paired $p<0.05$.", "tab:map", "4.5pt")
    delta = lambda f: np.array([100 * r[f]["delta"] for r in eff])
    rel = [(r, f) for r in eff for f in ("Q4_0", "nf4") if r[f]["relapse"]]
    shallow = [r for r in eff if r["window"] in ("3-5", "5-7", "7-9")]
    h, ho = m["all_five"], m["held_out"]
    sig = lambda f, up: sum((r[f]["delta"] > 0) == up and r[f]["p"] < 0.05 for r in eff)
    nums = {"mapRuns": len(m["runs"]), "mapEffective": len(eff), "mapHeld": len(eff) - len({id(r) for r, _ in rel}), "mapRelapses": len(rel), "mapShallow": len(shallow),
            "mapMeanQ": signed(delta("Q4_0").mean()), "mapMeanN": signed(delta("nf4").mean()), "mapSigUp": sig("Q4_0", True) + sig("nf4", True), "mapSigDown": sig("Q4_0", False) + sig("nf4", False),
            "mapRelapseLow": f"{min(100 * r[f]['delta'] for r, f in rel):.1f}", "mapRelapseHigh": f"{max(100 * r[f]['delta'] for r, f in rel):.1f}",
            "mapHfourA": f"{h['H4']['10_12'][0]} of {h['H4']['10_12'][1]}", "mapHfourB": f"{h['H4']['7_9'][0]} of {h['H4']['7_9'][1]}", "mapHfourP": f"{h['H4']['p']:.3f}",
            "mapHeldHfourA": f"{ho['H4']['10_12'][0]} of {ho['H4']['10_12'][1]}", "mapHeldHfourB": f"{ho['H4']['7_9'][0]} of {ho['H4']['7_9'][1]}", "mapHeldHfourP": f"{ho['H4']['p']:.2f}",
            "mapHthreeN": h["H3"]["not_comparable"]}
    return nums, {"table_map.tex": tex}


def net_of_original() -> dict:
    """The seed-12 rise in Q4_0 minus the original model's own change in Q4_0, with a bootstrap over questions."""
    it = json.load(open(os.path.join(ROOT, "results", "map", "base_seed12", "items.json")))
    rng = np.random.default_rng(0)
    orig = np.array(it["base|Q4_0|wmdp-bio"]) - np.array(it["base|fp16|wmdp-bio"])
    out = []
    for lr in ("0.0002", "0.00045", "0.001"):
        n = f"base_W10-12_lr{lr}_s12"
        dd = (np.array(it[f"{n}|Q4_0|wmdp-bio"]) - np.array(it[f"{n}|fp16|wmdp-bio"])) - orig
        boots = dd[rng.integers(0, len(dd), size=(10000, len(dd)))].mean(1)
        out.append((100 * dd.mean(), 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return {"netLow": f"{min(x for x, _ in out):.1f}", "netHigh": f"{max(x for x, _ in out):.1f}", "netP": f"{max(p for _, p in out):.3f}", "origQ": signed(100 * orig.mean())}


def deep() -> tuple[dict, dict]:
    """Numbers and tables of the relapse follow-up (results/PREREGISTRATION_4.md)."""
    path = os.path.join(ROOT, "results", "deep", "summary.json")
    if not os.path.exists(path):
        return {}, {}
    d = json.load(open(path)); runs = d["runs"]; A, E, D = d["part_a"], d["part_e"], d["part_d"]
    fresh = [r for r in runs if r["model"] == "base" and r["arm"] == "plain" and r["source"] == "new" and r["window"] == "10-12" and 41 <= r["seed"] <= 48]
    eff = [r for r in fresh if r["effective"]]
    fire = [r for r in runs if r["model"] == "fire" and r["window"] == "13-15" and r["arm"] == "plain" and r["effective"]]
    rerun = sorted((r for r in runs if r["arm"] == "rerun"), key=lambda r: float(r["lr"]))
    nums = {"deepFreshRuns": len(fresh), "deepFreshEff": len(eff), "deepRoneK": A["R1"]["relapsing_seeds_Q4_0"], "deepRoneN": A["R1"]["of"],
            "deepRoneLo": pct(A["R1"]["interval95"][0], 0), "deepRoneHi": pct(A["R1"]["interval95"][1], 0), "deepRtwoWins": A["R2"]["Q4_0_change_larger"], "deepRtwoP": f"{A['R2']['p']:.3f}",
            "deepSigUpQ": sum(r["Q4_0"]["delta"] > 0 and r["Q4_0"]["p"] < 0.05 for r in eff), "deepSigDownN": sum(r["nf4"]["delta"] < 0 and r["nf4"]["p"] < 0.05 for r in eff),
            "deepMeanQ": signed(100 * np.mean([r["Q4_0"]["delta"] for r in eff])), "deepMeanN": signed(100 * np.mean([r["nf4"]["delta"] for r in eff])),
            "rerunRises": ", ".join(signed(100 * r["Q4_0"]["delta"]) for r in rerun),
            "fireEff": len(fire), "fireUpN": sum(r["nf4"]["delta"] > 0 for r in fire), "fireDownQ": sum(r["Q4_0"]["delta"] < 0 for r in fire),
            "fireMeanQ": signed(100 * np.mean([r["Q4_0"]["delta"] for r in fire])) if fire else "", "fireMeanN": signed(100 * np.mean([r["nf4"]["delta"] for r in fire])) if fire else "",
            "fireRsixK": E["R6"]["seeds_with_nf4_only_relapse"], "fireRsixN": E["R6"]["of"], **net_of_original()}
    tables = {}
    # every run that relapses in a format, the seed-12 runs and their retraining
    rows = [r for r in runs if r["arm"] in ("plain", "rerun") and r["effective"] and (r["arm"] == "rerun" or r["Q4_0"]["relapse"] or r["nf4"]["relapse"] or (r["seed"] == 12 and r["window"] == "10-12"))]
    cell = lambda r, f: f"{pct(r[f]['wmdp'])} ({signed(100 * r[f]['delta'])})" + ("$^{*}$" if r[f]["relapse"] else "")
    compact = lambda r: f"{r['model'].capitalize()} {r['seed']}, lr {lr_text(r['lr'])}" + ("$^{\\dagger}$" if r["arm"] == "rerun" else "")
    body = "".join(f"{compact(r)} & {pct(r['wmdp_fp16'])} & {cell(r, 'Q4_0')} & {cell(r, 'nf4')} \\\\\n"
                   for r in sorted(rows, key=lambda r: (r["model"], r["seed"], r["arm"], float(r["lr"]))))
    tables["table_cases.tex"] = table("@{}lrrr@{}", r"Model, seed, lr & fp16 & \fq{} & \nf{} \\" + "\n", body,
                                      r"\wmdp{} accuracy (\%) of the same weights at fp16 and after each 4-bit round trip, with the change in points. $^{*}$ marks a relapse; $^{\dagger}$ marks a seed-12 run retrained with the same settings on an L4 GPU. Base and Water are edited at layers 10--12, Fire at layers 13--15.", "tab:cases", "3pt")
    # real llama.cpp files, compared by AuditKIT
    real = [(n, v) for models in d["part_b_real_files"].values() for n, v in models.items()]
    if real:
        keep = [(t, n, v) for t, models in d["part_b_real_files"].items() if t != "deep_ladder_s12" for n, v in models.items() if not n.endswith("_base") and "file_nb" not in n and "Q4_K_M" in v]
        def short(n):
            m = NAME.search(n.replace("_x10", "_W10-12").replace("lr2e-4", "lr0.0002").replace("lr1e-3", "lr0.001"))
            return f"{m['model'].split('_')[-1].capitalize()} {m['seed']}, {lr_text(m['lr'])}"
        rc = lambda v, f: f"{pct(v[f]['file'])} ({signed(100 * v[f]['rise'])})" + ("$^{*}$" if v[f]["relapse"] else "") if f in v else "--"
        body = "".join(f"{short(n)} & {pct(v['Q4_0']['fp16'])} & {rc(v, 'Q4_0')} & {rc(v, 'Q4_K_M')} & {rc(v, 'nf4')} \\\\\n" for t, n, v in sorted(keep, key=lambda x: short(x[1])))
        tables["table_real.tex"] = table("@{}lrrrr@{}", r" & & \multicolumn{2}{c}{Real file} & Round trip \\" + "\n" + r"\cmidrule(lr){3-4}\cmidrule(l){5-5}" + "\n" + r"Model, seed, lr & fp16 & \fq{} & \qkm{} & \nf{} \\" + "\n", body,
                                         r"Real llama.cpp files of every relapsing run of the map and the fresh seeds, and of the other Fire run of seed 44, built in the same job: \wmdp{} accuracy (\%) and change against fp16, compared with AuditKIT's paired bootstrap. \nf{} is the round trip, which the bitsandbytes loader reproduces. $^{*}$ marks a relapse.", "tab:real", "1.8pt", "\\footnotesize")
        others = sorted(100 * v["Q4_0"]["rise"] for t, n, v in keep if "base" in n.lower() and not v["Q4_0"]["relapse"])
        nums["realOtherRises"] = " and ".join(f"{x:.1f}" for x in others)
        rep_real = [v for t, models in d["part_b_real_files"].items() for n, v in models.items() if "file_nb600" in n]
        if rep_real:
            nums.update(repRealRise=signed(100 * rep_real[0]["Q4_0"]["rise"]), repRealP=f"{rep_real[0]['Q4_0']['p']:.2f}", repRealFile=pct(rep_real[0]["Q4_0"]["file"]))
        for n, v in real:
            key = "realOrig" if n.endswith("_base") else "realA" if "1e-3" in n and "s12" in n else "realB" if "2e-4" in n and "s12" in n and "file" not in n else None
            if key:
                nums.update({key + "Fp": pct(v["Q4_0"]["fp16"]), key + "File": pct(v["Q4_0"]["file"]), key + "Rise": signed(100 * v["Q4_0"]["rise"]), key + "P": f"{v['Q4_0']['p']:.3f}"})
    # where the rise comes from: one part of the model rounded at a time, for every relapse case
    C = d["part_c"]["by_format"]
    parts = (("before", "In front of the edit"), ("edited", "Edited matrices"), ("window_other", "Rest of the window"), ("after", "Behind the edit"), ("all", "Whole model"))
    cases = []
    if "Q4_0" in C:
        s12 = [k for k in C["Q4_0"] if k.endswith("_s12") and all(p in C["Q4_0"][k] for p, _ in parts)]
        if s12:
            cases.append(("Base 12", {p: np.mean([100 * C["Q4_0"][k][p]["rise"] for k in s12]) for p, _ in parts}))
    for fmt, key, label in (("Q4_0", "base_W10-12_lr0.00045_s43", "Base 43"), ("nf4", "fire_W13-15_lr0.002_s44", "Fire 44"), ("nf4", "water_W10-12_lr0.001_s42", "Water 42")):
        if key in C.get(fmt, {}) and all(p in C[fmt][key] for p, _ in parts):
            cases.append((label, {p: 100 * C[fmt][key][p]["rise"] for p, _ in parts}))
    if cases:
        body = "".join(f"{label} & " + " & ".join(signed(v[p]) for _, v in cases) + " \\\\\n" for p, label in parts)
        tables["table_cause.tex"] = table("@{}l" + "r" * len(cases) + "@{}", r" & \multicolumn{2}{c}{\fq{}} & \multicolumn{2}{c}{\nf{}} \\" + "\nRounded part & " + " & ".join(n for n, _ in cases) + " \\\\\n", body,
                                          r"Change in \wmdp{} (points) when only one part of the model is rounded to the format that relapses. Columns are the relapsing runs (model and seed; Base 12 is the mean of its three runs).", "tab:cause", "2.5pt", "\\footnotesize")
        nums.update(causeEditedMax=f"{max(v['edited'] for _, v in cases):.1f}", causeCases=len(cases) + 2)
        for n, v in cases:
            nums["cause" + n.replace(" ", "").replace("12", "A").replace("43", "B").replace("44", "").replace("42", "")] = f"{signed(v['before'])} in front, {signed(v['after'])} behind"
    # NF4 as bitsandbytes loads it
    for v in ("fire", "water"):
        pth = os.path.join(ROOT, "results", "deep", f"real_deep_bnb_{v}", "bnb.json")
        if os.path.exists(pth):
            by = {(r["model"], r["prec"]): r for r in json.load(open(pth))}
            name = next(m for m, _ in by if m != "base")
            nums.update({f"bnb{v.capitalize()}Fp": pct(by[(name, "fp16")]["wmdp-bio"]), f"bnb{v.capitalize()}File": pct(by[(name, "nf4_bitsandbytes")]["wmdp-bio"]),
                         f"bnb{v.capitalize()}Rise": signed(100 * (by[(name, "nf4_bitsandbytes")]["wmdp-bio"] - by[(name, "fp16")]["wmdp-bio"]))})
    qkm = [v["Q4_K_M"] for _, v in real if "Q4_K_M" in v and not _.endswith("_base")] if real else []
    nums.update(qkmFiles=len(qkm), qkmRelapses=sum(x["relapse"] for x in qkm), qkmMax=signed(max(100 * x["rise"] for x in qkm)) if qkm else "")
    # the repair: every relapsing run with its arms
    labels = (("repaired", "Trained on the file, 600"), ("repaired_nf4", "Trained on the file, 600"), ("plain_600", "Plain RMU, 600"), ("file_300", "Trained on the file, 300"), ("file_nf4_300", "Trained on the file, 300"), ("edited_only", "Edited matrices rounded"))
    body, rep = "", []
    for e in D["runs"]:
        if not e["arms"]:
            continue
        body += f"\\multicolumn{{4}}{{l}}{{\\emph{{{describe(e['plain'])}, in {'NF4' if e['format'] == 'nf4' else 'Q4\\_0'}}}}} \\\\\nPlain RMU, 300 & {pct(e['plain_fp16'])} & {pct(e['plain_file'])} ({signed(100 * e['plain_rise'])})$^{{*}}$ & \\\\\n"
        for k, label in labels:
            if k in e["arms"]:
                x = e["arms"][k]
                body += f"{label} & {pct(x['fp16'])} & {pct(x['file'])} ({signed(100 * x['own_rise'])})" + ("$^{*}$" if x["relapses"] else "") + f" & {signed(100 * x['mmlu_file_vs_plain']['delta'])} \\\\\n"
        rep += [e["arms"][k] for k in ("repaired", "repaired_nf4") if k in e["arms"] and (k == "repaired_nf4") == (e["format"] == "nf4")]
    if body:
        tables["table_repair.tex"] = table("lrrr", r"Training (batches) & fp16 & 4-bit file & MMLU \\" + "\n", body,
                                           r"The repair and its controls, each with the seed and learning rate of the relapsing run. \wmdp{} (\%) at fp16 and in the relapsing format, with the change in points; MMLU is the difference from the plain run in that format. $^{*}$ marks a relapse.", "tab:repair", "3.5pt", "\\footnotesize")
    nums.update(repN=len(rep), repA=sum(x["R5a"] for x in rep), repB=sum(x["R5b"] for x in rep), repC=sum(x["R5c"] for x in rep))
    return nums, tables


def shift() -> tuple[dict, dict]:
    """Numbers of the tokenizer experiment (results/PREREGISTRATION_2.md)."""
    path = os.path.join(ROOT, "results", "shift", "summary.json")
    if not os.path.exists(path):
        return {}, {}
    d = json.load(open(path)); A, H, R = d["all_five"], d["held_out"], d["runs"]
    old = [r for r in R if r["train_tok"] == "outdated"]; eff = [r for r in old if r["effective"]]
    rise = lambda rs: [100 * r["rise"] for r in rs]
    win = lambda rs, w: [r for r in rs if r["window"] == w]
    base = [r for r in old if r["model"] == "base"]
    both = [r for r in R if r["train_tok"] == "both" and r["effective"] and r["window"] == "5-7"]
    off = [r for r in R if r["train_tok"] == "official" and r["effective"]]
    nums = {"shiftRuns": len(R), "shiftEff": len(eff), "shiftUp": A["P1"]["wins"], "shiftDown": A["P1"]["losses"], "shiftPone": f"{A['P1']['p']:.4f}", "shiftRelapses": A["P1"]["pipeline_relapses"],
            "shiftHeldUp": H["P1"]["wins"], "shiftHeldDown": H["P1"]["losses"], "shiftHeldP": f"{H['P1']['p']:.3f}",
            **{f"shiftMean{k}": signed(np.mean(rise(win(eff, w)))) for k, w in (("Default", "5-7"), ("Mid", "7-9"), ("Deep", "10-12"))},
            "shiftMaxRise": f"{max(rise(eff)):.1f}", "shiftMidLow": signed(min(rise(win(eff, '7-9')))), "shiftMidHigh": signed(max(rise(win(eff, '7-9')))),
            "shiftPtwoWins": A["P2"]["wins"], "shiftPtwoN": A["P2"]["wins"] + A["P2"]["losses"], "shiftPthreeWins": A["P3"]["wins"], "shiftPthreeLosses": A["P3"]["losses"], "shiftPthreeP": f"{A['P3']['p']:.3f}",
            "shiftPfourWins": A["P4"]["wins"], "shiftPfourLosses": A["P4"]["losses"], "shiftBothGap": signed(np.mean(rise(both))),
            "shiftBaseLab": f"{100 * np.mean([r['wmdp_outdated'] for r in base]):.1f}", "shiftBaseLow": f"{100 * min(r['wmdp_official'] for r in base):.1f}", "shiftBaseHigh": f"{100 * max(r['wmdp_official'] for r in base):.1f}",
            "shiftReverseEff": len(off), "shiftReverseRel": sum(-r["rise"] >= 0.05 and r["p"] < 0.05 for r in off), "shiftReverseMean": signed(-np.mean(rise(off)))}
    orig = []
    for v in ("global", "fire", "earth", "water", "base"):      # how far each original model moves between the two tokenizer settings
        rows = {(r["model"], r["prec"], r["score_tok"]): r for r in json.load(open(os.path.join(ROOT, "results", "shift", f"{v}_seed21", "results.json")))}
        orig.append(abs(rows[("base", "fp16", "official")]["wmdp-bio"] - rows[("base", "fp16", "outdated")]["wmdp-bio"]))
    nums["shiftOrigMax"] = f"{100 * max(orig):.1f}"
    return nums, {}


def place() -> tuple[dict, dict]:
    """Numbers of the placement experiment and of the final models (results/PREREGISTRATION_3.md)."""
    nums = {}
    path = os.path.join(ROOT, "results", "place", "summary.json")
    if os.path.exists(path):
        d = json.load(open(path)); A = d["results"]["all_five"]
        q = lambda fmt, k: next(v for name, v in A[fmt].items() if name.startswith(k))
        for k, tag in (("Q1", "One"), ("Q2", "Two"), ("Q3", "Three")):
            nums.update({f"placeQ{tag}Wins": q("fp16", k)["wins"], f"placeQ{tag}Losses": q("fp16", k)["losses"], f"placeQ{tag}P": f"{q('fp16', k)['p']:.3f}"})
        nums.update(placeQTwoWinsQ=q("Q4_0", "Q2")["wins"], placeQTwoWinsN=q("nf4", "Q2")["wins"])
        L = json.load(open(os.path.join(ROOT, "results", "place", "circuitkit", "water", "layer_importance.json")))["per_layer"]
        share = {a: sum(L[str(l)]["mlp_abs"] for l in range(a, a + 3)) for a in range(1, 11)}
        nums["placeWaterMargin"] = f"{100 * (share[2] - share[3]) / sum(share.values()):.1f}"
        pairs = q("fp16", "Q2")["pairs"]
        nums.update(placeShallowMean=pct(np.mean([x["best_chosen"] for x in pairs])), placeDefaultMean=pct(np.mean([x["best_default"] for x in pairs])))
    path = os.path.join(ROOT, "results", "final", "summary.json")
    if os.path.exists(path):
        d = json.load(open(path)); Q, P = d["Q4"], d["pairs"]
        won = [x for x in P if x["difference"] < 0]
        change = [100 * v["change"] for x in P for f, v in x["formats"].items() if "change" in v]
        nums.update(finalWins=Q["lower_at_3_5"], finalLosses=Q["lower_at_5_7"], finalP=f"{Q['p']:.3f}", finalSig=Q["significant_pairs"], finalMeanNew=pct(Q["mean_wmdp_3_5"]), finalMeanOld=pct(Q["mean_wmdp_5_7"]),
                    finalGainLow=f"{min(-100 * x['difference'] for x in won):.1f}", finalGainHigh=f"{max(-100 * x['difference'] for x in won):.1f}",
                    finalCostMax=f"{max(100 * x['cost_3_5'] for x in P):.1f}", finalCostMean=f"{np.mean([100 * x['cost_3_5'] for x in P]):.1f}",
                    finalBaseCostOld=" and ".join(f"{100 * x['cost_5_7']:.1f}" for x in P if x["model"] == "base"), finalBaseCostNew=" and ".join(f"{100 * x['cost_3_5']:.1f}" for x in P if x["model"] == "base"),
                    finalChangeLow=signed(min(change)), finalChangeHigh=signed(max(change)), finalRelapses=sum(bool(v.get("relapse")) for x in P for v in x["formats"].values()))
    return nums, {}


def check() -> tuple[dict, dict]:
    """The full check of the released models (results/final/check.json)."""
    path = os.path.join(ROOT, "results", "final", "check.json")
    if not os.path.exists(path):
        return {}, {}
    d = json.load(open(path)); E = list(d.values())
    cols = (("wmdp-bio", "WMDP"), ("mmlu", "MMLU"), ("benign-bio", "Benign bio"), ("attack:five_shot", "5-shot"), ("attack:role_play", "Role"), ("probe:en_para", "Reword"), ("probe:hi", "Hindi"), ("probe:sw", "Swahili"))
    models = ("global", "fire", "earth", "water", "base")
    mean = lambda v, f: np.mean([f(e) for e in E if e["model"] == v])
    body = ""
    for v in models:      # two rows per model: the original, then the unlearned model
        body += f"{v.capitalize()} & original & " + " & ".join(pct(mean(v, lambda e: e['fp16'][k]['original'])) for k, _ in cols) + " & & & \\\\\n"
        body += " & unlearned & " + " & ".join(pct(mean(v, lambda e: e['fp16'][k]['model'])) for k, _ in cols)
        body += f" & {pct(mean(v, lambda e: e['real_files']['wmdp-bio']['Q4_K_M']['file']))} & {pct(mean(v, lambda e: e['outdated_tokenizer']['outdated']))} & {pct(mean(v, lambda e: e['fine_tuning']['after_hazard_questions']))} \\\\\n"
        body += "\\addlinespace[2pt]\n" if v != models[-1] else ""
    tex = ("\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{@{}ll" + "r" * len(cols) + "rrr@{}}\n\\toprule\n" +
           " & & \\multicolumn{" + str(len(cols)) + "}{c}{At fp16} & \\multicolumn{3}{c}{\\wmdp{} of the unlearned model} \\\\\n" +
           "\\cmidrule(lr){3-" + str(len(cols) + 2) + "}\\cmidrule(l){" + str(len(cols) + 3) + "-" + str(len(cols) + 5) + "}\nModel & & " + " & ".join(n for _, n in cols) +
           " & \\qkm{} file & Old tok. & Fine-tuned \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\caption{The released models (layers 3--5, mean of two seeds): accuracy (\\%) of the original and the unlearned model at fp16 on the hazard set, general knowledge, benign biology, two prompt attacks and reworded or translated hazard questions; then the unlearned model's \\wmdp{} in the real \\qkm{} file, through the outdated tokenizer, and after a 25-step fine-tuning attack. Prompt attacks and the fine-tuning attack use the 1,204 hazard questions not used as examples or for training.}\n\\label{tab:check}\n\\end{table*}\n")
    rng_ = lambda f: (min(f(e) for e in E), max(f(e) for e in E))
    ben = [100 * e["fp16"]["benign-bio"]["difference"] for e in E]
    ft = [100 * (e["fine_tuning"]["after_hazard_questions"] - e["fine_tuning"]["before"]) for e in E]
    real = [100 * e["real_files"]["wmdp-bio"][f]["change"] for e in E for f in ("Q4_0", "Q4_K_M")]
    tok = [100 * e["outdated_tokenizer"]["difference"] for e in E]
    probes = [k for k in E[0]["fp16"] if k.startswith("probe:")]
    nums = {"chkModels": len(E), "chkWmdpLow": pct(rng_(lambda e: e["fp16"]["wmdp-bio"]["model"])[0]), "chkWmdpHigh": pct(rng_(lambda e: e["fp16"]["wmdp-bio"]["model"])[1]),
            "chkMmluMax": f"{max(-100 * e['fp16']['mmlu']['difference'] for e in E):.1f}", "chkBenignLow": f"{min(-x for x in ben):.1f}", "chkBenignHigh": f"{max(-x for x in ben):.1f}", "chkBenignMean": f"{-np.mean(ben):.1f}",
            "chkFiveHigh": pct(rng_(lambda e: e["fp16"]["attack:five_shot"]["model"])[1]), "chkRoleUp": sum(e["fp16"]["gain:role_play"]["difference"] > 0 and e["fp16"]["gain:role_play"]["p"] < 0.05 for e in E),
            "chkFiveUp": sum(e["fp16"]["gain:five_shot"]["difference"] > 0 and e["fp16"]["gain:five_shot"]["p"] < 0.05 for e in E),
            "chkProbeSig": sum(e["fp16"][k]["difference"] < 0 and e["fp16"][k]["p"] < 0.05 for e in E for k in probes), "chkProbeN": len(E) * len(probes),
            "chkRealLow": signed(min(real)), "chkRealHigh": signed(max(real)), "chkTokLow": signed(min(tok)), "chkTokHigh": signed(max(tok)),
            "chkFtUp": sum(x >= 5 for x in ft), "chkFtMax": f"{max(ft):.1f}", "chkFtBase": " and ".join(pct(e["fine_tuning"]["after_hazard_questions"]) for e in E if e["model"] == "base")}
    return nums, {"table_check.tex": tex}


def main() -> None:
    nums, tables = {}, {}
    for part in (control, relapse_map, deep, shift, place, check):
        n, t = part(); nums.update(n); tables.update(t)
    for name, tex in tables.items():
        open(os.path.join(PAPER, name), "w").write(tex)
    open(os.path.join(PAPER, "numbers.tex"), "w").write("% generated by scripts/paper_numbers.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()))


if __name__ == "__main__":
    main()
