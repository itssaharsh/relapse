"""Writes the paper's numbers and tables from the result files, so nothing in the paper is typed by hand.

    python scripts/paper_numbers.py
Creates paper/numbers.tex and paper/table_*.tex. A table whose experiment has no results yet is skipped.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    shown = lambda m: round(get(m, "current_official"), 1) - round(get(m, "kaggle_copy"), 1)      # the change between the two values as displayed
    body = "".join(f"{label} & {get(m, 'kaggle_copy'):.1f} & {get(m, 'current_official'):.1f} & {signed(shown(m))} \\\\\n" for m, label in names)
    tex = table("lrrr", "Same weights & Outdated & Official & Change \\\\\n", body,
                r"\wmdp{} accuracy (\%) of the same weights on Global, scored in one job through each tokenizer setting. The weights were trained through the outdated setting. Lower is better.", "tab:control", "6pt")
    return {"ctlMaxRise": f"{max(shown(m) for m, _ in names):.1f}"}, {"table_control.tex": tex}


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
            "mapHthreeN": h["H3"]["not_comparable"],
            **{f"mapAt{k}": sum(any(r[f]["delta"] >= t and r[f]["p"] < 0.05 for f in ("Q4_0", "nf4")) for r in eff) for k, t in (("Three", 0.03), ("Four", 0.04))}}
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
    rows = [r for r in runs if r["arm"] == "plain" and r["effective"] and (r["Q4_0"]["relapse"] or r["nf4"]["relapse"] or (r["seed"] == 12 and r["window"] == "10-12"))]      # the seed-12 reruns are given in the text
    cell = lambda r, f: f"{pct(r[f]['wmdp'])} ({signed(100 * r[f]['delta'])})" + ("$^{*}$" if r[f]["relapse"] else "")
    compact = lambda r: f"{r['model'].capitalize()} {r['seed']}, lr {lr_text(r['lr'])}" + ("$^{\\dagger}$" if r["arm"] == "rerun" else "")
    body = "".join(f"{compact(r)} & {pct(r['wmdp_fp16'])} & {cell(r, 'Q4_0')} & {cell(r, 'nf4')} \\\\\n"
                   for r in sorted(rows, key=lambda r: (r["model"], r["seed"], r["arm"], float(r["lr"]))))
    def original(folder, label):
        it = json.load(open(os.path.join(ROOT, "results", *folder, "items.json")))
        acc = lambda f: float(np.mean(it[f"base|{f}|wmdp-bio"]))
        return acc, f"{label} & {pct(acc('fp16'))} & {pct(acc('Q4_0'))} ({signed(100 * (acc('Q4_0') - acc('fp16')))}) & {pct(acc('nf4'))} ({signed(100 * (acc('nf4') - acc('fp16')))}) \\\\\n"
    ob, row_b = original(("deep", "base_seed41_plain"), "\\emph{Original Base}")
    of, row_f = original(("deep", "fire_seed41_w13"), "\\emph{Original Fire}")
    ow, row_w = original(("deep", "water_seed41_plain"), "\\emph{Original Water}")
    body += "\\midrule\n" + row_b + row_f + row_w
    by_name = {r["name"]: r for r in runs}
    nums.update(origBaseQ=signed(100 * (ob("Q4_0") - ob("fp16"))), fireNet=signed(100 * (by_name["fire_W13-15_lr0.002_s44"]["nf4"]["delta"] - (of("nf4") - of("fp16")))),
                waterNet=signed(100 * (by_name["water_W10-12_lr0.001_s42"]["nf4"]["delta"] - (ow("nf4") - ow("fp16")))))
    gap = (ob("Q4_0") - ob("fp16")) - (ob("nf4") - ob("fp16"))      # R2 again, net of the original's own Q4_0 minus NF4 gap (not registered)
    seeds = [v for v in A["per_seed"].values() if v["effective"]]
    w = sum(v["mean_change_Q4_0"] - v["mean_change_nf4"] > gap for v in seeds); l = sum(v["mean_change_Q4_0"] - v["mean_change_nf4"] < gap for v in seeds)
    nums.update(rtwoNetWins=w, rtwoNetN=w + l, rtwoNetP=f"{sum(math.comb(w + l, i) for i in range(w, w + l + 1)) / 2 ** (w + l):.2f}")
    tables["table_cases.tex"] = table("@{}lrrr@{}", r"Model, seed, lr & fp16 & \fq{} & \nf{} \\" + "\n", body,
                                      r"\wmdp{} accuracy (\%) of the same weights at fp16 and after each 4-bit round trip, with the change in points. $^{*}$ marks a relapse. Base and Water are edited at layers 10--12, Fire at layers 13--15.", "tab:cases", "3pt")
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
        single = [100 * C[fmt][k]["edited"]["rise"] for fmt in C for k in C[fmt] if k != "base" and "edited" in C[fmt][k] and ((fmt == "Q4_0") == k.startswith("base"))]
        nums.update(causeEditedMax=f"{max(single):.1f}", causeCases=len(single))
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
    ctl = [e["arms"]["plain_600"] for e in D["runs"] if "plain_600" in e["arms"]]
    nums.update(ctlN=len(ctl), ctlA=sum(x["R5a"] for x in ctl), ctlB=sum(x["R5b"] for x in ctl), ctlC=sum(x["R5c"] for x in ctl), repBelow=sum(x["fp16"] < 0.30 for x in rep))
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
            "shiftHeldPtwoWins": H["P2"]["wins"], "shiftHeldPtwoN": H["P2"]["wins"] + H["P2"]["losses"], "shiftHeldPtwoP": f"{H['P2']['p']:.3f}",
            "shiftHeldPthreeWins": H["P3"]["wins"], "shiftHeldPthreeN": H["P3"]["wins"] + H["P3"]["losses"], "shiftHeldPthreeP": f"{H['P3']['p']:.2f}",
            "shiftBaseMidLow": f"{min(100 * r['rise'] for r in base if r['window'] == '7-9'):.1f}", "shiftBaseMidHigh": f"{max(100 * r['rise'] for r in base if r['window'] == '7-9'):.1f}",
            "shiftPoneModels": sum(v["p"] < 0.05 for v in A["P1"]["per_model"].values()),
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
    def windows(folder):      # share of MLP attribution of each three-layer window in layers 1-12, per model
        out = {}
        for v in ("global", "fire", "earth", "water", "base"):
            L = json.load(open(os.path.join(ROOT, "results", "place", folder, v, "layer_importance.json")))["per_layer"]
            sc = {f"{a}-{a + 2}": sum(L[str(l)]["mlp_abs"] for l in range(a, a + 3)) for a in range(1, 11)}
            out[v] = {k: x / sum(sc.values()) for k, x in sc.items()}
        return out
    if os.path.exists(os.path.join(ROOT, "results", "place", "circuitkit_more", "base", "layer_importance.json")):
        few, many = windows("circuitkit"), windows("circuitkit_more")
        top = lambda w: {v: max(w[v], key=w[v].get) for v in w}
        pooled = lambda w: sorted(((np.mean([w[v][k] for v in w]), k) for k in w["global"]), reverse=True)
        nums.update(ckChanged=sum(top(few)[v] != top(many)[v] for v in few), ckPooledFew=pooled(few)[0][1].replace("-", "--"), ckPooledMany=pooled(many)[0][1].replace("-", "--"),
                    ckLeadFew=f"{100 * (pooled(few)[0][0] - pooled(few)[1][0]):.1f}", ckLeadMany=f"{100 * (pooled(many)[0][0] - pooled(many)[1][0]):.1f}",
                    ckTopMany=sum(top(many)[v] == "3-5" for v in many))
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
    cost = {"3-5": [], "5-7": []}; forgot = {"3-5": [], "5-7": []}; mmlu = {"3-5": [], "5-7": []}
    for v in models:
        rows = {(r["model"], r["prec"]): r for r in json.load(open(os.path.join(ROOT, "results", "final", f"check_{v}", "certify.json"))) if "wmdp-bio" in r}
        for (m, p_), r in rows.items():
            if p_ == "fp16" and m != "base":
                w = re.search(r"_W(\d+-\d+)_", m).group(1); o = rows[("base", "fp16")]
                cost[w].append(100 * (o["benign-bio"] - r["benign-bio"])); forgot[w].append(100 * (o["wmdp-bio"] - r["wmdp-bio"])); mmlu[w].append(100 * (o["mmlu"] - r["mmlu"]))
    # hazard points forgotten per point of benign biology lost: mean forgetting over mean benign-biology loss, per window
    nums.update(placeRatioShallow=f"{np.mean(forgot['3-5']) / np.mean(cost['3-5']):.2f}", placeRatioDefault=f"{np.mean(forgot['5-7']) / np.mean(cost['5-7']):.2f}",
                chkForgotShallow=f"{np.mean(forgot['3-5']):.1f}", chkForgotDefault=f"{np.mean(forgot['5-7']):.1f}", chkMmluShallow=f"{np.mean(mmlu['3-5']):.1f}", chkMmluDefault=f"{np.mean(mmlu['5-7']):.1f}",
                chkPerMmluShallow=f"{np.mean(forgot['3-5']) / np.mean(mmlu['3-5']):.1f}", chkPerMmluDefault=f"{np.mean(forgot['5-7']) / np.mean(mmlu['5-7']):.1f}")
    nums.update(chkBenignDefaultMean=f"{np.mean(cost['5-7']):.1f}", chkBenignDefaultMedian=f"{np.median(cost['5-7']):.1f}", chkBenignMedian=f"{np.median(cost['3-5']):.1f}",
                chkFtUpA=sum(x >= 5 for e, x in zip(E, ft) if e["fine_tuning"] and list(d).__getitem__(E.index(e)).endswith("_s33")), chkFtUpB=sum(x >= 5 for e, x in zip(E, ft) if list(d).__getitem__(E.index(e)).endswith("_s34")))
    return nums, {"table_check.tex": tex}


def runtime() -> tuple[dict, dict]:
    """The exploratory check inside llama.cpp's own runtime (results/runtime/summary.json)."""
    path = os.path.join(ROOT, "results", "runtime", "summary.json")
    if not os.path.exists(path):
        return {}, {}
    d = json.load(open(path)); o, r = d["orig"], d["s12"]
    return {"rtQuestions": o["questions"], "rtAgreeF": f"{100 * min(x['same_answer_as_pytorch']['f16'] for x in (o, r)):.1f}", "rtAgreeQ": f"{100 * min(x['same_answer_as_pytorch']['Q4_0'] for x in (o, r)):.1f}", "rtOrig": signed(100 * o["change"]), "rtRun": signed(100 * r["change"]),
            "rtNet": signed(100 * d["net_of_original"]["change"]), "rtSameTok": o["prompts_tokenized_like_official"] + r["prompts_tokenized_like_official"], "rtPrompts": o["prompts"] + r["prompts"]}, {}


def shift2() -> tuple[dict, dict]:
    """Placement against the tokenizer (results/PREREGISTRATION_5.md)."""
    path = os.path.join(ROOT, "results", "shift2", "summary.json")
    if not os.path.exists(path):
        return {}, {}
    d = json.load(open(path)); W = d["by_window"]
    base = [r for r in d["not_effective"] if r["model"] == "base" and r["window"] == "3-5"]
    return {"stwoPairs": d["T1"]["pairs"], "stwoWins": d["T1"]["smaller_at_3_5"], "stwoShallowRel": d["T2"]["relapses"], "stwoShallowN": d["T2"]["effective"],
            "stwoMidRel": d["T3"]["7-9"][0], "stwoMidN": d["T3"]["7-9"][1], "stwoDefaultRel": d["T3"]["5-7"][0], "stwoDefaultN": d["T3"]["5-7"][1],
            "stwoShallowMean": signed(100 * W["3-5"]["mean_rise"]), "stwoDefaultMean": signed(100 * W["5-7"]["mean_rise"]), "stwoShallowMax": signed(100 * W["3-5"]["max_rise"]),
            "stwoShallowDeployed": pct(W["3-5"]["mean_wmdp_official"]), "stwoDefaultDeployed": pct(W["5-7"]["mean_wmdp_official"]), "stwoMidDeployed": pct(W["7-9"]["mean_wmdp_official"]),
            "stwoBaseLow": pct(min(r["wmdp_official"] for r in base)), "stwoBaseHigh": pct(max(r["wmdp_official"] for r in base)), "stwoBaseAll": len([r for r in d["not_effective"] if r["model"] == "base"]),
            # unfiltered: every run, effective or not (the runs outside the effective range are all Tiny Aya Base)
            **{f"stwoAll{k}": sum(r["pipeline_relapse"] for r in d["runs"] if r["window"] == w) for k, w in (("Default", "5-7"), ("Shallow", "3-5"), ("Mid", "7-9"))},
            **{f"stwoAll{k}Mean": signed(100 * np.mean([r["rise"] for r in d["runs"] if r["window"] == w])) for k, w in (("Default", "5-7"), ("Shallow", "3-5"), ("Mid", "7-9"))},
            "stwoAllN": len([r for r in d["runs"] if r["window"] == "3-5"]), "stwoRuns": len(d["runs"]),
            "stwoAllModels": " and ".join(sorted({r["model"].capitalize() for r in d["runs"] if r["window"] == "3-5" and r["pipeline_relapse"]})),
            "stwoBaseRiseLow": f"{100 * min(r['rise'] for r in base):.1f}", "stwoBaseRiseHigh": f"{100 * max(r['rise'] for r in base):.1f}",
            "stwoBaseAuditedLow": pct(min(r["wmdp_outdated"] for r in base)), "stwoBaseAuditedHigh": pct(max(r["wmdp_outdated"] for r in base)),
            "stwoDefaultAudited": pct(W["5-7"]["mean_wmdp_outdated"]), "stwoShallowAudited": pct(W["3-5"]["mean_wmdp_outdated"]),
            "stwoMidMean": signed(100 * W["7-9"]["mean_rise"]), **stwo_originals()}, {}


def stwo_originals() -> dict:
    """How far the original models move on WMDP-bio between the two tokenizer settings (official minus outdated), in the jobs of pre-registration 5."""
    move = {}
    for v in ("global", "fire", "earth", "water", "base"):
        for s in (51, 52):
            rows = {(r["model"], r["score_tok"]): r for r in json.load(open(os.path.join(ROOT, "results", "shift2", f"{v}_seed{s}", "results.json"))) if r["prec"] == "fp16"}
            move.setdefault(v, []).append(100 * (rows[("base", "official")]["wmdp-bio"] - rows[("base", "outdated")]["wmdp-bio"]))
    four = [x for v in move if v != "base" for x in move[v]]
    return {"stwoOrigLow": signed(min(four)), "stwoOrigHigh": signed(max(four)), "stwoOrigBase": signed(np.mean(move["base"])), "stwoOrigFourMax": f"{max(abs(x) for x in four):.1f}"}


def glance() -> tuple[dict, dict]:
    """The result in brief: SafeTune's default layers against layers 3-5, from pre-registrations 3 (final seeds) and 5."""
    p2, pf = os.path.join(ROOT, "results", "shift2", "summary.json"), os.path.join(ROOT, "results", "final", "summary.json")
    if not (os.path.exists(p2) and os.path.exists(pf)):
        return {}, {}
    s2, fin = json.load(open(p2)), json.load(open(pf)); W = s2["by_window"]
    rng = np.random.default_rng(0)
    from analyze_map_shim import paired
    relapses = {"3-5": 0, "5-7": 0}; cost = {"3-5": [], "5-7": []}; benign = {"3-5": [], "5-7": []}; forgot = {"3-5": [], "5-7": []}
    for v in ("global", "fire", "earth", "water", "base"):
        d = os.path.join(ROOT, "results", "final", f"{v}_seed33")
        rows = {(r["model"], r["prec"]): r for r in json.load(open(os.path.join(d, "results.json")))}; items = json.load(open(os.path.join(d, "items.json")))
        for (m, p), r in rows.items():
            if p != "fp16" or m == "base":
                continue
            w = re.search(r"_W(\d+-\d+)_", m).group(1)
            cost[w].append(100 * (rows[("base", "fp16")]["mmlu"] - r["mmlu"]))
            for f in ("Q4_0", "nf4"):
                delta, pv = paired(items[f"{m}|fp16|wmdp-bio"], items[f"{m}|{f}|wmdp-bio"], rng)
                relapses[w] += int(delta >= 0.05 and pv < 0.05)
        c = {(r["model"], r["prec"]): r for r in json.load(open(os.path.join(ROOT, "results", "final", f"check_{v}", "certify.json"))) if "wmdp-bio" in r}
        for (m, p), r in c.items():
            if p == "fp16" and m != "base":
                benign[re.search(r"_W(\d+-\d+)_", m).group(1)].append(100 * (c[("base", "fp16")]["benign-bio"] - r["benign-bio"]))
                forgot[re.search(r"_W(\d+-\d+)_", m).group(1)].append(100 * (c[("base", "fp16")]["wmdp-bio"] - r["wmdp-bio"]))
    row = lambda label, a_, b_: f"{label} & {a_} & {b_} \\\\\n"
    body = (row("\\wmdp{} as audited (\\%)", pct(W["5-7"]["mean_wmdp_outdated"]), pct(W["3-5"]["mean_wmdp_outdated"]))
            + row("\\wmdp{} as deployed (\\%)", pct(W["5-7"]["mean_wmdp_official"]), "\\textbf{" + pct(W["3-5"]["mean_wmdp_official"]) + "}")
            + row("Relapses, effective runs", f"{s2['T3']['5-7'][0]} of {s2['T3']['5-7'][1]}", "\\textbf{" + f"{s2['T2']['relapses']} of {s2['T2']['effective']}" + "}")
            + row("Rise $\\geq$5 points, all runs", f"{sum(r['pipeline_relapse'] for r in s2['runs'] if r['window'] == '5-7')} of {sum(r['window'] == '5-7' for r in s2['runs'])}",
                  f"{sum(r['pipeline_relapse'] for r in s2['runs'] if r['window'] == '3-5')} of {sum(r['window'] == '3-5' for r in s2['runs'])}")
            + "\\midrule\n"
            + row("MMLU points lost", f"{np.mean(cost['5-7']):.1f}", f"{np.mean(cost['3-5']):.1f}")
            + row("Benign-biology points lost", f"{np.mean(benign['5-7']):.1f}", f"{np.mean(benign['3-5']):.1f}")
            + row("Hazard points per benign point", f"{np.mean(forgot['5-7']) / np.mean(benign['5-7']):.2f}", f"{np.mean(forgot['3-5']) / np.mean(benign['3-5']):.2f}"))
    tex = table("@{}lrr@{}", "Layers edited & 5--7 (default) & 3--5 \\\\\n", body,
                "The result in brief. Top: runs unlearned and audited through the outdated tokenizer setting, then deployed through the official one. Effective runs: the four post-trained variants; all runs add Tiny Aya Base, at chance when audited. Bottom: the final runs, official tokenizer, five models and two seeds; the last row is hazard points forgotten per point of benign biology lost. Lower \\wmdp{} is better.", "tab:glance", "4pt").replace("\\begin{table}[t]", "\\begin{table}[!b]")      # page one, at the foot of the first column
    return {"glanceRelDefault": relapses["5-7"], "glanceRelNew": relapses["3-5"]}, {"table_glance.tex": tex}


def mcnemar() -> tuple[dict, dict]:
    """Exact McNemar p-values of the relapse cases, Holm-corrected over the cells screened in the same experiment (scripts/mcnemar_holm.py)."""
    from mcnemar_holm import holm, screens
    S = screens()
    def sci(p):
        mant, exp = f"{p:.1e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$" if p < 0.001 else f"{p:.3f}"
    pick = lambda screen, run, f: next(c for c in S[screen] if c["run"] == run and c["format"] == f)
    fire, water = pick("fire", "fire_W13-15_lr0.002_s44", "nf4"), pick("others", "water_W10-12_lr0.001_s42", "nf4")
    base = [c for k in ("map", "base") for c in S[k] if c["relapse"]]
    everything = [c for cells in S.values() for c in cells]
    pooled = dict(zip([id(c) for c in everything], holm([c["p"] for c in everything])))
    rel = [c for c in everything if c["relapse"]]
    return {"mcCells": len(everything), "mcMapCells": len(S["map"]), "mcBaseCells": len(S["base"]), "mcFireCells": len(S["fire"]), "mcWaterCells": len(S["others"]),
            "mcFireP": sci(fire["p"]), "mcFireHolm": sci(fire["p_holm"]), "mcWaterP": sci(water["p"]), "mcWaterHolm": sci(water["p_holm"]),
            "mcBaseN": len(base), "mcBaseHolmMax": sci(max(c["p_holm"] for c in base)), "mcRelapses": len(rel), "mcHolmMax": sci(max(c["p_holm"] for c in rel)), "mcPooledMax": sci(max(pooled[id(c)] for c in rel)),
            "mcOtherSig": sum(c["p_holm"] < 0.05 and c["rise"] > 0 and not c["relapse"] for c in everything)}, {}


def revisions() -> tuple[dict, dict]:
    """Which model and tokenizer revision each experiment used: the job ledger (results/survey/jobs.json) against the official repos' history."""
    base = os.path.join(ROOT, "results", "survey")
    if not os.path.exists(os.path.join(base, "jobs.json")):
        return {}, {}
    import datetime
    J = [j for j in json.load(open(os.path.join(base, "jobs.json"))) if j["run_suffix"] != "_smoke"]
    H = json.load(open(os.path.join(base, "official_history.json")))
    smoke = [j for j in json.load(open(os.path.join(base, "jobs.json"))) if j["run_suffix"] == "_smoke"]
    def main_at(when: str) -> tuple:      # the head of each official repo's main branch at a time, by its public history
        return tuple(next(c["sha"][:10] for c in H[f"CohereLabs/tiny-aya-{v}"] if c["date"] <= when) for v in ("global", "fire", "earth", "water", "base"))
    core = lambda j: set(j["settings"]) - {"LLAMA_COMMIT", "REPO_COMMIT", "OUT_ROOT"}
    between = lambda a, b: (lambda j: a <= j["started"] <= b)
    rows = [("Relapse map (PR1)", lambda j: core(j) == {"SEED", "VARIANT"} and j["started"] < "2026-11-21T19"),
            ("Tokenizer control", lambda j: not core(j)),
            ("Tokenizer experiment (PR2)", lambda j: j["out"] == "changed_shift.pt"),
            ("CircuitKIT, 64 prompts", lambda j: core(j) == {"VARIANT"}),
            ("Placement (PR3)", lambda j: j["experiment"] == "place"),
            ("Relapse follow-up, cause, repairs, their real files (PR4); Stage~1 rescoring", between("2026-11-21T18:34", "2026-11-22T02:05")),
            ("Final models, full check, real files, publishing (PR3, am.~1)", between("2026-11-22T02:31", "2026-11-22T07:39")),
            ("Runtime check", lambda j: j["flavor"].startswith("cpu")),
            ("Placement and tokenizer (PR5)", lambda j: j["out"] == "changed_shift2.pt"),
            ("NPO pilot", lambda j: j["out"] == "npo_deltas.pt"),
            ("CircuitKIT, 160 prompts", lambda j: j["started"] >= "2026-11-24")]
    used, sets, body = set(), [], ""
    stamp = lambda t: datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").strftime("%-d Nov %H:%M")
    for label, test in rows:
        g = [j for i, j in enumerate(J) if i not in used and test(j)]
        used |= {i for i, j in enumerate(J) if i not in used and test(j)}
        revs = sorted({main_at(j["started"]) for j in g})
        for r in revs:
            if r not in sets:
                sets.append(r)
        a_, b_ = stamp(g[0]["started"]), stamp(g[-1]["started"])
        span = a_ if a_ == b_ else f"{a_}--{b_[-5:]}" if a_[:-5] == b_[:-5] else f"{a_} -- {b_}"
        body += f"{label} & {len(g)} & {span} & " + ", ".join(f"R{sets.index(r) + 1}" for r in revs) + " \\\\\n"
    assert len(used) == len(J), "a job is in no row of the revision table"
    date = lambda sha: next(c["date"][:10] for cs in H.values() for c in cs if c["sha"].startswith(sha))
    names = ("Global", "Fire", "Earth", "Water", "Base")
    desc = "; ".join(f"R{i + 1}: " + ", ".join(f"{n} \\texttt{{{sha}}}" for n, sha in zip(names, r)) + f" (committed {' and '.join(sorted({date(x) for x in r}))})" for i, r in enumerate(sets))
    same = all(len({json.dumps({k: v for k, v in c["files"].items() if k != "generation_config.json"}, sort_keys=True) for c in H[f"CohereLabs/tiny-aya-{v}"] if c["sha"][:10] in {r[i] for r in sets}}) == 1
               for i, v in enumerate(("global", "fire", "earth", "water", "base")))
    body += ("\\midrule\nStage~1 (withdrawn results; probe translations) & -- & 9--11 Nov, on Kaggle & Kaggle's copy, version 1 \\\\\n")
    tex = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.44\\textwidth} r l >{\\raggedright\\arraybackslash}p{0.2\\textwidth}@{}}\n\\toprule\nExperiment & Jobs & Jobs started (UTC, 2026) & Revision \\\\\n\\midrule\n" + body +
           "\\bottomrule\n\\end{tabular}\n\\caption{Model and tokenizer revision of every experiment. No Stage~2 run recorded the revision it downloaded: each job took the main branch of \\texttt{CohereLabs/tiny-aya-<variant>} when it started, so the revision is read from the job's start time (Hugging Face Jobs' record, \\texttt{results/survey/jobs.json}) and the repositories' public history (\\texttt{results/survey/official\\_history.json}). " + desc +
           ("." + (" Among tokenizer files, configuration and weights, R1 and R2 differ only in \\texttt{generation\\_config.json}: the tokenizer files and weights have the same content hashes." if len(sets) > 1 and same else "")) +
           f" Job counts exclude {len(smoke)} three-batch pipeline tests. The outdated setting inside Stage~2 jobs is the same revision's tokenizer files with \\texttt{{tokenizer\\_class}} set to \\texttt{{CohereTokenizerFast}}; Stage~1 used the copy Kaggle held (uploaded 16 February 2026), whose revision Kaggle identifies only as version 1.}}\n\\label{{tab:revisions}}\n\\end{{table*}}\n")
    everything = json.load(open(os.path.join(base, "jobs.json")))
    pr5 = [j for j in J if j["out"] == "changed_shift2.pt"]
    return {"revJobs": len(J), "revSmoke": len(smoke), "revSets": len(sets), "revAll": len(everything), "revCompleted": sum(j["outcome"] == "COMPLETED" for j in everything),
            "revPrFiveFirst": pr5[0]["started"][11:19], "revPrFiveLast": pr5[-1]["started"][11:19], "revPrFiveCommit": pr5[0]["repo_commit"][:7], "revPrFiveFlavor": pr5[0]["flavor"].replace("JobHardware.", "").lower()}, {"table_revisions.tex": tex}


def survey() -> tuple[dict, dict]:
    """The survey of public copies (results/survey/, written by scripts/survey_copies.py)."""
    base = os.path.join(ROOT, "results", "survey")
    if not os.path.exists(os.path.join(base, "copies.csv")):
        return {}, {}
    import csv
    import datetime
    R = list(csv.DictReader(open(os.path.join(base, "copies.csv"))))
    S = json.load(open(os.path.join(base, "summary.json"))); P = json.load(open(os.path.join(base, "probe_sample.json")))
    official_repos = set(S["official_repos"])
    hf = [r for r in R if not r["repo"].startswith("kaggle:")]; kg = [r for r in R if r["repo"].startswith("kaggle:")]
    derived = [r for r in hf if r["repo"] not in official_repos]
    out = [r for r in derived if r["setting"] == "outdated"]; others = [r for r in out if not r["repo"].startswith("itssaharsh/")]
    num = lambda x: int(x) if str(x).isdigit() else 0
    th = lambda n: f"{n:,}".replace(",", "{,}")
    gguf = [r for r in derived if "tokenizer.ggml.pre" in r["evidence"]]
    day = datetime.datetime.strptime(S["date"], "%Y-%m-%d")
    return {"surveyDate": f"{day.day} {day.strftime('%B %Y')}", "surveyFound": th(len(hf)), "surveyRead": len([r for r in derived if r["setting"] in ("official", "outdated")]),
            "surveyOutdated": len(out), "surveyOutdatedOthers": len(others), "surveyOurs": len(out) - len(others), "surveyDownloads": th(sum(num(r["downloads"]) for r in others)),
            "surveyDownloadsMonth": th(sum(num(r["downloads_30d"]) for r in others)), "surveyAfterFix": sum(r["last_modified"] > "2026-09-30" for r in others),
            "surveyUnknown": len([r for r in derived if r["setting"] == "unknown"]), "surveyNoFiles": sum("no tokenizer_config.json" in r["evidence"] for r in derived),
            "surveyOfficialRepos": sum(v == "official" for v in S["official_repos"].values()), "surveyKaggle": len(kg), "surveyKaggleOutdated": sum(r["setting"] == "outdated" for r in kg),
            "surveyGguf": len(gguf), "surveyGgufOther": sum("tokenizer.ggml.pre=tiny_aya" not in r["evidence"] and "tokenizer.ggml.pre=cohere2moe" not in r["evidence"] for r in gguf),
            "surveySampled": P["sampled"], "surveySampledDiffer": P["split_differently"], "surveyTransformers": P["transformers"],
            "surveyShare": f"{100 * len(out) / len([r for r in derived if r['setting'] in ('official', 'outdated')]):.0f}"}, {}


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def hypotheses() -> list[tuple[str, str, str, str]]:
    """Every registered hypothesis as (pre-registration, id, statement, outcome), the outcome read from the flags the registered analyses wrote.

    Outcomes: held, part (held in part), failed, untestable. Pre-registrations 1 and 2 name the held-out models as the confirmatory test."""
    res = lambda *p: json.load(open(os.path.join(ROOT, "results", *p)))
    m, d, sh, pl, fin, s2 = res("map", "summary.json"), res("deep", "summary.json"), res("shift", "summary.json"), res("place", "summary.json"), res("final", "summary.json"), res("shift2", "summary.json")
    ok = lambda flag: "held" if flag else "failed"
    H, P = m["held_out"], sh["held_out"]
    q = lambda k: next(v for name, v in pl["results"]["all_five"]["fp16"].items() if name.startswith(k))
    real = [v["Q4_0"]["relapse"] for t, models in d["part_b_real_files"].items() if t != "deep_ladder_s12" for n, v in models.items() if "base" in n.lower() and not n.endswith("_base") and "file_nb" not in n]
    D = d["part_d"]; r5 = [D[k] for k in ("R5a_holds_in", "R5b_holds_in", "R5c_holds_in")]
    r7 = [e["dose_rule"]["R7"] for e in D["runs"] if e["plain"].startswith("water")]      # Water is the rule's only test
    return [("1", "H1", "A relapse appears in one 4-bit format, not both", ok(H["H1"]["holds"])),
            ("1", "H2", "A single-format audit misses relapses in both directions", ok(H["H2"]["holds"])),
            ("1", "H3", "Layers 7--9 cost less than 5--7 at equal forgetting", "untestable" if H["H3"]["7_9_cheaper"] + H["H3"]["5_7_cheaper"] + H["H3"]["ties"] == 0 else ok(H["H3"]["holds"])),
            ("1", "H4", "Layers 10--12 relapse more than layers 7--9", ok(H["H4"]["holds"])),
            ("2", "P1", "Unlearned through one tokenizer, a run relapses through the other", ok(P["P1"]["holds"])),
            ("2", "P2", "The rise is smaller at layers 7--9 than at 5--7", ok(P["P2"]["holds"])),
            ("2", "P3", "Training on both tokenizers narrows the gap", ok(P["P3"]["holds"])),
            ("2", "P4", "Training on both protects the deployed model", ok(P["P4"]["holds"])),
            ("3", "Q1", "Layers 7--9 forget more than 5--7 within 3 MMLU points", ok(q("Q1")["holds"])),
            ("3", "Q2", "Layers 3--5 forget more than 5--7 within 3 MMLU points", ok(q("Q2")["holds"])),
            ("3", "Q3", "Each model's own CircuitKIT window beats 5--7", ok(q("Q3")["holds"])),
            ("3", "Q4", "Layers 3--5 beat 5--7 again on fresh seeds", ok(fin["Q4"]["holds"])),
            ("4", "R1", "The Base relapse in \\fq{} recurs on fresh seeds", ok(d["part_a"]["R1"]["holds"])),
            ("4", "R2", "On Base, \\fq{} raises the score more than \\nf{}", ok(d["part_a"]["R2"]["holds"])),
            ("4", "R3", "The relapse shows in the real \\fq{} file", "held" if all(real) else "part" if any(real) else "failed"),
            ("4", "R4", "It comes from the layers in front of the edit", ok(d["part_c"]["R4"]["holds_in_every_run"])),
            ("4", "R5", "Training on the file repairs it (three criteria)", "held" if min(r5) == D["repaired_runs"] else "part" if max(r5) > 0 else "failed"),
            ("4", "R6", "A relapse in \\nf{} and not \\fq{} exists", ok(d["part_e"]["R6"]["holds"])),
            ("4", "R7", "A rule chooses how long to train the repair", ok(all(r7))),
            ("5", "T1", "The tokenizer rise is smaller at layers 3--5 than at 5--7", ok(s2["T1"]["holds"])),
            ("5", "T2", "Under 1 in 5 runs at layers 3--5 relapse through the tokenizer", ok(s2["T2"]["holds"])),
            ("5", "T3", "Layers 7--9 again hold and layers 5--7 again relapse", ok(s2["T3"]["holds"]))]


def totals() -> tuple[dict, dict]:
    """The number of unlearning runs with a results.json under results/, and the tally of registered hypotheses."""
    import glob
    base = os.path.join(ROOT, "results")
    names = {(os.path.relpath(f, base).split(os.sep)[0], r["model"])      # (experiment folder, run name); the original models are not runs
             for f in glob.glob(os.path.join(base, "**", "results.json"), recursive=True) for r in json.load(open(f)) if r.get("model") != "base"}
    hyp = hypotheses()
    count = lambda o: sum(x[3] == o for x in hyp)
    label = {"held": "Held", "part": "In part", "failed": "\\textbf{Failed}", "untestable": "Untestable"}
    body = "".join(("\\addlinespace[2pt]\n" if i and hyp[i - 1][0] != h[0] else "") + f"{h[1]} & {h[2]} & {label[h[3]]} \\\\\n" for i, h in enumerate(hyp))
    tex = table("@{}l>{\\raggedright\\arraybackslash}p{0.68\\columnwidth}l@{}", " & Registered hypothesis & Outcome \\\\\n", body,
                "Every registered hypothesis, its outcome read from the flags the registered analyses wrote. H: pre-registration 1; P: 2; Q: 3; R: 4; T: 5. For H and P the held-out models are the confirmatory test; P2 and P3 held on all five models and failed there.", "tab:hyp", "3pt", "\\footnotesize")
    return {"totalRuns": len(names), "totalNpo": sum(e == "npo_pilot" for e, _ in names), "hypN": len(hyp), "hypHeld": count("held"), "hypPart": count("part"), "hypFailed": count("failed"), "hypUntestable": count("untestable"),
            "hypFailedWord": WORDS[count("failed")], "hypHeldList": ", ".join(h[1] for h in hyp if h[3] == "held"), "hypFailedList": ", ".join(h[1] for h in hyp if h[3] == "failed"),
            "hypPartList": " and ".join(h[1] for h in hyp if h[3] == "part"), "hypUntestableList": ", ".join(h[1] for h in hyp if h[3] == "untestable")}, {"table_hyp.tex": tex}


def main() -> None:
    nums, tables = {}, {}
    for part in (control, relapse_map, deep, shift, place, check, runtime, shift2, glance, mcnemar, revisions, survey, totals):
        n, t = part(); nums.update(n); tables.update(t)
    for name, tex in tables.items():
        if name in ("table_hyp.tex", "table_real.tex", "table_cause.tex", "table_repair.tex", "table_revisions.tex"):      # appendix floats: placed where they fall, whatever their height
            tex = tex.replace("\\begin{table}[t]", "\\begin{table}[!t]").replace("\\begin{table*}[t]", "\\begin{table*}[!t]")
        open(os.path.join(PAPER, name), "w").write(tex)
    open(os.path.join(PAPER, "numbers.tex"), "w").write("% generated by scripts/paper_numbers.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()))


if __name__ == "__main__":
    main()
