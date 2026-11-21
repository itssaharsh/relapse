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


def table(columns: str, head: str, body: str, caption: str, label: str, sep: str = "3.5pt") -> str:
    return ("\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{" + sep + "}\n\\begin{tabular}{" + columns + "}\n\\toprule\n" + head + "\\midrule\n" + body +
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
                r"The pre-registered map, through the official tokenizer. A relapse is a rise of at least 5 points over fp16 with paired $p<0.05$.", "tab:map", "6pt")
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
    body = "".join(f"{describe(r['name'])}{', retrained' if r['arm'] == 'rerun' else ''} & {pct(r['wmdp_fp16'])} & {cell(r, 'Q4_0')} & {cell(r, 'nf4')} \\\\\n"
                   for r in sorted(rows, key=lambda r: (r["model"], r["seed"], r["arm"], float(r["lr"]))))
    tables["table_cases.tex"] = table("lrrr", r"Run & fp16 & \fq{} & \nf{} \\" + "\n", body,
                                      r"\wmdp{} accuracy (\%) of the same weights at fp16 and after each 4-bit round trip, with the change in points. $^{*}$ marks a relapse. Base is edited at layers 10--12, Fire at layers 13--15.", "tab:cases")
    # real llama.cpp files, compared by AuditKIT
    real = [(n, v) for models in d["part_b_real_files"].values() for n, v in models.items()]
    if real:
        short = lambda n: "Original" if n.endswith("_base") else re.sub(r".*?(lr[^_]+)_s(\d+)$", lambda m: f"Seed {m.group(2)}, {m.group(1).replace('lr', 'lr ')}", n.replace("_x10", "").replace("_W10-12", "").replace("_W13-15", ""))
        rc = lambda v, f: f"{pct(v[f]['file'])} ({signed(100 * v[f]['rise'])})" + ("$^{*}$" if v[f]["relapse"] else "") if f in v else "--"
        body = "".join(f"{short(n)} & {pct(v['Q4_0']['fp16'])} & {rc(v, 'Q4_0')} & {rc(v, 'Q4_K_M')} \\\\\n" for n, v in sorted(real, key=lambda x: (not x[0].endswith('_base'), x[0])))
        tables["table_real.tex"] = table("lrrr", r"Tiny Aya Base & fp16 & \fq{} file & \qkm{} file \\" + "\n", body,
                                         r"Real llama.cpp files: \wmdp{} accuracy (\%) and change against fp16, compared with AuditKIT's paired bootstrap. $^{*}$ marks a relapse.", "tab:real")
        for n, v in real:
            key = "realOrig" if n.endswith("_base") else "realA" if "1e-3" in n and "s12" in n else "realB" if "2e-4" in n and "s12" in n and "file" not in n else None
            if key:
                nums.update({key + "Fp": pct(v["Q4_0"]["fp16"]), key + "File": pct(v["Q4_0"]["file"]), key + "Rise": signed(100 * v["Q4_0"]["rise"]), key + "P": f"{v['Q4_0']['p']:.3f}"})
    # where the rise comes from: one part of the model rounded at a time (seed 12, mean of the three runs)
    C = d["part_c"]["by_format"]
    if "Q4_0" in C and "nf4" in C:
        parts = (("before", "Layers in front of the edit"), ("edited", "The three edited matrices"), ("window_other", "Other matrices in the window"), ("after", "Layers behind the edit"), ("all", "Whole model"))
        s12 = [k for k in C["Q4_0"] if k.endswith("_s12")]
        mean = lambda fmt, part: np.mean([100 * C[fmt][k][part]["rise"] for k in s12 if part in C[fmt].get(k, {})])
        body = "".join(f"{label} & {signed(100 * C['Q4_0']['base'][part]['rise'])} & {signed(mean('Q4_0', part))} & {signed(100 * C['nf4']['base'][part]['rise'])} & {signed(mean('nf4', part))} \\\\\n" for part, label in parts)
        tables["table_cause.tex"] = table("lrrrr", r" & \multicolumn{2}{c}{\fq{}} & \multicolumn{2}{c}{\nf{}} \\" + "\nPart rounded to 4 bits & Orig. & Unl. & Orig. & Unl. \\\\\n", body,
                                          r"Change in \wmdp{} (points) when only one part of the model is rounded, for the original Tiny Aya Base and for the three unlearned seed-12 runs (mean).", "tab:cause", "3pt")
        nums.update(causeAfterUnl=signed(mean("Q4_0", "after")), causeAfterOrig=signed(100 * C["Q4_0"]["base"]["after"]["rise"]), causeBeforeUnl=signed(mean("Q4_0", "before")), causeEditedUnl=signed(mean("Q4_0", "edited")))
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
                                           r"The repair and its controls, each with the seed and learning rate of the relapsing run. \wmdp{} (\%) at fp16 and in the relapsing format, with the change in points; MMLU is the difference from the plain run in that format. $^{*}$ marks a relapse.", "tab:repair")
    nums.update(repN=len(rep), repA=sum(x["R5a"] for x in rep), repB=sum(x["R5b"] for x in rep), repC=sum(x["R5c"] for x in rep))
    return nums, tables


def main() -> None:
    nums, tables = {}, {}
    for part in (control, relapse_map, deep):
        n, t = part(); nums.update(n); tables.update(t)
    for name, tex in tables.items():
        open(os.path.join(PAPER, name), "w").write(tex)
    open(os.path.join(PAPER, "numbers.tex"), "w").write("% generated by scripts/paper_numbers.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()))


if __name__ == "__main__":
    main()
