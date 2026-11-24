"""Every number on the page and in the write-up comes from the runs scored with the official tokenizer, or the tokenizer control."""
import json
import os
import re

from analyze_map import TARGET
from build_corrected import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(os.path.join(ROOT, "space", "data", "results.json")))
SRC = os.path.join(ROOT, "results", "correct_tokenizer")
DOCS = ("README.md", os.path.join("docs", "ARTICLE.md"))
read = lambda d: open(os.path.join(ROOT, d)).read()


def test_page_data_is_rebuilt_exactly():
    assert build(RES)["grid"] == RES["grid"]


def allowed():
    pct, pts, pv = {25.0, 95.0}, set(), set()
    rows = json.load(open(os.path.join(SRC, "grid_stage1_rescored.json")))
    by = {(r["model"], r["prec"]): r for r in rows}
    for r in rows:
        pct |= {round(100 * r["wmdp-bio"], 1), round(100 * r["mmlu"], 1)}
    for s in json.load(open(os.path.join(SRC, "audit_stage1_rescored.json"))):
        pts |= {round(abs(100 * s["delta"]), 1)} | {round(abs(100 * x), 1) for x in s["ci95"]}
        pv |= {round(s["p_value"], 2), round(s["p_value"], 3)}
    c = json.load(open(os.path.join(SRC, "tokenizer_control.json")))
    for r in c:
        pct.add(round(100 * r["wmdp-bio"], 1))
    for m in {r["model"] for r in c}:
        a = next(r for r in c if r["model"] == m and r["tokenizer"] == "current_official")
        b = next(r for r in c if r["model"] == m and r["tokenizer"] == "kaggle_copy")
        pts.add(round(abs(100 * (a["wmdp-bio"] - b["wmdp-bio"])), 1))
    pts.add(7.4)      # the withdrawn Stage 1 claim, quoted once in the correction
    for m in ("rmu_deeper", "rmu_deeper_file"):
        for p in ("fp16", "Q4_0", "Q4_K_M"):
            pts.add(round(abs(100 * (by[(m, p)]["wmdp-bio"] - by[(m, "fp16")]["wmdp-bio"])), 1))
    m = json.load(open(os.path.join(ROOT, "results", "map", "summary.json")))    # the pre-registered map
    pct.add(100 * TARGET)
    for r in m["runs"]:
        pts |= {round(100 * r[f]["delta"], 1) for f in ("Q4_0", "nf4") if r.get(f, {}).get("relapse")}
    pv.add(round(m["all_five"]["H4"]["p"], 3))
    return pct, pts, pv


def test_every_number_in_the_write_up_is_measured():
    pct, pts, pv = allowed()
    pts |= {round(p, 1) for p in pct}
    bad = []
    for doc in DOCS:
        text = re.sub(r"<!-- (scores|recipes|stage2|brief):start -->.*?<!-- (scores|recipes|stage2|brief):end -->", "", read(doc), flags=re.S)
        bad += [f"{doc}: {x}%" for x in re.findall(r"(?<![\d.$])(\d+(?:\.\d+)?)%", text) if float(x) not in pct]
        bad += [f"{doc}: {x} points" for x in re.findall(r"(?<![\d.$])(\d+\.\d+) points?", text) if float(x) not in pts]
        bad += [f"{doc}: p = {x}" for x in re.findall(r"p = (0\.\d+)", text) if float(x) not in pv]
    assert not bad, bad


def test_stage2_block_is_rebuilt_from_the_results():
    from build_readme import REPO, block, brief
    for doc in DOCS:
        assert "<!-- brief:start -->\n" + brief() + "<!-- brief:end -->" in read(doc)
    text = block()
    assert "<!-- stage2:start -->\n" + text + "<!-- stage2:end -->" in read("README.md")
    assert "<!-- stage2:start -->\n" + text.replace("](results/", "](" + REPO + "results/") + "<!-- stage2:end -->" in read(os.path.join("docs", "ARTICLE.md"))


def test_withdrawn_claims_stay_withdrawn():
    for doc in DOCS:
        t = read(doc)
        assert "forgot as much" not in t and "What we withdraw" in t
    assert "deeper" not in {m["id"] for m in RES["methods"]}
