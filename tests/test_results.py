"""Every number on the page and in the write-up comes from the raw runs."""
import json
import os
import re

from build_results import build, load_rows, load_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(os.path.join(ROOT, "space", "data", "results.json")))
METHODS = json.load(open(os.path.join(ROOT, "results", "methods.json")))


def test_page_data_is_rebuilt_exactly_from_raw_runs():
    rebuilt = build(load_rows(), METHODS, RES)
    assert rebuilt["grid"] == RES["grid"]
    assert len([c for g in RES["grid"].values() for c in g.values()]) == 35


def allowed():
    pct, pts, pv = {25.0, 95.0, 100.0}, {2.0}, set()   # chance, the "95% interval" level, certainty; "within 2 points"
    grid = RES["grid"]
    for m, g in grid.items():
        for f, c in g.items():
            vals = [c["forget"], c["retain"]] + [v for k, v in c.get("probes", {}).items() if not k.endswith("pcorrect")]
            for v in vals:
                pct |= {round(100 * v, 1), float(round(100 * v))}
            o = grid["original"].get(f)
            if o and m != "original":
                pts.add(round(abs(100 * (o["retain"] - c["retain"])), 1))
    for s in load_stats(METHODS):
        pts |= {round(abs(100 * s["delta"]), 1)} | {round(abs(100 * x), 1) for x in s["ci95"]}
        pct |= {round(100 * s["acc_a"], 1), round(100 * s["acc_b"], 1)}
        pv |= {round(s["p_value"], 2), round(s["p_value"], 3)}
    return pct, pts, pv


def test_every_number_in_the_write_up_is_measured():
    pct, pts, pv = allowed()
    bad = []
    for doc in ("README.md", os.path.join("docs", "ARTICLE.md")):
        text = open(os.path.join(ROOT, doc)).read()
        text = re.sub(r"<!-- results:start -->.*?<!-- results:end -->", "", text, flags=re.S)  # generated table
        for x in re.findall(r"(?<![\d.$])(\d+(?:\.\d+)?)%", text):
            if float(x) not in pct:
                bad.append(f"{doc}: {x}%")
        for x in re.findall(r"(?<![\d.$])(\d+(?:\.\d+)?) points?", text):
            if float(x) not in pts:
                bad.append(f"{doc}: {x} points")
        for x in re.findall(r"p = (0\.\d+)", text):
            if float(x) not in pv:
                bad.append(f"{doc}: p = {x}")
        for a, b in re.findall(r"(-?\d+\.\d) to \+?(-?\d+\.\d)", text):
            if abs(float(a)) not in pts or abs(float(b)) not in pts:
                bad.append(f"{doc}: interval {a} to {b}")
    assert not bad, bad
