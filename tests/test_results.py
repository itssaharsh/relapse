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
    assert len([c for g in RES["grid"].values() for c in g.values()]) >= 35


def allowed():
    # chance, the 40% cut-off, the "95% interval" level; the page's rules: a relapse is 5 points back, red at 3 points lost
    pct, pts, pv = {25.0, 40.0, 95.0, 100.0}, {2.0, 3.0, 5.0}, set()
    runs = {}
    for p in RES.get("frontier", []):
        pct |= {round(100 * p["forget"], 1), round(100 * p["retain"], 1)}
        if p.get("price") is not None:
            pts.add(round(abs(100 * p["price"]), 1))
        runs.setdefault(p["id"], {})[p["prec"]] = p["forget"]
    for d in runs.values():                                     # how far each run's hazard score moved from its fp16 value
        pts |= {round(abs(100 * (v - d["fp16"])), 1) for v in d.values()} if "fp16" in d else set()
    for w in RES.get("windows", []):
        for c in w["cost"].values():
            pts |= {round(abs(100 * c["lo"]), 1), round(abs(100 * c["hi"]), 1)}
        for c in w["shift"].values():
            pts |= {round(abs(100 * c[k]), 1) for k in ("mean", "lo", "hi")}
    grid = RES["grid"]
    for m, g in grid.items():
        for f, c in g.items():
            vals = [c["forget"], c["retain"]] + [v for k, v in c.get("probes", {}).items() if not k.endswith("pcorrect")]
            for v in vals:
                pct |= {round(100 * v, 1), float(round(100 * v))}
            o = c.get("orig") or grid["original"].get(f)
            if o and m != "original":
                pts.add(round(abs(100 * (o["retain"] - c["retain"])), 1))
                pct |= {round(100 * o["forget"], 1), round(100 * o["retain"], 1)}
            if "fp16" in g:
                pts.add(round(abs(100 * (c["forget"] - g["fp16"]["forget"])), 1))
    import glob as _g
    for f in _g.glob(os.path.join(ROOT, "results", "raw", "*.json")):   # every raw run, including the tuning sweeps
        rows = json.load(open(f))
        if isinstance(rows, list):
            for r in rows:
                for k in ("wmdp-bio", "mmlu", "acc"):
                    if isinstance(r, dict) and isinstance(r.get(k), float):
                        pct |= {round(100 * r[k], 1), float(round(100 * r[k]))}
    loc = json.load(open(os.path.join(ROOT, "results", "raw", "locate_relapse.json")))   # one part of the model rounded at a time
    ref = {r["model"]: r["wmdp-bio"] for r in loc if r["part"] == "none"}
    pts |= {round(abs(100 * (r["wmdp-bio"] - ref[r["model"]])), 1) for r in loc}
    for s in RES.get("recipes", []):
        pts |= {round(abs(100 * s["delta"]), 1)} | {round(abs(100 * x), 1) for x in s["ci95"]}
        pct |= {round(100 * s["acc_a"], 1), round(100 * s["acc_b"], 1)}
        pv |= {round(s["p_value"], 2), round(s["p_value"], 3)}
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
        text = re.sub(r"<!-- (results|recipes|windows):start -->.*?<!-- (results|recipes|windows):end -->", "", text, flags=re.S)  # generated tables
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


def sweep_ranges(cut=0.40):
    out = {}
    for prec in ("fp16", "Q4_0", "NF4"):
        for fam in ("default", "circuit"):
            ps = [p["price"] for p in RES["frontier"] if p["family"] == fam and p["prec"] == prec and p["forget"] <= cut]
            out[(prec, fam)] = f"{100 * min(ps):.1f} to {100 * max(ps):.1f}"
    return out


def test_the_sweep_ranges_in_the_write_up_are_the_current_ones():
    """The 'runs that got WMDP-bio to 40% or below' ranges are recomputed from the data, so new runs cannot leave them stale."""
    for doc in ("README.md", os.path.join("docs", "ARTICLE.md")):
        text = open(os.path.join(ROOT, doc)).read()
        for key, rng in sweep_ranges().items():
            assert rng in text, (doc, key, rng)


DOCS = ("README.md", os.path.join("docs", "ARTICLE.md"))
read = lambda doc: open(os.path.join(ROOT, doc)).read()


def test_run_counts_in_the_write_up():
    fams = {w["family"]: w["runs"] for w in RES["windows"]}
    assert sum(fams.values()) == len({p["id"] for p in RES["frontier"] if p["family"] != "original"})
    for doc in DOCS:
        text = read(doc)
        assert f"{sum(fams.values())} unlearning runs" in text or f"{sum(fams.values())} runs in all" in text, (doc, fams)
        assert f"{fams['default']} RMU settings at the default layers" in text or f"{fams['default']} settings at the default layers" in text, doc
        assert f"{fams['default'] + fams['circuit']} runs at the default and CircuitKIT layers" in text, doc


def test_stability_claim():
    """Outside the window that relapses, no run's hazard score rose by more than the stated amount in any file."""
    held = [w for w in RES["windows"] if w["family"] != "deeper"]
    runs = {}
    for p in RES["frontier"]:
        if p["family"] in {w["family"] for w in held}:
            runs.setdefault(p["id"], {})[p["prec"]] = p["forget"]
    d = [v - r["fp16"] for r in runs.values() for f, v in r.items() if f != "fp16"]
    sent = (f"Across those {len(runs)} runs and every format, no hazard score rose by more than {100 * max(d):.1f} points from its fp16 value "
            f"({len(d)} run-and-file pairs; the largest fall was {-100 * min(d):.1f})")
    deep = next(w for w in RES["windows"] if w["family"] == "deeper")["shift"]["Q4_0"]
    assert deep["lo"] > 0 and deep["n"] == 4            # "rose from fp16 to Q4_0 in all four runs"
    means = [abs(100 * w["shift"]["Q4_0"]["mean"]) for w in held]
    assert all(w["shift"]["Q4_0"]["mean"] < 0 for w in held)
    for doc in DOCS:
        assert sent in read(doc), (doc, sent)
        assert f"averaged {min(means):.1f} to {max(means):.1f} points lower" in read(doc), doc


def test_cutoff_sensitivity_claim():
    def apart(cut, prec):
        get = lambda fam: [p["price"] for p in RES["frontier"] if p["family"] == fam and p["prec"] == prec and p["forget"] <= cut]
        return min(get("default")) > max(get("circuit"))
    precs, cuts = ("fp16", "Q4_0", "NF4"), [c / 100 for c in range(35, 47)]
    ok = [c for c in cuts if all(apart(c, p) for p in precs)]
    bad = next(c for c in cuts if c not in ok)
    assert ok == [c for c in cuts if c < bad]
    sent = f"The ranges stay apart for cut-offs from 35% to {round(100 * ok[-1])}%; from {round(100 * bad)}% they overlap in {' and '.join(p for p in precs if not apart(bad, p))}."
    for doc in DOCS:
        assert sent in read(doc), (doc, sent)


def test_attribution_and_edit_size_claims():
    w = {x["family"]: round(x["attribution"], 2) for x in RES["windows"]}
    assert (w["shallower"], w["default"], w["circuit"], w["deeper"], w["deepest"]) == (0.13, 0.23, 0.30, 0.13, 0.15)
    rows = json.load(open(os.path.join(ROOT, "results", "raw", "edit_vs_rounding.json")))
    ids = {p["id"]: (p["family"], p["seed"]) for p in RES["frontier"]}
    rng = lambda fam: [r["mean"] for r in rows if ids[r["id"]][0] == fam]
    sent = f"the layer-10-to-12 edits are {min(rng('deeper')):.2f} to {max(rng('deeper')):.2f} times its size, and the CircuitKIT-layer edits that hold are {min(rng('circuit')):.2f} to {max(rng('circuit')):.2f}"
    for doc in DOCS:
        assert sent in read(doc), (doc, sent)


def test_circuitkit_window_claim():
    """'7 to 9 has the most MLP attribution of any three-layer window in layers 1 to 12.'"""
    m = RES["circuit"]["mlp_abs"]
    sums = {i: sum(m[i:i + 3]) for i in range(1, 11)}        # windows starting at layers 1..10 end at 12
    assert max(sums, key=sums.get) == 7
    assert round(sums[7], 2) == 0.30 and round(sums[5], 2) == 0.23


def test_generated_tables_are_filled_in():
    for doc in DOCS:
        for name in ("recipes", "windows") + (("results",) if doc == "README.md" else ()):
            body = re.search(rf"<!-- {name}:start -->(.*?)<!-- {name}:end -->", read(doc), flags=re.S).group(1)
            assert body.count("\n|") >= 4, (doc, name)
