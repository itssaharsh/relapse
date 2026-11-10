"""Rebuild space/data/results.json and the README results table from the raw run files.

Usage:
    python scripts/build_results.py                  # reads results/raw/*.json (download with --fetch)
    python scripts/build_results.py --fetch          # first pulls the raw runs from the Hugging Face dataset

Raw rows look like {"model": <run name>, "prec": <format>, "wmdp-bio": acc, "wmdp-bio_pcorrect": p, "mmlu": acc, ...}.
results/methods.json maps run names to the page's methods (original / standard / fix).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "space", "data", "results.json")
README = os.path.join(ROOT, "README.md")
HF_DATASET = "itssaharsh/relapse-runs"
FORMATS = ["fp16", "bf16", "Q8_0", "Q6_K", "Q4_K_M", "Q4_0", "NF4"]


def load_rows() -> list[dict]:
    """Grid rows first. Sweep rows follow, flagged: they only fill a format the grid runs did not score for that model."""
    rows = []
    for pat, fill in (("grid_*.json", False), ("sweep_*.json", True)):
        for f in sorted(glob.glob(os.path.join(RAW, pat))):
            data = json.load(open(f))
            rows += [dict(r, _file=os.path.basename(f), _fill=fill) for r in data if "wmdp-bio" in r and "mmlu" in r]
    return rows


def norm_prec(p: str) -> str:
    return "NF4" if p.lower() == "nf4" else p


def build(rows: list[dict], methods: dict, template: dict) -> dict:
    grid: dict = {}
    same_run_base = {(r["_file"], norm_prec(r["prec"])): r for r in rows if r["model"] == "base"}
    refs = []
    for r in rows:
        m = methods["runs"].get(r["model"])
        if not m:
            continue
        f = norm_prec(r.get("prec", "fp16"))
        if r.get("_fill") and (m == "original" or f in grid.get(m, {})):
            continue
        cell = {"forget": round(r["wmdp-bio"], 6), "retain": round(r["mmlu"], 6)}
        b = same_run_base.get((r["_file"], f))
        if b is not None and m != "original":
            refs.append((cell, f, {"forget": round(b["wmdp-bio"], 6), "retain": round(b["mmlu"], 6)}))
        for k in ("wmdp-bio_pcorrect", "mmlu_pcorrect"):
            if k in r:
                cell[k.replace("wmdp-bio", "forget").replace("mmlu", "retain")] = round(r[k], 4)
        for k, v in r.items():
            if k.startswith("probe:"):
                cell.setdefault("probes", {})[k.split(":", 1)[1]] = round(v, 4)
        grid.setdefault(m, {})[f] = cell
    # a cell is compared with the original scored in the same run; say so where that differs from the page's original
    # (a real llama.cpp file against the bit-exact round trip of the same format)
    for cell, f, ref in refs:
        o = grid.get("original", {}).get(f)
        if o is None or max(abs(o["forget"] - ref["forget"]), abs(o["retain"] - ref["retain"])) > 6e-5:   # beyond 4-decimal rounding
            cell["orig"] = ref
    out = dict(template)
    out.pop("_provisional", None)
    out.pop("_note", None)
    out["grid"] = grid
    out["formats"] = [x for x in template["formats"] if any(x["id"] in g for g in grid.values())]
    out["run"] = methods.get("run", out.get("run", {}))
    out["stats"] = load_stats(methods)
    out["frontier"] = load_frontier(methods)
    out["families"] = methods.get("families", {})
    out["recipes"] = load_recipes()
    out["windows"] = load_windows(out["frontier"], out["families"])
    rob = load_robustness(methods)
    if rob:
        out["robustness"] = rob
    cpath = os.path.join(RAW, "circuitkit_layers.json")
    if os.path.exists(cpath):
        out["circuit"] = json.load(open(cpath))
    ex = pick_example(methods)
    if ex:
        out["example"] = ex
    elif "example" in methods:
        out["example"] = methods["example"]
    if "languages" in methods:
        out["languages"] = methods["languages"]
    if "links" in methods:
        out["links"] = methods["links"]
    return out


def load_stats(methods: dict) -> list[dict]:
    """AuditKIT paired comparisons (bootstrap CI + p-value) from results/raw/audit_<tag>.json summaries."""
    out, seen = [], set()
    for f in sorted(glob.glob(os.path.join(RAW, "audit_*.json"))):
        data = json.load(open(f))
        if not isinstance(data, list):
            continue
        for d in data:
            m = methods["runs"].get(d.get("model"))
            if not m:
                continue
            row = {k: v for k, v in d.items() if k not in ("retention", "summary", "a", "b")}
            row["method"] = m
            row.setdefault("kind", "forget_holds")
            if row["kind"] == "forget_holds":
                row.setdefault("from", "fp16"); row.setdefault("to", "Q4_K_M")
            key = (m, row["kind"], row.get("prec"), row.get("from"), row.get("to"))
            if key not in seen:
                seen.add(key); out.append(row)
    return out


def load_frontier(methods: dict) -> list[dict]:
    """Every unlearning run we made (tuning sweeps included) as forget/keep points, for the trade-off chart.
    A run that appears in a sweep and in a final grid is taken once per format, the final grid first."""
    fr = methods.get("frontier_runs", {})
    files = sorted(glob.glob(os.path.join(RAW, "grid_final*.json"))) + sorted(glob.glob(os.path.join(RAW, "grid_*.json"))) \
        + sorted(glob.glob(os.path.join(RAW, "sweep_*.json")))
    seen, out, base = set(), [], {}
    for f in files:
        for r in json.load(open(f)):
            if "wmdp-bio" not in r or "mmlu" not in r:
                continue
            prec = norm_prec(r.get("prec", "fp16"))
            if r["model"] == "base":
                base.setdefault(prec, {"forget": round(r["wmdp-bio"], 6), "retain": round(r["mmlu"], 6)})
                continue
            meta = fr.get(r["model"])
            if not meta or (meta["id"], prec) in seen:
                continue
            seen.add((meta["id"], prec))
            out.append({"family": meta["family"], "id": meta["id"], "seed": meta["seed"], "qa": meta["qa"], "prec": prec,
                        "forget": round(r["wmdp-bio"], 6), "retain": round(r["mmlu"], 6)})
    for o in out:
        b = base.get(o["prec"])
        o["price"] = round(b["retain"] - o["retain"], 6) if b else None
    return [{"family": "original", "id": "original", "seed": 0, "qa": False, "prec": k, **v, "price": 0.0} for k, v in base.items()] + out


def load_windows(frontier: list[dict], families: dict, cut: float = 0.40) -> list[dict]:
    """One row per layer window that was edited: CircuitKIT's MLP attribution for it, what forgetting cost there
    (runs at or below the cut-off), and how each run's hazard score moved from fp16 to the 4-bit files."""
    cpath = os.path.join(RAW, "circuitkit_layers.json")
    mlp = json.load(open(cpath))["mlp_abs"] if os.path.exists(cpath) else None
    out = []
    for fam, meta in sorted(families.items(), key=lambda kv: kv[1]["layers"][0]):
        pts = [p for p in frontier if p["family"] == fam]
        if not pts:
            continue
        ids = sorted({p["id"] for p in pts})
        by = {(p["id"], p["prec"]): p for p in pts}
        row = {"family": fam, "label": meta["label"], "layers": meta["layers"], "runs": len(ids),
               "attribution": round(sum(mlp[l] for l in meta["layers"]), 4) if mlp else None, "cut": cut, "cost": {}, "shift": {}}
        for prec in FORMATS:
            ps = [p["price"] for p in pts if p["prec"] == prec and p["forget"] <= cut]
            if ps:
                row["cost"][prec] = {"n": len(ps), "lo": min(ps), "hi": max(ps)}
            d = [round(by[(i, prec)]["forget"] - by[(i, "fp16")]["forget"], 6) for i in ids if prec != "fp16" and (i, prec) in by and (i, "fp16") in by]
            if d:
                row["shift"][prec] = {"n": len(d), "mean": round(sum(d) / len(d), 6), "lo": min(d), "hi": max(d)}
        out.append(row)
    return out


def windows_table(rows: list[dict]) -> str:
    sg = lambda x: f"{100 * x:+.1f}"
    span = lambda c: "none got there" if not c else (f"{100 * c['lo']:.1f}" if c["n"] == 1 else f"{100 * c['lo']:.1f} to {100 * c['hi']:.1f}") + f" ({c['n']} run{'s' if c['n'] > 1 else ''})"
    shift = lambda c: "not run" if not c else f"{sg(c['mean'])} ({sg(c['lo'])} to {sg(c['hi'])})"
    cut = round(100 * rows[0]["cut"])
    out = [f"| Layers edited | CircuitKIT MLP attribution | Runs | MMLU points lost at fp16, runs at {cut}% WMDP-bio or below | WMDP-bio change, fp16 to Q4_0: mean (range) | fp16 to NF4 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        name = f"{r['layers'][0]} to {r['layers'][-1]}" + {"default": " (SafeTune default)", "circuit": " (CircuitKIT)"}.get(r["family"], " (control)")
        out.append(f"| {name} | {r['attribution']:.2f} | {r['runs']} | {span(r['cost'].get('fp16'))} | {shift(r['shift'].get('Q4_0'))} | {shift(r['shift'].get('NF4'))} |")
    return "\n".join(out)


def load_recipes() -> list[dict]:
    """Paired CircuitKIT-layer vs default-layer comparisons (scripts/recipe_stats.py)."""
    f = os.path.join(RAW, "audit_recipes.json")
    return json.load(open(f)) if os.path.exists(f) else []


def recipes_table(rows: list[dict]) -> str:
    sg = lambda x: f"{100 * x:+.1f}"
    fp = lambda p: "< 0.001" if p < 0.0005 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
    out = ["| Seed | File | WMDP-bio: default / CircuitKIT layers | MMLU: default / CircuitKIT layers | MMLU kept extra (95% interval) | p |",
           "|---|---|---|---|---|---|"]
    key = lambda r: (r["seed"], r["prec"] != "fp16")
    for seed, prec in sorted({(r["seed"], r["prec"]) for r in rows}, key=lambda k: (k[0], k[1] != "fp16")):
        m = next(r for r in rows if (r["seed"], r["prec"], r["set"]) == (seed, prec, "mmlu"))
        w = next(r for r in rows if (r["seed"], r["prec"], r["set"]) == (seed, prec, "wmdp-bio"))
        out.append(f"| {seed + 1} | {prec} | {pct(w['acc_a'])} / {pct(w['acc_b'])} | {pct(m['acc_a'])} / {pct(m['acc_b'])} | "
                   f"{sg(m['delta'])} ({sg(m['ci95'][0])} to {sg(m['ci95'][1])}) | {fp(m['p_value'])} |")
    return "\n".join(out)


def load_robustness(methods: dict) -> list[dict]:
    """Adversarial prompting and relearning attacks (results/raw/robustness_*.json), mapped to the page's methods."""
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "robustness_*.json"))):
        for r in json.load(open(f)):
            m = methods["runs"].get(r["model"])
            if m:
                out.append({"method": m, "prec": norm_prec(r["prec"]), "attack": r["attack"], "acc": r["acc"], "n": r["n"]})
    return out


def pick_example(methods: dict):
    """The named example: a WMDP-bio question the original model answers confidently that unlearning removed.
    The question text is withheld on purpose (hazardous); only its id and the four answer probabilities are shown."""
    files = sorted(glob.glob(os.path.join(RAW, "items_*.json")))
    if not files:
        return None
    items = {}
    for f in files:
        for k, v in json.load(open(f)).items():
            parts = k.split("|")
            if len(parts) == 2 or parts[2] == "wmdp-bio":
                items[(parts[0], norm_prec(parts[1]))] = v
    from datasets import load_dataset  # answers only; text is never written out
    answers = load_dataset("cais/wmdp", "wmdp-bio", split="test")["answer"]
    runs = {m: r for r, m in methods["runs"].items()}
    need = [(runs.get("original"), "fp16"), (runs.get("standard"), "Q4_K_M"), (runs.get("circuit"), "Q4_K_M")]
    if any(k not in items for k in need if k[0]):
        need = [k for k in need if k in items]
    best, score = None, -1
    for q in range(len(answers)):
        a = answers[q]
        try:
            po = items[(runs["original"], "fp16")][q][a]
            pu = max(items[k][q][a] for k in need[1:]) if len(need) > 1 else 1
        except (KeyError, IndexError):
            continue
        if po - pu > score:
            best, score = q, po - pu
    if best is None:
        return None
    probs = {}
    for (run, prec), v in items.items():
        m = methods["runs"].get(run)
        if m:
            probs.setdefault(m, {})[prec] = v[best]
    return {"qid": best, "answer": answers[best], "text_withheld": True, "set": "WMDP-bio", "probs": probs}


def pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "n/a"


def table(res: dict) -> str:
    ms = [m["id"] for m in res["methods"]]
    labels = {m["id"]: m.get("short", m["label"]) for m in res["methods"]}
    head = "| File (WMDP-bio / MMLU) | " + " | ".join(labels[m] for m in ms) + " |"
    sep = "|---|" + "---|" * len(ms)
    lines = [head, sep]
    for f in res["formats"]:
        cells = []
        for m in ms:
            c = res["grid"].get(m, {}).get(f["id"])
            cells.append(f"{pct(c['forget'])} / {pct(c['retain'])}" if c else "not run")
        lines.append(f"| {f['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download raw runs from the HF dataset first")
    a = ap.parse_args()
    if a.fetch:
        from huggingface_hub import snapshot_download
        snapshot_download(HF_DATASET, repo_type="dataset", allow_patterns=["raw/*.json"], local_dir=os.path.join(ROOT, "results"))
    methods = json.load(open(os.path.join(ROOT, "results", "methods.json")))
    template = json.load(open(OUT))
    res = build(load_rows(), methods, template)
    json.dump(res, open(OUT, "w"), indent=1)
    fill = lambda text, name, body: re.sub(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", lambda m: m.group(1) + body + "\n" + m.group(2), text, flags=re.S)
    for doc in (README, os.path.join(ROOT, "docs", "ARTICLE.md")):
        if not os.path.exists(doc):
            continue
        txt = open(doc).read()
        if doc == README:
            txt = fill(txt, "results", table(res))
        if res.get("recipes"):
            txt = fill(txt, "recipes", recipes_table(res["recipes"]))
        if res.get("windows"):
            txt = fill(txt, "windows", windows_table(res["windows"]))
        open(doc, "w").write(txt)
    print(f"wrote {OUT} ({sum(len(v) for v in res['grid'].values())} cells)")
    print(table(res))


if __name__ == "__main__":
    main()
