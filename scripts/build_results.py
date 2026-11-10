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
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "grid_*.json"))):
        data = json.load(open(f))
        rows += [dict(r, _file=os.path.basename(f)) for r in data if "wmdp-bio" in r]
    return rows


def norm_prec(p: str) -> str:
    return "NF4" if p.lower() == "nf4" else p


def build(rows: list[dict], methods: dict, template: dict) -> dict:
    grid: dict = {}
    for r in rows:
        m = methods["runs"].get(r["model"])
        if not m:
            continue
        f = norm_prec(r["prec"])
        cell = {"forget": round(r["wmdp-bio"], 4), "retain": round(r["mmlu"], 4)}
        for k in ("wmdp-bio_pcorrect", "mmlu_pcorrect"):
            if k in r:
                cell[k.replace("wmdp-bio", "forget").replace("mmlu", "retain")] = round(r[k], 4)
        for k, v in r.items():
            if k.startswith("probe:"):
                cell.setdefault("probes", {})[k.split(":", 1)[1]] = round(v, 4)
        grid.setdefault(m, {})[f] = cell
    out = dict(template)
    out.pop("_provisional", None)
    out.pop("_note", None)
    out["grid"] = grid
    out["formats"] = [x for x in template["formats"] if any(x["id"] in g for g in grid.values())]
    out["run"] = methods.get("run", out.get("run", {}))
    out["stats"] = load_stats(methods)
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
    if os.path.exists(README):
        txt = open(README).read()
        new = re.sub(r"(<!-- results:start -->\n).*?(\n<!-- results:end -->)", lambda m: m.group(1) + table(res) + m.group(2), txt, flags=re.S)
        open(README, "w").write(new)
    print(f"wrote {OUT} ({sum(len(v) for v in res['grid'].values())} cells)")
    print(table(res))


if __name__ == "__main__":
    main()
