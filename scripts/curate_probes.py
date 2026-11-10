"""Curate the translated / reworded WMDP-bio probe sets with CuratorKIT.

Input: probes_raw.json from kaggle/03_probe_translations.py (Tiny Aya Global translations with back-translation chrF).
Gate 1 (ours): strip letter prefixes the translator copied into options; keep a translation only when its
back-translation scores chrF >= 35 against the English source; keep parsed paraphrases that differ from the source.
Gate 2 (CuratorKIT): SchemaGate, exact + MinHash dedup and TextCleaner (strip_html off so nothing in a question
is silently deleted), with CuratorKIT's manifest, provenance and dataset card per language.

    python scripts/curate_probes.py probes_raw.json out_dir
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

LET = "ABCD"
PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d][\)\.:]\s+)")


def main(raw_path: str, out: str, min_chrf: float = 35.0) -> None:
    from curatorkit import Curator, CuratorConfig

    rows = json.load(open(raw_path))
    by_lang = collections.defaultdict(list)
    for r in rows:
        if "question" not in r:
            continue
        keep = (r["kind"] == "paraphrase" and r["status"] == "kept") or \
               (r["kind"] == "translation" and (r.get("backtranslation_chrf") or 0) >= min_chrf)
        if not keep:
            continue
        choices = [PREFIX.sub("", c).strip() for c in r["choices"]]
        item = r["question"].strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LET, choices))
        by_lang["en_para" if r["lang"] == "en" else r["lang"]].append(
            {"instruction": item, "input": "", "output": LET[r["answer"]], "source_id": r["source_id"],
             "kind": r["kind"], "chrf": r.get("backtranslation_chrf")})
    os.makedirs(out, exist_ok=True)
    final, report = [], {}
    for lang, items in sorted(by_lang.items()):
        src = os.path.join(out, f"gate1_{lang}.json")
        json.dump(items, open(src, "w"), ensure_ascii=False)
        cfg = CuratorConfig(dataset=src, format="alpaca", min_tokens=5, max_tokens=2048, dedup="minhash",
                            minhash_threshold=0.9, clean=True, output_dir=os.path.join(out, "curatorkit", lang),
                            export_formats=["alpaca"], name=f"relapse-wmdp-bio-probes-{lang}")
        try:
            cfg.clean_transforms = [t for t in (cfg.clean_transforms or []) if "html" not in str(t).lower()] or cfg.clean_transforms
        except Exception:
            pass
        Curator(cfg).run()
        kept = {json.loads(l)["instruction"] for l in open(os.path.join(out, "curatorkit", lang, "sft_alpaca.jsonl"))}
        for it in items:
            if it["instruction"] in kept:
                final.append({"source_id": it["source_id"], "lang": lang, "kind": it["kind"], "item": it["instruction"],
                              "answer": LET.index(it["output"]), "backtranslation_chrf": it["chrf"]})
        report[lang] = {"after_gate1": len(items), "after_curatorkit": len(kept)}
        os.remove(src)
    json.dump(final, open(os.path.join(out, "probes.json"), "w"), ensure_ascii=False)
    json.dump(report, open(os.path.join(out, "curation_report.json"), "w"), indent=1)
    print(report, len(final))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
