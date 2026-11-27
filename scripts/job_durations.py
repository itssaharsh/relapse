"""How long every Stage 2 job ran, from Hugging Face's own job record, and what that costs at list prices.

    python scripts/job_durations.py            # writes results/survey/job_durations.json (needs a Hugging Face login that owns the jobs)

For each job: when it was created, when it started and finished, its hardware and its outcome. No job id, command or
setting value is stored. The hourly prices are Hugging Face's list prices for Jobs hardware as shown on
https://huggingface.co/pricing on 27 November 2026; the cost here is hours times list price, not a bill.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES = {"cpu-xl": 1.00, "t4-small": 0.40, "t4-medium": 0.60, "l4x1": 0.80, "l40sx1": 1.80}      # US dollars per hour, https://huggingface.co/pricing, 27 November 2026
stamp = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def main() -> None:
    from huggingface_hub import HfApi
    jobs = []
    for j in HfApi().list_jobs():
        ran = (j.finished_at - j.started_at).total_seconds() if j.started_at and j.finished_at else None
        jobs.append({"created": stamp(j.created_at), "started": stamp(j.started_at), "finished": stamp(j.finished_at), "flavor": str(j.flavor).replace("JobHardware.", "").lower(),
                     "outcome": str(j.status.stage), "experiment": (j.environment or {}).get("EXPERIMENT", ""), "seconds": ran})
    jobs.sort(key=lambda x: x["created"])
    assert all(x["flavor"] in PRICES for x in jobs), "a hardware flavor has no list price here"
    out = {"source": "huggingface_hub HfApi().list_jobs(): created_at, started_at, finished_at, flavor, status", "prices_usd_per_hour": PRICES,
           "prices_source": "https://huggingface.co/pricing, read on 27 November 2026", "jobs": jobs}
    json.dump(out, open(os.path.join(ROOT, "results", "survey", "job_durations.json"), "w"), indent=1)
    hours = {f: sum(x["seconds"] or 0 for x in jobs if x["flavor"] == f) / 3600 for f in PRICES}
    print({f: round(h, 1) for f, h in hours.items()}, "| list-price cost:", round(sum(hours[f] * PRICES[f] for f in PRICES), 2), "| jobs with no recorded start or finish:", sum(x["seconds"] is None for x in jobs))


if __name__ == "__main__":
    main()
