"""AuditKIT-owned MCQ scoring for an IN-MEMORY Hugging Face model (weights may be overwritten between runs).

Path chosen: AuditKIT's native loglikelihood-MCQ stack, no generation hack:
    Sample(input=<prompt ending 'Answer:'>, choices=[' A',' B',' C',' D'], target=<idx>, kind=MCQ)
    -> MCQAdapter('mcq_loglikelihood')  (4 Requests per item: prompt + target=' A'..' D')
    -> InMemoryHFMCQModel.loglikelihood()  (our thin Model backend: next-token logprob of each letter)
    -> Runner argmax over the 4 logprobs -> Acc metric  -> RunResult (save()/to_dict()/summary())
    -> ak.compare(base, other) -> RunComparison (retention, flips) + paired bootstrap p-value.
We add only what AuditKIT lacks: a percentile CI on the paired delta and a weights digest in Model.identity().
"""
from __future__ import annotations
import hashlib, math, random
from typing import Any, Sequence
import torch
import auditkit as ak
from auditkit.adapter import MCQAdapter
from auditkit.metric import Acc
from auditkit.model import Model, LogLikelihood
from auditkit.runner import Runner
from auditkit.runspec import RunConfig, RunSpec
from auditkit.sample import Sample
from auditkit.scenario import ListScenario
from auditkit.types import Capability, TaskKind
from auditkit._bootstrap import score_pairs, paired_bootstrap

LETTERS = [" A", " B", " C", " D"]


def weights_digest(model, n_per_tensor: int = 64) -> str:
    """Cheap digest of the CURRENT weights. Needed because AuditKIT's DiskCache keys runs by
    Model.identity(); the same in-memory object with overwritten weights would otherwise be served
    the previous run's cached result (stale-cache hazard)."""
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            flat = p.detach().reshape(-1)
            step = max(1, flat.numel() // n_per_tensor)
            h.update(name.encode()); h.update(flat[::step][:n_per_tensor].float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class InMemoryHFMCQModel(Model):
    """AuditKIT Model backend around a live HF causal LM + tokenizer (no loading, no copies)."""
    name = "inmem-hf"
    is_local = True

    def __init__(self, model, tok, tag: str = "", max_len: int | None = None):
        self.model, self.tok, self.tag, self.max_len = model, tok, tag, max_len
        self._model_name = getattr(getattr(model, "config", None), "_name_or_path", "inmem")

    def capabilities(self):
        return {Capability.GENERATE, Capability.LOGLIKELIHOOD}

    def identity(self) -> dict:
        return {"name": self.name, "model": self._model_name, "tag": self.tag,
                "weights": weights_digest(self.model)}

    def model_info(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters())
        ps = list(self.model.parameters()); nz = sum(int((p != 0).sum()) for p in ps)
        return {"is_local": True, "model_name": self._model_name, "total_params": n, "nonzero_params": nz,
                "sparsity": 1 - nz / n, "size_mb": sum(p.numel() * p.element_size() for p in ps) / 2**20, "tag": self.tag}

    def generate(self, requests):  # not used for MCQ-by-logprob; greedy 1-token for completeness
        from auditkit.model import Result_, Generated
        out = []
        for r in requests:
            ids = self.tok(r.prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                nxt = self.model(**ids).logits[0, -1].argmax().item()
            out.append(Result_(completions=[Generated(text=self.tok.decode([nxt]))]))
        return out

    def _ids(self, prompt: str, target: str):
        ctx = self.tok(prompt, add_special_tokens=True)["input_ids"]
        full = self.tok(prompt + target, add_special_tokens=True)["input_ids"]
        if full[:len(ctx)] == ctx and len(full) > len(ctx):
            return ctx, full[len(ctx):]
        return ctx, self.tok(target, add_special_tokens=False)["input_ids"]  # fallback

    @torch.no_grad()
    def loglikelihood(self, requests, bs: int = 8):
        """Batched: one left-padded forward per unique prompt (last position only), reused for its 4 letters."""
        dev = next(self.model.parameters()).device
        self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        tgt_ids = {}
        for r in requests:
            _, tgt = self._ids(r.prompt, r.params["target"])
            tgt_ids[(r.prompt, r.params["target"])] = tgt
        want = sorted({t[0] for t in tgt_ids.values() if len(t) == 1})
        uniq = sorted({r.prompt for r in requests}, key=len)
        cache = {}
        for b in range(0, len(uniq), bs):
            ps = uniq[b:b + bs]
            enc = self.tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len or 1024).to(dev)
            lp = torch.log_softmax(self.model(**enc, logits_to_keep=1).logits[:, -1, :].float(), -1)
            am = lp.argmax(-1).tolist(); sel = lp[:, want].cpu().tolist()
            for p, a, row in zip(ps, am, sel):
                cache[p] = (a, dict(zip(want, row)))
        out = []
        for r in requests:
            tgt = tgt_ids[(r.prompt, r.params["target"])]
            a, d = cache[r.prompt]
            if len(tgt) == 1 and tgt[0] in d:
                out.append(LogLikelihood(logprob=float(d[tgt[0]]), is_greedy=bool(a == tgt[0])))
            else:
                raise ValueError("multi-token MCQ targets are not used in Relapse")
        return out


def audit_mcq(model, tok, items: Sequence[tuple[str, int]], run_name: str, *, max_len: int | None = None,
              tag: str | None = None) -> "ak.report.RunResult":
    """items = [(prompt_ending_in_'Answer:', answer_idx)]. Returns an AuditKIT RunResult (headline {'acc': ..}).
    Per-prediction `choice_likelihoods` (back-filled, see bug note) = raw logprobs of ' A'..' D'; `metadata['p_letters']` = softmax over those 4."""
    samples = [Sample(input=p, target=int(a), id=str(i), task="mcq", kind=TaskKind.MCQ, choices=list(LETTERS))
               for i, (p, a) in enumerate(items)]
    spec = RunSpec(scenario=ListScenario(samples, name=f"mcq_{len(samples)}"),
                   model=InMemoryHFMCQModel(model, tok, tag=tag or run_name, max_len=max_len),
                   adapter=MCQAdapter("mcq_loglikelihood"), metrics=[Acc()],
                   config=RunConfig(track_performance=True), run_name=run_name)
    res = Runner().run(spec)
    for p in res.predictions:       # AuditKIT bug: Prediction.choice_likelihoods stays None (only in p.context); fill it + probs
        lp = p.choice_likelihoods or (p.context or {}).get("choice_likelihoods")
        if lp:
            p.choice_likelihoods = list(lp)
            m = max(lp); e = [math.exp(x - m) for x in lp]; z = sum(e)
            p.metadata["p_letters"] = [x / z for x in e]
            p.metadata["pred_letter"] = "ABCD"[max(range(4), key=lambda i: lp[i])]
    res.metadata = {**res.metadata, "run_name": run_name}
    return res


def audit_compare(run_a, run_b, metric: str = "acc", n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Delta = run_b - run_a on `metric`, via ak.compare + AuditKIT's by-sample-id paired bootstrap (p-value);
    plus a percentile 95% CI of the paired delta (AuditKIT reports only p) and McNemar-style flip counts."""
    rc = ak.compare(run_a, run_b)
    sig = rc.significance(metric)                      # {n, mean_baseline, mean_candidate, delta, p_value, significant}
    pairs = score_pairs(run_a, run_b, metric)
    diffs = [c - b for b, c in pairs]; n = len(diffs); rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot)) if n else [float("nan")]
    ret = rc.retention(metric)
    return {"a": run_a.run_id, "b": run_b.run_id, "metric": metric, "n": n,
            "acc_a": sig.get("mean_baseline"), "acc_b": sig.get("mean_candidate"), "delta": sig.get("delta"),
            "ci95": [boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]],
            "p_value": sig.get("p_value"), "significant": sig.get("significant"),
            "right_to_wrong": sum(1 for b, c in pairs if b > c), "wrong_to_right": sum(1 for b, c in pairs if c > b),
            "retention": ret if not hasattr(ret, "__dict__") else vars(ret), "summary": rc.summary()}
