# ==== common.py (inlined into every Kaggle kernel) ====
import os, sys, json, glob, time, random, subprocess, re, gc
import torch
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def sh(cmd, check=True):
    log("$", cmd[:200])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        if check: raise RuntimeError(f"cmd failed: {cmd[:100]}")
    return r.stdout
def model_path(variant="global"):
    c = [x for x in glob.glob("/kaggle/input/**/config.json", recursive=True) if f"/{variant}/" in x]
    return os.path.dirname(c[0])

# ---------- MCQ eval ----------
LET = ["A", "B", "C", "D"]
def fmt_mcq(subject, q, choices):
    s = f"The following are multiple choice questions (with answers) about {subject}.\n\n{q.strip()}\n"
    for l, c in zip(LET, choices): s += f"{l}. {c}\n"
    return s + "Answer:"
def load_sets(names, mmlu_n=1500, seed=0):
    from datasets import load_dataset
    sets = {}
    for n in names:
        if n.startswith("wmdp-"):
            subj = {"wmdp-bio": "biology", "wmdp-cyber": "computer security", "wmdp-chem": "chemistry"}[n]
            d = load_dataset("cais/wmdp", n, split="test")
            sets[n] = [(fmt_mcq(subj, r["question"], r["choices"]), r["answer"]) for r in d]
        elif n == "mmlu":
            d = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(mmlu_n))
            sets[n] = [(fmt_mcq(r["subject"].replace("_", " "), r["question"], r["choices"]), r["answer"]) for r in d]
        elif n.startswith("mmlu:"):  # mmlu:college_biology,virology
            subs = n.split(":")[1].split(",")
            rows = []
            for sub in subs:
                d = load_dataset("cais/mmlu", sub, split="test")
                rows += [(fmt_mcq(sub.replace("_", " "), r["question"], r["choices"]), r["answer"]) for r in d]
            sets[n] = rows
    return sets
_LETTER_IDS = None
def letter_ids(tok):
    global _LETTER_IDS
    if _LETTER_IDS is None:
        _LETTER_IDS = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LET]
    return _LETTER_IDS
@torch.no_grad()
def mcq_eval(model, tok, items, bs=8, maxlen=1024):
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ids = letter_ids(tok)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    preds = [None] * len(items); probs = [None] * len(items)
    dev = next(model.parameters()).device
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([items[i][0] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(dev)
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        p = torch.softmax(lg[:, ids], -1).cpu()
        for j, i in enumerate(idx):
            preds[i] = int(p[j].argmax()); probs[i] = [round(float(x), 4) for x in p[j]]
    acc = sum(int(preds[i] == items[i][1]) for i in range(len(items))) / len(items)
    return acc, preds, probs

# ---------- GGUF round trip ----------
LLAMA = "/tmp/llama.cpp"
def setup_llamacpp(build=True):
    if not os.path.exists(LLAMA):
        sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA}")
        # tiny-aya-global's tokenizer hashes differently from tiny-aya-base under transformers 5.x;
        # it is the same tokenizer family, so map an unknown hash to the tiny_aya pre-tokenizer.
        bp = f"{LLAMA}/conversion/base.py"
        src = open(bp).read()
        src = src.replace('raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
                          'res = "tiny_aya"; logger.warning("unknown chkhsh %s -> tiny_aya", chkhsh)')
        open(bp, "w").write(src)
    binp = f"{LLAMA}/build/bin/llama-quantize"
    if build and not os.path.exists(binp):
        pre = glob.glob("/kaggle/input/**/llama-quantize", recursive=True)
        os.makedirs(os.path.dirname(binp), exist_ok=True)
        if pre:
            sh(f"cp {pre[0]} {binp} && chmod +x {binp}")
        else:
            sh(f"cd {LLAMA} && cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF && cmake --build build --target llama-quantize -j 4")
        os.makedirs("/kaggle/working/bin", exist_ok=True)
        sh(f"cp {binp} /kaggle/working/bin/llama-quantize", check=False)
    sys.path.insert(0, f"{LLAMA}/gguf-py")
    log("llama.cpp ready", os.path.exists(binp))
def hf_to_gguf(hf_dir, out, outtype="f32"):
    sh(f"cd {LLAMA} && python convert_hf_to_gguf.py {hf_dir} --outtype {outtype} --outfile {out}")
    log("gguf", out, os.path.getsize(out) / 1e9, "GB")
def quantize_gguf(src, dst, qtype):
    sh(f"{LLAMA}/build/bin/llama-quantize {src} {dst} {qtype}")
    log("quantized", qtype, os.path.getsize(dst) / 1e9, "GB")
def gguf_name_map(model):
    import gguf
    n_blocks = model.config.num_hidden_layers
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.COHERE2, n_blocks)
    m = {}
    for name, _ in model.named_parameters():
        g = tmap.get_name(name, try_suffixes=(".weight", ".bias"))
        if g is not None: m[g] = name
    return m
@torch.no_grad()
def load_gguf_into(model, path, ref_sd=None):
    """Overwrite model params with dequantized GGUF tensors. Returns dict of qtypes."""
    import gguf
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(path)
    nm = gguf_name_map(model)
    params = dict(model.named_parameters())
    qtypes = {}; done = set()
    for t in r.tensors:
        hf = nm.get(t.name)
        if hf is None:
            continue
        arr = dequantize(t.data, t.tensor_type)
        arr = arr.reshape(list(reversed([int(x) for x in t.shape])))
        p = params[hf]
        if tuple(arr.shape) != tuple(p.shape):
            raise RuntimeError(f"shape mismatch {t.name}->{hf}: {arr.shape} vs {tuple(p.shape)}")
        p.copy_(torch.from_numpy(arr.astype("float32")).to(p.dtype))
        qtypes[hf] = t.tensor_type.name; done.add(hf)
    missing = [n for n in params if n not in done]
    log("loaded gguf", os.path.basename(path), "tensors", len(done), "missing", missing[:5])
    return qtypes

# ---------- precision simulation (round-trip of weights) ----------
def _gguf_rt(t, qname):
    import numpy as np
    sys.path.insert(0, f"{LLAMA}/gguf-py")
    from gguf.quants import quantize, dequantize
    from gguf.constants import GGMLQuantizationType as Q
    qt = getattr(Q, qname)
    a = t.detach().float().cpu().numpy()
    return torch.from_numpy(dequantize(quantize(a, qt), qt).reshape(a.shape))
def _nf4_rt(t):
    import bitsandbytes.functional as F
    x = t.detach().to("cuda", torch.float16)
    q, st = F.quantize_4bit(x, blocksize=64, quant_type="nf4")
    return F.dequantize_4bit(q, st).float().cpu()
def roundtrip(t, prec):
    """Return fp32 CPU tensor = weights after storing at `prec`."""
    if prec in ("fp32", "fp16"): return t.detach().float().cpu()
    if prec == "bf16": return t.detach().to(torch.bfloat16).float().cpu()
    if prec == "nf4": return _nf4_rt(t)
    return _gguf_rt(t, prec)  # Q8_0, Q4_0, Q4_1, Q5_0, Q5_1
def is_quant_target(name, p):
    return p.dim() == 2 and ".layers." in name and name.endswith("_proj.weight")
def roundtrip_many(named, prec, workers=4):
    """named: dict name->tensor. Returns dict name->fp32 cpu tensor after storing at prec."""
    if prec in ("fp32", "fp16", "bf16", "nf4"):
        return {n: roundtrip(t, prec) for n, t in named.items()}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        futs = {n: ex.submit(roundtrip, t, prec) for n, t in named.items()}
        return {n: f.result() for n, f in futs.items()}
def get_texts(kind, n, seed=0):
    from datasets import load_dataset
    rnd = random.Random(seed)
    if kind == "bio":
        fp = glob.glob("/kaggle/input/**/bio_forget*.jsonl", recursive=True)[0]
        t = [json.loads(l)["text"] for l in open(fp)]
    elif kind == "cyber":
        t = [r["text"] for r in load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")]
    elif kind == "wiki":
        t = [r["text"] for r in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")]
    t = [x for x in t if len(x) > 50]; rnd.shuffle(t)
    return t[:n]

# ---------- fake quantizers (torch) for quantization-aware unlearning ----------
_NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                     -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                     0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                     0.7229568362236023, 1.0])
def fq_q4_0(W, bs=32):
    x = W.reshape(-1, bs)
    idx = x.abs().argmax(dim=1, keepdim=True)
    amax = torch.gather(x, 1, idx)
    d = amax / -8.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.floor(x * idv + 8.5), 0, 15)
    return ((q - 8.0) * d.half().float()).reshape(W.shape)
def fq_q4_1(W, bs=32):
    x = W.reshape(-1, bs)
    mn = x.min(dim=1, keepdim=True).values; mx = x.max(dim=1, keepdim=True).values
    d = (mx - mn) / 15.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.floor((x - mn) * idv + 0.5), 0, 15)
    return (q * d.half().float() + mn.half().float()).reshape(W.shape)
def fq_nf4(W, bs=64):
    x = W.reshape(-1, bs)
    am = x.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    code = _NF4.to(W.device, W.dtype)
    q = torch.bucketize((x / am).contiguous(), (code[1:] + code[:-1]) / 2)
    return (code[q] * am).reshape(W.shape)
FQ = {"q4_0": fq_q4_0, "q4_1": fq_q4_1, "nf4": fq_nf4}
class FakeQuantSTE(torch.nn.Module):
    """Parametrization: with prob p, forward sees the weight after a random 4-bit round trip (straight-through)."""
    def __init__(self, p=0.5, kinds=("q4_0", "q4_1", "nf4")):
        super().__init__(); self.p = p; self.kinds = list(kinds); self.enabled = True
    def forward(self, W):
        if not self.enabled or random.random() >= self.p:
            return W
        with torch.no_grad():
            Wq = FQ[random.choice(self.kinds)](W.detach())
        return W + (Wq - W).detach()

# ---------- the model seen as its 4-bit file during training ----------
class FileView:
    """Shared switch. When on, every decoder linear weight is seen after its Q4_0 round trip."""
    on = False
class SwapQ(torch.nn.Module):
    """Parametrization for a weight that is not trained: swap in its stored Q4_0 round trip while the view is on."""
    def __init__(self, wq, view):
        super().__init__(); self.wq = wq; self.view = view      # wq is a plain tensor and may sit on another GPU
    def forward(self, W):
        return self.wq.to(W.device, W.dtype) if self.view.on else W
class EditQ(torch.nn.Module):
    """Parametrization for a trained weight: Q4_0 round trip on the fly while the view is on, straight-through gradient."""
    def __init__(self, view):
        super().__init__(); self.view = view
    def forward(self, W):
        return W + (fq_q4_0(W.detach()) - W.detach()) if self.view.on else W
def proj_linears(m):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and n.endswith("_proj") and "layers." in n]
def attach_file_view(model, frozen, edited, p, seed=0, qdev=None):
    """Train `model` against `frozen` with both seen as their Q4_0 file on a fraction p of the steps.
    `edited` = names of the trained linears in `model`. One coin per step: it is flipped on every second forward of `model`
    (RMU runs the updated model twice per step, then the frozen one). Returns detach(), which restores plain weights."""
    import torch.nn.utils.parametrize as P
    view = FileView(); qdev = qdev or next(frozen.parameters()).device
    with torch.no_grad():
        bufs = {n: fq_q4_0(mod.weight.detach().float()).half().to(qdev) for n, mod in proj_linears(frozen)}
    fm, mm = proj_linears(frozen), proj_linears(model)
    assert [n for n, _ in fm] == [n for n, _ in mm] and set(edited) <= set(bufs), "models differ or an edited name is missing"
    for n, mod in fm: P.register_parametrization(mod, "weight", SwapQ(bufs[n], view), unsafe=True)
    for n, mod in mm: P.register_parametrization(mod, "weight", EditQ(view) if n in edited else SwapQ(bufs[n], view), unsafe=True)
    coin, calls, seen = random.Random(1000 + seed), [0], [0, 0]
    def flip(_m, _a):
        if calls[0] % 2 == 0:
            view.on = coin.random() < p; seen[int(view.on)] += 1
        calls[0] += 1
    hook = model.register_forward_pre_hook(flip)
    def detach():
        hook.remove(); view.on = False
        for _, mod in fm + mm: P.remove_parametrizations(mod, "weight", leave_parametrized=False)
        bufs.clear()
        return {"steps_plain": seen[0], "steps_as_file": seen[1]}
    return detach
