# ==== main: CPU validation of the llama.cpp GGUF round trip for Tiny Aya (cohere2) ====
from transformers import AutoTokenizer, AutoModelForCausalLM
import collections
MP = model_path("global")
setup_llamacpp(build=True)
hf_to_gguf(MP, "/tmp/base-f16.gguf", "f16")
tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16, device_map={"": "cpu"}).eval()
orig = {n: p.detach().clone() for n, p in model.named_parameters()}
items = load_sets(["wmdp-bio"])["wmdp-bio"][:40]
acc0, _, _ = mcq_eval(model, tok, items, bs=4); log("hf fp32 40q acc", acc0)
q = load_gguf_into(model, "/tmp/base-f16.gguf")
diffs = {n: float((p.detach().float() - orig[n].float()).abs().max()) for n, p in model.named_parameters()}
log("f16 gguf roundtrip: max abs diff", max(diffs.values()), "n>1e-3:", sum(d > 1e-3 for d in diffs.values()), [k for k, v in diffs.items() if v > 1e-3][:5])
acc1, _, _ = mcq_eval(model, tok, items, bs=4); log("gguf f16 40q acc", acc1)
for qt in ["Q8_0", "Q4_K_M"]:
    path = f"/tmp/base-{qt}.gguf"; quantize_gguf("/tmp/base-f16.gguf", path, qt)
    qtypes = load_gguf_into(model, path)
    mix = collections.Counter(qtypes.values()); log(qt, "mix", dict(mix))
    dn = [k for k, v in qtypes.items() if "down_proj" in k]
    log(qt, "down_proj types by layer:", {k.split(".")[2]: qtypes[k] for k in dn})
    err = {n: float((p.detach().float() - orig[n].float()).abs().mean()) for n, p in model.named_parameters() if is_quant_target(n, p)}
    log(qt, "mean abs weight err", sum(err.values()) / len(err))
    acc, _, _ = mcq_eval(model, tok, items, bs=4); log(qt, "40q acc", acc)
json.dump({"ok": True}, open("/kaggle/working/ok.json", "w"))
