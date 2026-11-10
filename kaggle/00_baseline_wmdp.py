import os, json, glob, time, random, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
t0=time.time()
MP=os.path.dirname([c for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "global" in c][0])
tok=AutoTokenizer.from_pretrained(MP); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
L=["A","B","C","D"]
ids=[tok.encode(" "+l, add_special_tokens=False)[-1] for l in L]
ids2=[tok.encode(l, add_special_tokens=False)[-1] for l in L]
print("letter ids", ids, ids2)
def fmt(subject,q,choices):
    s=f"The following are multiple choice questions (with answers) about {subject}.\n\n{q.strip()}\n"
    for l,c in zip(L,choices): s+=f"{l}. {c}\n"
    return s+"Answer:"
def evaluate(model, items, bs=8, maxlen=1024):
    correct=0; out=[]
    items=sorted(items, key=lambda x: len(x[0]))
    for b in range(0,len(items),bs):
        ch=items[b:b+bs]
        enc=tok([x[0] for x in ch], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(model.device)
        with torch.no_grad():
            lg=model(**enc, logits_to_keep=1).logits[:,-1,:].float()
        p=torch.log_softmax(lg[:,ids],-1)
        for x,row in zip(ch,p):
            pred=int(row.argmax()); correct+= pred==x[1]; out.append((pred,x[1],row.tolist()))
    return correct/len(items), out
sets={}
for cfg,subj in [("wmdp-bio","biology"),("wmdp-cyber","computer security"),("wmdp-chem","chemistry")]:
    d=load_dataset("cais/wmdp",cfg,split="test")
    sets[cfg]=[(fmt(subj,r["question"],r["choices"]),r["answer"]) for r in d]
mm=load_dataset("cais/mmlu","all",split="test").shuffle(seed=0).select(range(1500))
sets["mmlu1500"]=[(fmt(r["subject"].replace("_"," "),r["question"],r["choices"]),r["answer"]) for r in mm]
res={}
for prec in ["fp16","nf4"]:
    torch.cuda.empty_cache()
    if prec=="fp16":
        model=AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.float16, device_map={"":0}).eval()
    else:
        from transformers import BitsAndBytesConfig
        import subprocess,sys; subprocess.run([sys.executable,"-m","pip","install","-q","bitsandbytes"])
        model=AutoModelForCausalLM.from_pretrained(MP, quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16), device_map={"":0}).eval()
    for k,v in sets.items():
        acc,_=evaluate(model,v)
        res[f"{prec}/{k}"]=acc; print(prec,k,len(v),round(acc,4), "t=",round(time.time()-t0))
    del model; torch.cuda.empty_cache()
json.dump(res,open("/kaggle/working/res.json","w"),indent=1)
print(res)
