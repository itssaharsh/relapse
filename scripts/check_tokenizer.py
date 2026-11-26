"""Does this copy of Tiny Aya carry the outdated tokenizer setting?

    python scripts/check_tokenizer.py CohereLabs/tiny-aya-global                 # a Hugging Face repo, at main
    python scripts/check_tokenizer.py CohereLabs/tiny-aya-global@00590ff258      # ... at a revision
    python scripts/check_tokenizer.py kaggle:cohere-labs/tiny-aya/transformers/global/1
    python scripts/check_tokenizer.py path/to/model.gguf                         # a GGUF file, local or https://
    python scripts/check_tokenizer.py owner/repo:file.gguf                       # a GGUF file inside a Hugging Face repo
    python scripts/check_tokenizer.py path/to/model_folder --json

What it looks at
  A transformers copy keeps the setting in tokenizer_config.json. Cohere changed `tokenizer_class` from
  `CohereTokenizerFast` to `PreTrainedTokenizerFast` on 30 September 2026 "so transformers v5 loads the pre-tokenizer".
  A copy that still names `CohereTokenizerFast` (or `CohereTokenizer`, the same class in transformers v5, which Tiny Aya Base
  named) is OUTDATED: under transformers v5 the same text is split into different tokens, at line breaks and digit groups
  (see the paper, Section 4.1; --probe shows it).
  A GGUF file has no tokenizer class. llama.cpp splits text by the pre-tokenizer named in the file's header
  (`tokenizer.ggml.pre`). `tiny_aya` (or `cohere2moe`) is the split the model was trained with; anything else is reported
  as OUTDATED, because that file will not split text the way the official tokenizer does.

What it reads
  Only tokenizer_config.json, or the first part of a GGUF header through HTTP range requests. It never downloads a weight file.
  With --probe (needs `transformers` and `tokenizers`) it also fetches tokenizer.json and special_tokens_map.json and reports
  whether the installed transformers splits a neutral multiple-choice prompt the way tokenizer.json itself does.

Exit status: 0 official, 1 outdated, 2 could not tell. As a check at load time:
    from check_tokenizer import check
    assert check("path/or/repo")["setting"] == "official"
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUTDATED_CLASSES = ("CohereTokenizerFast", "CohereTokenizer")      # one class under transformers v5: the second name is an alias of the first
TRAINED_PRE = ("tiny_aya", "cohere2moe")      # the two names llama.cpp maps to Tiny Aya's pre-tokenizer
AGENT = "check-tokenizer/1.0"
HF = "https://huggingface.co"
KAGGLE = "https://www.kaggle.com/api/v1"
MAX_HEADER = 96 * 2 ** 20      # stop reading a GGUF header after this many bytes
PROBE = "The following are multiple choice questions (with answers) about geography.\n\nWhich city had 1,273 bridges in 2024?\nA. Lyon\nB. Porto\nC. Turin\nD. Ghent\nAnswer:"


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """Follows redirects, but does not pass the Authorization header on to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.headers = {k: v for k, v in new.headers.items() if k.lower() != "authorization"}
            new.unredirected_hdrs = {k: v for k, v in new.unredirected_hdrs.items() if k.lower() != "authorization"}
        return new


_OPENER = urllib.request.build_opener(_SameHostAuth)


def _token(host: str) -> str | None:
    if "huggingface.co" in host:
        t = os.environ.get("HF_TOKEN")
        p = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token")
        return t or (open(p).read().strip() if os.path.exists(p) else None)
    if "kaggle.com" in host:
        t = os.environ.get("KAGGLE_API_TOKEN")
        p = os.path.expanduser("~/.kaggle/access_token")
        return t or (open(p).read().strip() if os.path.exists(p) else None)
    return None


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": AGENT}
    tok = _token(urllib.parse.urlsplit(url).netloc)
    if tok:
        headers["Authorization"] = "Bearer " + tok
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(6):
        try:
            with _OPENER.open(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(min(300, int(e.headers.get("Retry-After") or 30) + 2))      # rate limited: wait as long as the server asks


def _get_json(url: str):
    return json.loads(_get(url))


# ---------- GGUF header ----------
class _Reader:
    """Reads forward through a local file or a URL, fetching the URL in ranges and never more than it is asked for."""

    def __init__(self, src: str):
        self.src, self.remote = src, src.startswith(("http://", "https://"))
        self.pos, self.buf, self.buf_at, self.fetched = 0, b"", 0, 0
        self.file = None if self.remote else open(src, "rb")

    def read(self, n: int) -> bytes:
        if not self.remote:
            self.file.seek(self.pos); out = self.file.read(n); self.pos += n; self.fetched += len(out)
            return out
        if not (self.buf_at <= self.pos and self.pos + n <= self.buf_at + len(self.buf)):
            size = max(n, 2 ** 20 if self.fetched == 0 else 4 * 2 ** 20)
            if self.fetched + size > MAX_HEADER:
                raise ValueError("the GGUF header is larger than this tool will read")
            self.buf, self.buf_at = _get(self.src, self.pos, self.pos + size - 1), self.pos
            self.fetched += len(self.buf)
        out = self.buf[self.pos - self.buf_at:self.pos - self.buf_at + n]; self.pos += n
        if len(out) < n:
            raise ValueError("the file ended inside the GGUF header")
        return out

    def skip(self, n: int) -> None:
        self.pos += n      # nothing is fetched for bytes that are skipped


_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _string(r: _Reader, keep: bool = True) -> str | None:
    n = struct.unpack("<Q", r.read(8))[0]
    if n > 2 ** 24:
        raise ValueError("not a GGUF string")
    if not keep:
        r.read(n) if n < 64 else r.skip(n)      # short strings are already in the buffer; long ones are skipped unread
        return None
    return r.read(n).decode("utf-8", "replace")


def _value(r: _Reader, kind: int, keep: bool):
    if kind in _SIZES:
        return struct.unpack(_FORMATS[kind], r.read(_SIZES[kind]))[0] if keep else r.skip(_SIZES[kind])
    if kind == 8:
        return _string(r, keep)
    if kind == 9:
        inner, count = struct.unpack("<IQ", r.read(12))
        if inner in _SIZES:
            r.skip(_SIZES[inner] * count)
        else:
            for _ in range(count):
                _value(r, inner, False)
        return None
    raise ValueError(f"unknown GGUF value type {kind}")


def gguf_header(src: str, want=("general.architecture", "general.name", "tokenizer.ggml.model", "tokenizer.ggml.pre")) -> tuple[dict, int]:
    """The wanted header fields of a GGUF file (local path or URL), and how many bytes were read to get them."""
    r = _Reader(src)
    magic, version, _tensors, pairs = struct.unpack("<4sIQQ", r.read(24))
    if magic != b"GGUF":
        raise ValueError("not a GGUF file")
    found = {"gguf.version": version}
    for _ in range(pairs):
        key = _string(r)
        kind = struct.unpack("<I", r.read(4))[0]
        value = _value(r, kind, key in want)
        if key in want:
            found[key] = value
            if all(k in found for k in want):
                break
    return found, r.fetched


# ---------- the verdict ----------
def _from_config(cfg: dict) -> tuple[str, str]:
    cls = cfg.get("tokenizer_class")
    if cls in OUTDATED_CLASSES:
        return "outdated", f"tokenizer_class={cls}"
    return ("official", f"tokenizer_class={cls}") if cls else ("unknown", "tokenizer_config.json names no tokenizer_class")


def _from_gguf(src: str) -> tuple[str, str, int]:
    h, n = gguf_header(src)
    pre = h.get("tokenizer.ggml.pre")
    if pre is None:
        return "unknown", f"GGUF header names no pre-tokenizer (tokenizer.ggml.model={h.get('tokenizer.ggml.model')})", n
    return ("official" if pre in TRAINED_PRE else "outdated"), f"tokenizer.ggml.pre={pre}", n


def _hf_info(repo: str, revision: str) -> dict:
    return _get_json(f"{HF}/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}")


def check(target: str, probe: bool = False, files: list[str] | None = None) -> dict:
    """Checks one copy. Returns target, kind, revision, setting (official, outdated or unknown), evidence, read (bytes read).

    `files` is the file list of a Hugging Face repo at the given revision, if the caller already has it (saves one request)."""
    out = {"target": target, "kind": None, "revision": None, "setting": "unknown", "evidence": "", "read": 0}
    try:
        if target.startswith("kaggle:"):      # kaggle:owner/model/framework/variation/version
            out["kind"] = "kaggle"; out["revision"] = target.rsplit("/", 1)[-1]
            raw = _get(f"{KAGGLE}/models/{target[7:]}/download/tokenizer_config.json")
            out["setting"], out["evidence"] = _from_config(json.loads(raw)); out["read"] = len(raw)
        elif target.startswith(("http://", "https://")) or (os.path.isfile(target) and target.endswith(".gguf")):
            out["kind"] = "gguf"
            out["setting"], out["evidence"], out["read"] = _from_gguf(target)
        elif os.path.isdir(target) or os.path.isfile(target):
            out["kind"] = "local"
            path = os.path.join(target, "tokenizer_config.json") if os.path.isdir(target) else target
            if os.path.exists(path):
                out["setting"], out["evidence"] = _from_config(json.load(open(path))); out["read"] = os.path.getsize(path)
            else:
                ggufs = sorted(f for f in os.listdir(target) if f.endswith(".gguf"))
                if not ggufs:
                    out["evidence"] = "no tokenizer_config.json and no GGUF file"
                else:
                    out["kind"] = "gguf"; out["setting"], out["evidence"], out["read"] = _from_gguf(os.path.join(target, ggufs[0])); out["evidence"] += f" ({ggufs[0]})"
        else:      # a Hugging Face repo: owner/name, owner/name@revision, owner/name[@revision]:file.gguf
            spec, _, inner = target.partition(":")
            repo, _, revision = spec.partition("@")
            if files is None or not revision:
                info = _hf_info(repo, revision or "main")
                revision, files = info.get("sha", revision or "main"), [s["rfilename"] for s in info.get("siblings", [])]
            sha = revision
            out["kind"], out["revision"] = "hf", sha
            url = lambda f: f"{HF}/{repo}/resolve/{sha}/{urllib.parse.quote(f)}"
            if inner or "tokenizer_config.json" not in files:
                ggufs = [inner] if inner else sorted((f for f in files if f.endswith(".gguf")), key=lambda f: ("mmproj" in f, len(f), f))
                if not ggufs:
                    out["evidence"] = "no tokenizer_config.json and no GGUF file (the tokenizer comes from another repo)"
                else:
                    out["kind"] = "hf-gguf"; out["setting"], out["evidence"], out["read"] = _from_gguf(url(ggufs[0])); out["evidence"] += f" ({ggufs[0]})"
            else:
                raw = _get(url("tokenizer_config.json"))
                out["setting"], out["evidence"] = _from_config(json.loads(raw)); out["read"] = len(raw)
                if probe:
                    out["probe"] = _probe({f: _get(url(f)) for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if f in files})
    except urllib.error.HTTPError as e:
        out["evidence"] = f"HTTP {e.code}" + (" (gated or private: accept the model's terms or set HF_TOKEN)" if e.code in (401, 403) else "")
    except (urllib.error.URLError, ValueError, OSError, KeyError, json.JSONDecodeError, struct.error) as e:
        out["evidence"] = f"{type(e).__name__}: {e}"
    return out


def _probe(files: dict[str, bytes]) -> dict:
    """Splits a neutral multiple-choice prompt with tokenizer.json itself and with the installed transformers; reports where they differ."""
    import tempfile

    import tokenizers
    import transformers
    with tempfile.TemporaryDirectory() as d:
        for name, raw in files.items():
            open(os.path.join(d, name), "wb").write(raw)
        ref = tokenizers.Tokenizer.from_file(os.path.join(d, "tokenizer.json")).encode(PROBE, add_special_tokens=False)
        tok = transformers.AutoTokenizer.from_pretrained(d)
        got = tok(PROBE, add_special_tokens=False)["input_ids"]
    pieces = lambda ids: [tok.decode([i]) for i in ids]
    a, b = pieces(ref.ids), pieces(got)
    only_ref, only_got = [x for x in a if x not in b], [x for x in b if x not in a]
    return {"transformers": transformers.__version__, "class_loaded": type(tok).__name__, "same_split": ref.ids == got, "tokens_as_trained": len(ref.ids), "tokens_as_loaded": len(got),
            "pieces_only_as_trained": sorted(set(only_ref))[:12], "pieces_only_as_loaded": sorted(set(only_got))[:12]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Reports whether a copy of Tiny Aya carries the outdated tokenizer setting. Reads tokenizer files or a GGUF header only.")
    ap.add_argument("targets", nargs="+", help="Hugging Face repo[@revision][:file.gguf], kaggle:owner/model/framework/variation/version, a GGUF file or URL, or a model folder")
    ap.add_argument("--json", action="store_true", help="print one JSON object per target")
    ap.add_argument("--probe", action="store_true", help="also split a neutral prompt with the installed transformers (Hugging Face repos only)")
    a = ap.parse_args()
    worst = 0
    for t in a.targets:
        r = check(t, probe=a.probe)
        worst = max(worst, {"official": 0, "outdated": 1}.get(r["setting"], 2))
        if a.json:
            print(json.dumps(r))
        else:
            print(f"{r['setting'].upper():9s} {t}" + (f" @ {r['revision'][:10]}" if r["revision"] else "") + f"   [{r['evidence']}; read {r['read']:,} bytes]")
            if "probe" in r:
                p = r["probe"]
                print(f"          transformers {p['transformers']} loads {p['class_loaded']}: " + ("the same split as tokenizer.json" if p["same_split"] else
                      f"a different split ({p['tokens_as_trained']} tokens as trained, {p['tokens_as_loaded']} as loaded; pieces only as loaded: {p['pieces_only_as_loaded']})"))
    return worst


if __name__ == "__main__":
    sys.exit(main())
