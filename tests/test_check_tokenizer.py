"""scripts/check_tokenizer.py: the two public copies named in the paper give opposite answers, and the GGUF header reader finds the pre-tokenizer."""
import json
import os
import struct

import pytest

from check_tokenizer import OUTDATED_CLASSES, check, gguf_header
from survey_copies import TWO_COPIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf(path, pre: str | None) -> str:
    """A GGUF header with the fields in the order llama.cpp writes them, a token list in front of the pre-tokenizer to be skipped."""
    pairs = [_string("general.architecture") + struct.pack("<I", 8) + _string("cohere2"),
             _string("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 2),
             _string("tokenizer.ggml.model") + struct.pack("<I", 8) + _string("gpt2"),
             _string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 300) + b"".join(_string(f"tok{i}" * (1 + i % 40)) for i in range(300)),
             _string("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, 300) + struct.pack("<300i", *range(300))]
    if pre is not None:
        pairs.append(_string("tokenizer.ggml.pre") + struct.pack("<I", 8) + _string(pre))
    open(path, "wb").write(struct.pack("<4sIQQ", b"GGUF", 3, 0, len(pairs)) + b"".join(pairs))
    return str(path)


def test_gguf_header_names_the_pre_tokenizer(tmp_path):
    header, read = gguf_header(_gguf(tmp_path / "a.gguf", "tiny_aya"))
    assert header["tokenizer.ggml.pre"] == "tiny_aya" and header["general.architecture"] == "cohere2" and read > 0
    assert check(_gguf(tmp_path / "b.gguf", "tiny_aya"))["setting"] == "official"
    assert check(_gguf(tmp_path / "c.gguf", "default"))["setting"] == "outdated"
    assert check(_gguf(tmp_path / "d.gguf", None))["setting"] == "unknown"


def test_a_folder_is_judged_by_its_tokenizer_class(tmp_path):
    for cls, want in (("CohereTokenizerFast", "outdated"), ("CohereTokenizer", "outdated"), ("PreTrainedTokenizerFast", "official"), (None, "unknown")):
        d = tmp_path / str(cls); d.mkdir()
        json.dump({"tokenizer_class": cls} if cls else {}, open(d / "tokenizer_config.json", "w"))
        assert check(str(d))["setting"] == want


def test_the_two_copies_were_recorded_with_opposite_answers():
    rec = json.load(open(os.path.join(ROOT, "results", "survey", "two_copies.json")))["copies"]
    assert [r["target"] for r in rec] == list(TWO_COPIES)
    assert [r["setting"] for r in rec] == ["official", "outdated"]


def test_the_two_copies_give_opposite_answers_today():
    """Live: Cohere's Hugging Face repo and the Kaggle Models copy. Skipped without network access or without access to the gated repo."""
    official, kaggle = (check(t) for t in TWO_COPIES)
    if "unknown" in (official["setting"], kaggle["setting"]):
        pytest.skip(f"could not read a copy: {official['evidence']} / {kaggle['evidence']}")
    assert official["setting"] == "official" and "PreTrainedTokenizerFast" in official["evidence"]
    assert kaggle["setting"] == "outdated" and any(c in kaggle["evidence"] for c in OUTDATED_CLASSES)
    assert max(official["read"], kaggle["read"]) < 100_000      # tokenizer_config.json only
