import numpy as np
import pytest
import torch

from relapse.precision import FAKE_QUANT, is_quant_target, roundtrip

gguf = pytest.importorskip("gguf")


@pytest.mark.parametrize("name,qname", [("q4_0", "Q4_0"), ("q4_1", "Q4_1")])
def test_fake_quant_is_bit_exact_with_llama_cpp(name, qname):
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import dequantize, quantize

    x = (np.random.default_rng(0).standard_normal((64, 256)) * 0.02).astype(np.float32)
    ref = dequantize(quantize(x, getattr(Q, qname)), getattr(Q, qname)).reshape(x.shape)
    ours = FAKE_QUANT[name](torch.from_numpy(x)).numpy()
    assert np.array_equal(ref, ours)


def test_roundtrip_fp16_is_identity_and_bf16_rounds():
    t = torch.randn(8, 32) * 0.02
    assert torch.equal(roundtrip(t, "fp16"), t)
    b = roundtrip(t, "bf16")
    assert torch.equal(b, t.to(torch.bfloat16).float())
    assert not torch.equal(b, t)


def test_small_deltas_vanish_more_at_lower_bits():
    rng = torch.Generator().manual_seed(0)
    w = torch.randn(256, 256, generator=rng) * 0.02
    d = torch.randn(256, 256, generator=rng) * 1e-5
    q8 = (roundtrip(w + d, "Q8_0") != roundtrip(w, "Q8_0")).float().mean()
    q4 = (roundtrip(w + d, "Q4_0") != roundtrip(w, "Q4_0")).float().mean()
    assert q4 < q8


def test_quant_targets():
    assert is_quant_target("model.layers.3.mlp.down_proj.weight", torch.zeros(2, 2))
    assert not is_quant_target("model.embed_tokens.weight", torch.zeros(2, 2))
    assert not is_quant_target("model.layers.3.input_layernorm.weight", torch.zeros(2))
