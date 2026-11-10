"""Training with the model seen as its 4-bit file: the switch shows Q4_0 weights, flips once per step, and leaves no trace."""
import torch

from common import attach_file_view, fq_q4_0, proj_linears


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module(); self.self_attn.q_proj = torch.nn.Linear(64, 64, bias=False)
        self.mlp = torch.nn.Module(); self.mlp.down_proj = torch.nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return x + self.mlp.down_proj(torch.tanh(self.self_attn.q_proj(x)))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(3)])

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


def test_file_view_switch():
    torch.manual_seed(0)
    frozen = Toy(); model = Toy(); model.load_state_dict(frozen.state_dict())
    base = {n: p.detach().clone() for n, p in frozen.named_parameters()}
    edited = {"layers.2.mlp.down_proj"}
    detach = attach_file_view(model, frozen, edited, 1.0, seed=0)
    x = torch.randn(2, 64)
    model(x)                                                    # first forward of a step flips the coin; p = 1 turns the view on
    lin, flin = dict(proj_linears(model)), dict(proj_linears(frozen))
    q = lambda n: fq_q4_0(base[n + ".weight"]).half().float()
    assert torch.equal(lin["layers.0.self_attn.q_proj"].weight, q("layers.0.self_attn.q_proj"))
    assert torch.equal(flin["layers.2.mlp.down_proj"].weight, q("layers.2.mlp.down_proj"))        # the frozen reference is rounded too
    assert torch.allclose(lin["layers.2.mlp.down_proj"].weight, fq_q4_0(base["layers.2.mlp.down_proj.weight"]))
    out = model(x).sum(); out.backward()                        # straight-through: the trained weight gets a gradient
    g = dict(model.named_parameters())["layers.2.mlp.down_proj.parametrizations.weight.original"].grad
    assert g is not None and g.abs().sum() > 0
    info = detach()
    assert info == {"steps_plain": 0, "steps_as_file": 1}        # two forwards of the model = one step
    assert [n for n, _ in model.named_parameters()] == list(base)
    assert all(torch.equal(p, base[n]) for n, p in model.named_parameters())
    assert all(torch.equal(p, base[n]) for n, p in frozen.named_parameters())


def test_file_view_coin_is_per_step():
    frozen = Toy(); model = Toy(); model.load_state_dict(frozen.state_dict())
    detach = attach_file_view(model, frozen, {"layers.2.mlp.down_proj"}, 0.5, seed=0)
    x = torch.randn(1, 64)
    for _ in range(200):
        model(x); model(x)
    info = detach()
    assert info["steps_plain"] + info["steps_as_file"] == 200 and 70 <= info["steps_as_file"] <= 130
