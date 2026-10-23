"""Arms, nonlinearities and topologies (PROTOKOL_v5 §3)."""
import math
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_EPS = 1e-8

# Main study (§3.3): 18 FF arms.
P_ARMS = {"P-R": "R", "P-L": "L", "P-G": "G", "P-T": "T", "P-BANK": "B"}
K_ARMS = {"K-R": "RRR", "K-L": "LLL", "K-G": "GGG", "K-T": "TTT", "K-LLT": "LLT", "K-GGT": "GGT"}
K_ARMS.update({"K-" + "".join(p): "".join(p) for p in permutations("LGT")})
K_ARMS["K-BANK"] = "BBB"
FF_ARMS = list(P_ARMS) + list(K_ARMS)
BP_ARMS = {"BP-P-R": "P-R", "BP-K-LGT": "K-LGT"}

H1_GROUP = ["K-R", "K-L", "K-G", "K-T", "K-LLT", "K-GGT"]
H2_GROUP = ["K-LTG", "K-GLT", "K-GTL", "K-TLG", "K-TGL"]
H3_GROUP = ["P-R", "P-L", "P-G", "P-T"]
HYPOTHESES = {"H1": ("K-LGT", H1_GROUP), "H2": ("K-LGT", H2_GROUP), "H3": ("P-BANK", H3_GROUP)}

MINIPILOT_ARMS = ["P-R", "P-T", "P-BANK", "K-LGT", "K-G", "K-T", "K-GGT", "K-GLT"]
E9_ARMS = ["P-R", "P-BANK", "K-R", "K-LGT", "K-LLT", "K-GGT"]
E9_WIDTHS = [32, 64, 128]
E10_ARMS = ["P-R", "P-T", "P-BANK", "K-R", "K-LGT", "K-GGT"]
E10_WIDTHS = [8, 16, 32, 64]

# Block inner widths matched per FF layer (§3.1, §3.4): w -> (block-1 inner, block-2 inner).
K_INNER = {250: (164, 103), 128: (99, 53), 64: (56, 26), 32: (30, 13)}  # for 794-dim FF input (D4)
# Spiral (§3.4 E10): uniform K width u matched on total parameters.
SPIRAL_U = {8: 4, 16: 7, 32: 15, 64: 29}


def bank_sizes(n: int):
    return (math.ceil(n / 3), math.ceil((n - 1) / 3), n // 3)


def apply_act(kind: str, z: torch.Tensor) -> torch.Tensor:
    if kind == "R":
        return F.relu(z)
    if kind == "L":
        return F.leaky_relu(z, negative_slope=0.5)
    if kind == "G":
        return F.gelu(z, approximate="none")
    if kind == "T":
        return torch.tanh(z)
    if kind == "B":
        n_l, n_g, _ = bank_sizes(z.shape[-1])
        return torch.cat(
            [
                F.leaky_relu(z[..., :n_l], negative_slope=0.5),
                F.gelu(z[..., n_l : n_l + n_g], approximate="none"),
                torch.tanh(z[..., n_l + n_g :]),
            ],
            dim=-1,
        )
    raise ValueError(kind)


class FFLayer(nn.Module):
    """One FF layer: a chain of Linear + nonlinearity sublayers (1 for P, 3 for K)."""

    def __init__(self, dims, acts):
        super().__init__()
        assert len(dims) - 1 == len(acts)
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.acts = list(acts)
        self.out_dim = dims[-1]

    def forward(self, x):
        for lin, act in zip(self.linears, self.acts):
            x = apply_act(act, lin(x))
        return x


def goodness(a: torch.Tensor) -> torch.Tensor:
    return (a * a).sum(dim=-1)


def normalize(a: torch.Tensor) -> torch.Tensor:
    return a / (a.norm(dim=-1, keepdim=True) + NORM_EPS)


def layer_dims(arm: str, w: int = 250, in_dim: int = 794, spiral: bool = False):
    """Return ([dims layer 1], [dims layer 2], (acts1, acts2))."""
    if arm in P_ARMS:
        act = P_ARMS[arm]
        return [in_dim, w], [w, w], ([act], [act])
    acts = list(K_ARMS[arm])
    if spiral:
        u = SPIRAL_U[w]
        return [in_dim, u, u, u], [u, u, u, u], (acts, acts)
    a, b = K_INNER[w]
    return [in_dim, a, a, w], [w, b, b, w], (acts, acts)


def build_ff(arm: str, w: int = 250, in_dim: int = 794, spiral: bool = False):
    d1, d2, (a1, a2) = layer_dims(arm, w, in_dim, spiral)
    return FFLayer(d1, a1), FFLayer(d2, a2)


class BPNet(nn.Module):
    """BP arm: same topology as the FF arm + linear readout (§3.3, E2)."""

    def __init__(self, ff_arm: str, n_classes: int = 10, w: int = 250, in_dim: int = 784, spiral: bool = False):
        super().__init__()
        self.layer1, self.layer2 = build_ff(ff_arm, w, in_dim, spiral)
        self.readout = nn.Linear(self.layer2.out_dim, n_classes)

    def forward(self, x):
        return self.readout(self.layer2(self.layer1(x)))


def n_params(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())
