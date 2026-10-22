"""Bias-free one-hidden-layer ReLU MLP (two-homogeneous in its parameters)."""
import torch
import torch.nn as nn


def make_mlp(d_in: int, width: int, d_out: int, seed: int = 0) -> nn.Sequential:
    """He-normal init, no biases, as in Appendix G.1/G.2."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(d_in, width, bias=False), nn.ReLU(), nn.Linear(width, d_out, bias=False)
    )
    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    return model


def init_weights(d_in: int, width: int, d_out: int, seed: int = 0):
    """Same initial weights as `make_mlp`, returned as (W1 [d_in, width], W2 [width, d_out])."""
    m = make_mlp(d_in, width, d_out, seed)
    return m[0].weight.detach().T.contiguous(), m[2].weight.detach().T.contiguous()
