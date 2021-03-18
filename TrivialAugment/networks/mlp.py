import torch
from torch import nn


def MLP(D_out,in_dims,adaptive_dropouter_creator):
    print('adaptive dropouter', adaptive_dropouter_creator)
    in_dim = 1
    for d in in_dims: in_dim *= d
    ada_dropper = adaptive_dropouter_creator(100) if adaptive_dropouter_creator is not None else None
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, 300),
        nn.Tanh(),
        nn.Linear(300,100),
        ada_dropper or nn.Identity(),
        nn.Tanh(),
        nn.Linear(100,D_out)
    )
    model.adaptive_dropouters = [ada_dropper] if ada_dropper is not None else []
    return model
