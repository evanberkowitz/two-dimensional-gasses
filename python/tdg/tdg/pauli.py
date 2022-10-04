#!/usr/bin/env python3

import torch

PauliMatrix = torch.tensor([
    [[+1,0],[0,+1]],
    [[0,+1],[+1,0]],
    [[0,-1j],[+1j,0]],
    [[+1,0],[0,-1]]
    ])
