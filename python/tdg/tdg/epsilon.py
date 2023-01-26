#!/usr/bin/env python3

import torch

epsilon = torch.tensor([
    [[0,0,0],[0,0,+1],[0,-1,0]],
    [[0,0,-1],[0,0,0],[+1,0,0]],
    [[0,+1,0],[-1,0,0],[0,0,0]]
    ])
r'''
The totally-antisymmetric Levi-Civita symbol :math:`\epsilon` with three indices.

:math:`\epsilon[0,1,2]=+1`.  Even permutations of the indices are also +1; odd permutations give -1.
All other entries vanish.
'''
