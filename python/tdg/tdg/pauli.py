#!/usr/bin/env python3

from functools import cached_property
import numpy as np

PauliMatrix = np.array([
    [[+1,0],[0,+1]],
    [[0,+1],[+1,0]],
    [[0,-1j],[+1j,0]],
    [[+1,0],[0,-1]]
    ])
