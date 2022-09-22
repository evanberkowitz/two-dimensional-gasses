#!/usr/bin/env python3

from functools import cached_property
import numpy as np

PauliMatrix0 = np.array([[+1,0],[0,+1]])
PauliMatrix1 = np.array([[0,+1],[+1,0]])
PauliMatrix2 = np.array([[0,+1],[+1,0]]) * 1j
PauliMatrix3 = np.array([[+1,0],[0,-1]])

PauliIdentity = PauliMatrix0
PauliMatrices = np.array([PauliMatrix0, PauliMatrix1, PauliMatrix2])
PauliMatrix   = PauliMatrices
