#!/usr/bin/env python

import torch
import tdg
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,3, figsize=(10,3))

L = tdg.Lattice(9)
random = torch.rand(L.sites)

# The random data:
L.plot_2d_scalar(ax[0], random)
# Completely symmetrized:
L.plot_2d_scalar(ax[1], tdg.symmetry.D4A1(L)(random)) 
# Symmetrized while holding (0,1) fixed, so that it's symmetric across the y axis.
L.plot_2d_scalar(ax[2], tdg.symmetry.D4A1(L, fixed=torch.tensor(((0,1),)))(random))

