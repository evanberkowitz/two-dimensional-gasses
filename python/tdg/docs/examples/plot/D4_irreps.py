#!/usr/bin/env python

import torch
import tdg, tdg.symmetry
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4,4, figsize=(10,12))

L = tdg.Lattice(9)
random = torch.rand(L.sites)

symmetry = tdg.symmetry.D4(L)

L.plot_2d_scalar(ax[0,0], random)
[a.remove() for a in ax[0,1:]]

for a, irrep in zip(ax[1], ('A1', 'A2', 'B1', 'B2')):
    L.plot_2d_scalar(a, symmetry(random, irrep).real)
    a.set_title(irrep)

for real, imag, irrep in zip(ax[2], ax[3], (("E", +1), ("E", -1), ("E'", +1), ("E'", -1))):
    L.plot_2d_scalar(real,  symmetry(random, irrep).real)
    L.plot_2d_scalar(imag,  symmetry(random, irrep).imag)
    real.set_title(irrep)

ax[2,0].set_ylabel('real')
ax[3,0].set_ylabel('imag')

fig.tight_layout()
plt.show()
