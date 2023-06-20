#!/usr/bin/env python

import torch
import matplotlib.pyplot as plt
import tdg.references

fig, ax = plt.subplots(1,1, figsize=(8,6))

alpha = torch.linspace(-1.0, +1.0, 1000)
tdg.references.energy_comparison(ax, alpha=alpha)

ax.set_xlim([min(alpha), max(alpha)])
ax.set_ylim([-0.08, 0.3])
ax.legend()
fig.tight_layout()
plt.show()
