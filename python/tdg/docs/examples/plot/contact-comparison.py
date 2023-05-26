#!/usr/bin/env python

import torch
import matplotlib.pyplot as plt
import tdg.others

fig, ax = plt.subplots(1,1, figsize=(8,6))

alpha = torch.linspace(-0.625, +0.625, 1000)
tdg.others.contact_comparison(ax, alpha=alpha)

ax.set_xlim([min(alpha), max(alpha)])
ax.legend()
fig.tight_layout()
