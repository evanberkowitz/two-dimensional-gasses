#!/usr/bin/env python

import matplotlib.pyplot as plt
import tdg

fig, ax = plt.subplots(1,2)

L = tdg.Lattice(5)
L.plot_2d_scalar(ax[0], L.coordinates[:,0])
L.plot_2d_scalar(ax[1], L.coordinates[:,1], center_origin=False)

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].set_xlabel('x')

fig.tight_layout()
plt.show()
