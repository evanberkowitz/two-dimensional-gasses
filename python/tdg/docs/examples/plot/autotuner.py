#!/usr/bin/env python

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(9999) # for reproducibility

import matplotlib.pyplot as plt
import tdg
import tdg.HMC as HMC

S = tdg.action._demo(
        nx=9, nt=16,
        beta=0.061224489795918366, mu=torch.tensor(0.), h=torch.tensor([0., 0., 0.]),
        C0=-1.2388577106478569)
H = HMC.Hamiltonian(S)

A = HMC.Autotuner(H, HMC.Omelyan)
integrator, start = A.target(0.75, start='hot', starting_md_steps=15)

fig, ax = plt.subplots(2,1, figsize=(8,6))
A.plot_history(ax[0])
A.plot_models(ax[1])
ax[1].legend()
fig.tight_layout()
plt.show()

# Then use the integrator and start to begin HMC...
