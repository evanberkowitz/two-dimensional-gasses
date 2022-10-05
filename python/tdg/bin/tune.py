#!/usr/bin/env python3

import torch
torch.set_printoptions(precision=8)

import tdg
from tdg import EffectiveRangeExpansion
from tdg import ReducedTwoBodyA1Hamiltonian as A1
from tdg.Luescher import Zeta2D
Z = Zeta2D()

import argparse

parser = argparse.ArgumentParser(
    allow_abbrev=False,
    prog='tune.py',
    description="Tune a sum of LegoSpheres to a given effective range expansion.",
    epilog='''
    Try, for example
    tune.py --ere 1 --nx 9 --radius 0 0 --radius 0 1 --start -4.2 0.8 --plot example.pdf
'''
)
parser.add_argument('--ere', nargs='+', type=float, help="Scattering length, effective range^2, ... normalized by 2π/L appropriately.")
parser.add_argument('--nx',  type=int, help='sites on a side')
parser.add_argument('--radius', nargs=2, type=int, action='append', default=[], help='each pair gives the radius of one LegoSphere.')
parser.add_argument('--start', nargs='*', type=float, help='A starting guess for each LegoSphere coefficient.')
parser.add_argument('--plot', type=str, default="", help="'show' or a name of a file")
parser.add_argument('--verbose', action="store_true",  default=False)

args = parser.parse_args()
if args.verbose:
    print(args)

assert len(args.radius) == len(args.start), "Provide one starting guess for each LegoSphere radius."

# First we need to know what scattering amplitudes we're aiming for
ere = EffectiveRangeExpansion(torch.tensor(args.ere))
# and the two-body Hamiltonian we want to tune.
H = A1(tdg.Lattice(args.nx), [tdg.LegoSphere(r) for r in args.radius])

# Do "inverse Luescher" to find one energy level for each sphere we need to tune,
energies = ere.target_energies(args.nx, H.spheres)
# and tune H until we find coefficients that reproduce those energies.
coefficients = H.tuning(
    energies.clone().detach(),
    start=torch.tensor(args.start, dtype=torch.float64, requires_grad=True)
)

for r, c in zip(args.radius, coefficients.clone().detach().numpy()):
    print(r, c)

if not args.plot:
    exit()

# To plot we need to compute the energies with those coefficients
E = H.eigenenergies(coefficients)
# and convert into the dimensionless x
x = E* args.nx**2 / (2*torch.pi)**2

import numpy as np
import matplotlib.pyplot as plt

# First draw the zeta function.
fig, ax = plt.subplots(1,1, figsize=(12,8))
exact = torch.linspace(-5.001,25.001, 1000)
Z.plot(ax, exact)
# and the analytic piece of the ERE
ax.plot(exact, ere.analytic(exact), color='black', linestyle='dashed')

# Now evaluate on x
z = Z(x)/torch.pi**2

# and plot.
ax.plot( x.clone().detach().numpy(), z.clone().detach().numpy(), color='orange',linestyle='none', marker='o')

ax.set_xlim([-5,25])
ax.set_ylim([-1,1])

fig.tight_layout()

if args.plot == "show":
    plt.show()
else:
    plt.savefig(args.plot)
