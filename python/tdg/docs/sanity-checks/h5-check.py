#!/usr/bin/env python

# We will do a small example and save and read things along the way.

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import h5py as h5

import tdg, tdg.HMC

from tqdm import tqdm

parser = tdg.cli.ArgumentParser()
parser.add_argument('--storage', type=str, default='h5-check.h5', help="An h5 file to use for reading and writing.  Defaults to h5-check.h5")
args = parser.parse_args()

nx = 3
lattice = tdg.Lattice(nx)

radii = [[0,0]]
ere = tdg.EffectiveRangeExpansion(torch.tensor([1.0]))
precomputed_C = torch.tensor([-0.027839275445905677], requires_grad=True)
tuning = tdg.Tuning(ere, lattice, radii, C = precomputed_C)

nt = 8
beta = torch.tensor(0.125)
mu = torch.tensor(-0.0)
S  = tuning.Action(nt, beta, mu)
ensemble = tdg.ensemble.GrandCanonical(S)

H = tdg.HMC.Hamiltonian(S)
integrator = tdg.HMC.Omelyan(H, 20)
ensemble.generate(100, tdg.HMC.MarkovChain(H, integrator), progress=tqdm)

pre = ensemble.N.mean()

with h5.File(args.storage, 'w') as f:
    ensemble.to_h5(f.create_group('Ensemble'))
with h5.File(args.storage, 'r') as f:
    ENSEMBLE = tdg.ensemble.GrandCanonical.from_h5(f['Ensemble'])

post = ENSEMBLE.N.mean(axis=0)
assert pre == post
