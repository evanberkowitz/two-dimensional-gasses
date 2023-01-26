#!/usr/bin/env python

# To perform an end-to-end check of the HMC and observable pipeline, we can
#
# 1. Do a grand-canonical HMC
# 2. Canonically project to the two-particle S=0 sector
# 3. Measure some observables.
#
# and compare the result to an exact calculation in the same two-particle S=0 sector,
# using a numerical Hilbert-space Trotterization for the same action.

import itertools

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import h5py as h5

import tdg, tdg.HMC as HMC

from tqdm import tqdm

import logging
logging_levels = {
    'CRITICAL': logging.CRITICAL,
    'ERROR':    logging.ERROR,
    'WARNING':  logging.WARNING,
    'INFO':     logging.INFO,
    'DEBUG':    logging.DEBUG,
    'NOTSET':   logging.NOTSET,
}

def action():
    # Just an example that we settled on.

    nx = 3
    nt = 8
    beta = torch.tensor(0.125)
    # radii = [[0,0],[1,1]]
    radii = [[0,0]]

    lattice = tdg.Lattice(nx)

    ere = tdg.EffectiveRangeExpansion(torch.tensor([1.0]))
    tuning = tdg.Tuning(ere, lattice, radii, C = torch.tensor([-0.027839275445905677], requires_grad=True))

    mu   = torch.tensor(-1.0)
    return tuning.Action(nt, beta, mu)

def ensemble(S, cfgs, file, clobber=False):
    # Read from disk if available.
    # Do HMC if not available or if there are not enough configurations.

    H = HMC.Hamiltonian(S)
    integrator = HMC.Omelyan(H, 50, 1)
    hmc = HMC.MarkovChain(H, integrator)

    ens= tdg.ensemble.GrandCanonical(S)

    if not clobber:
        try:
            with h5.File(file, 'r') as f:
                ens = ens.from_configurations(torch.from_numpy(f['/ensemble'][()]))
            if len(ens.configurations) >= cfgs:
                return ens.from_configurations(ens.configurations[-cfgs:])
        except:
            pass

    ens.generate(cfgs, hmc, progress=tqdm)
    with h5.File(file, 'w') as f:
        f['/ensemble'] = ens.configurations.clone().detach().numpy()

    return ens

def two_body_trotter_product(S):
    # Returns [ exp(-∆t K) exp(-∆t V) ]^nt in the two-body S=0 sector.
    # different from the continuum exp(-β H) by O(∆t^2), but equal
    # (up to an overall proportionality constant) to < PP_{2,0} >
    # using our numerical partition function.
    #
    # That proportionality constant cancels in expectation values.

    L = S.Spacetime.Lattice

    K = torch.kron(L.kappa, torch.eye(L.sites)) + torch.kron(torch.eye(L.sites), L.kappa)
    V = torch.diag(S.Potential.spatial(L).ravel()).to(torch.complex128)

    dt = S.dt

    trotter = torch.matmul(torch.matrix_exp(-dt * K), torch.matrix_exp(-dt * V))
    return torch.matrix_power(trotter, S.Spacetime.nt)

def two_body_double_occupancy(L):
    # A diagonal matrix such that
    #   1 if the two particles are on the same site,
    #   0 otherwise
    return torch.diag(torch.tensor(
        tuple((1.0 if a == b else 0.0 for a,b in itertools.product(range(L.sites), range(L.sites)))),
        dtype=torch.complex128
    ))

def two_body_expectation(trotterization, observable):
    return torch.matmul(trotterization, observable).trace() / trotterization.trace()

def naive_error(data, digits=5):
    # Just a quick-and-dirty estimate with no autocorrelation estimates.
    mean = data.real.mean()
    err  = data.real.std() / np.sqrt(len(data))
    
    return f'{mean:+.{digits}f}±{err:.{digits}f}'

if __name__ == '__main__':

    # A simple command-line interface.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='WARNING', help="A logging level, one of CRITICAL, ERROR, WARNING, INFO, DEBUG; defaults to WARNING.")
    parser.add_argument('--cfgs', type=int, default=1000)
    parser.add_argument('--storage', type=str, default='canonical-double-occupancy.h5')
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)10s %(message)s', level=logging_levels[args.log])

    # Use the same action for both the exact and Monte-Carlo calculation.
    S = action()

    # Exact:
    trotterization = two_body_trotter_product(S)
    print(f'Exact trotterized two-body        double occupancy  {two_body_expectation(trotterization, two_body_double_occupancy(S.Spacetime.Lattice)):+.8f}')

    # Canonically-projected grand-canonical estimate:
    e = ensemble(S, args.cfgs, args.storage, clobber=args.clobber)
    c = tdg.ensemble.Canonical(e)
    s = c.Sector(Particles=2, Spin=0)
    print(f'Monte carlo canonically projected double occupancy  {naive_error(s.DoubleOccupancy/s.weight.mean())}')
