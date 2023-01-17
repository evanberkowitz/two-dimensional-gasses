#!/usr/bin/env python

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import tdg

def beta0doubleOccupancy(lattice, nt, C0, beta):
    # At the beta=0 grand-canonical expectation values
    # can be measured on the 0 configuration.
    #
    # That makes an exact calculation of the double occupancy easy;
    # it should come out to be 1/4 on each site, which can be shown
    # directly from the path integral, but also from analyzing each
    # site individually:  the four states are |none>, |up>, |down>,
    # |up and down> and only the last has double occupancy, so the
    # expectation value is 1/4.

    spacetime = tdg.Spacetime(nt, lattice)
    potential = tdg.Potential(tdg.LegoSphere([0,0], C0))
    S = tdg.Action(spacetime, potential, beta)

    all_zero = spacetime.vector(1)
    ensemble = tdg.ensemble.GrandCanonical(S).from_configurations(all_zero)

    return ensemble.doubleOccupancy[0], ensemble.DoubleOccupancy[0]

if __name__ == '__main__':
    import argparse

    small = 1e-7
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=3)
    parser.add_argument('--beta', type=float, default=small)
    parser.add_argument('--nt', type=int, default=8)
    parser.add_argument('--C0', type=float, default=-0.123)
    parser.add_argument('--threshold', type=float, default=small)
    args = parser.parse_args()

    do, DO = beta0doubleOccupancy(tdg.Lattice(args.nx), args.nt, args.C0, args.beta)

    assert (torch.abs((DO / args.nx**2) - 0.25)) < args.threshold
