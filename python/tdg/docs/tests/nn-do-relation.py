#!/usr/bin/env python

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import tdg

# The point of this test is to check an operator identity is correctly reflected in observables.
# The identity relates the nn two-point correlator at zero distance, the double occupancy, and the number operator.
#
#   nn(0) = ( 2 DoubleOccupancy + N ) / Volume

if __name__ == '__main__':

    digits = 12
    small = 1e-7
    parser = tdg.cli.ArgumentParser()
    parser.add_argument('--nx', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--nt', type=int, default=8)
    parser.add_argument('--C0', type=float, default=-0.123)
    parser.add_argument('--measurements', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=small)
    args = parser.parse_args()

    import logging
    logger = logging.getLogger(__name__)

    lattice = tdg.Lattice(args.nx)
    logger.debug(f'{lattice=}')
    spacetime = tdg.Spacetime(args.nt, lattice)
    logger.debug(f'{spacetime=}')

    potential = tdg.Potential(tdg.LegoSphere([0,0], args.C0))
    logger.debug(f'{potential=}')
    S = tdg.Action(spacetime, potential, args.beta)
    logger.debug(f'action={S}')

    logger.debug(f'drawing {args.measurements} quenched samples to check an observable identity')
    ensemble = tdg.ensemble.GrandCanonical(S).from_configurations(S.quenched_sample(torch.Size([args.measurements])))

    nn = ensemble.nn[:,0]
    DO = ensemble.DoubleOccupancy
    N  = ensemble.N

    logger.debug(f'<nn(0)> = {nn.mean():.{digits}f}')
    logger.debug(f'<DO>    = {DO.mean():.{digits}f}')
    logger.debug(f'<N>     = {N.mean():.{digits}f}')
    logger.debug(f'<2DO+N> / V = {(2*DO.mean() + N.mean())/lattice.sites:.{digits}f}')

    difference = torch.abs(nn - (2*DO+N)/lattice.sites)
    logger.debug(f'<nn(0) - (2DO+N)/V > = {difference.mean():.{digits}f}')

    logger.debug(f'threshold = {args.threshold}')

    assert (difference < args.threshold).all()
