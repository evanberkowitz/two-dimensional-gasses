#!/usr/bin/env python

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import tdg
from tdg.symmetry import D4

# The point of this test is to check the unitarity of a D4-projection

def norm_squared(v):
    return (v.conj() * v).sum(axis=-1)

def norms(symmetrizer, v):
    ns = dict()
    total = 0.
    for i in symmetrizer.irreps:
        n = norm_squared(symmetrizer(v, irrep=i))
        ns[i] = n
        total += n

    ns['total'] = total

    return ns

if __name__ == '__main__':
    digits = 12
    small = 1e-7

    parser = tdg.cli.ArgumentParser()
    parser.add_argument('--nx', type=int, default=5)
    parser.add_argument('--measurements', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=small)
    args = parser.parse_args()

    import logging
    logger = logging.getLogger(__name__)

    lattice = tdg.Lattice(args.nx)
    logger.debug(f'{lattice=}')
    symmetrize = D4(lattice)

    logger.debug(f'drawing {args.measurements} ')
    vectors = torch.rand((args.measurements, lattice.sites))

    v2 = norm_squared(vectors)

    u2 = torch.tensor([norms(symmetrize, v)['total'] for v in vectors])

    diff = v2 - u2
    logger.debug(f'Maxmial difference is {diff.abs().max()}')
    assert ((v2 - u2).abs() < args.threshold).all()


