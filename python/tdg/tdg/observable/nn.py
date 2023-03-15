import torch
from tdg.observable import observable

@observable
def nn(ensemble):
    r'''
    The convolution of n with n :math:`n*n`, which plays an important role in the density-density fluctuation correlator.

    Configurations slowest, then (linearized) sites.
    '''

    L = ensemble.Action.Spacetime.Lattice

    c = (
    + torch.einsum('caass,cbbtt->cab', ensemble.G, ensemble.G)
    - torch.einsum('cbats,cabst->cab', ensemble.G, ensemble.G)
    + torch.einsum('c,ab->cab', ensemble.N('fermionic') / L.sites, torch.eye(L.sites))
    )

    return torch.einsum('cab,bra->cr',
                        c,
                        L.convolver + 0.j
    )

