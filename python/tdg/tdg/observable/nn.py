import torch
from tdg.observable import observable, derived

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

@derived
def density_density_fluctuations(ensemble):
    r'''
    A derived quantity, :math:`\left\langle n*n \right\rangle - \left\langle n \right\rangle * \left\langle n \right\rangle`.

    Bootstraps first, then relative coordinate.

    .. todo::
       The disconnected convolution may be fourier accelerated.
    '''

    L = ensemble.Action.Spacetime.Lattice

    return ensemble.nn - torch.einsum('ca,cb,bra->cr', ensemble.n('fermionic'), ensemble.n('fermionic'), 0.j+L.convolver)
