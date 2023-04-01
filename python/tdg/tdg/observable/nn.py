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

    # return torch.einsum('cab,bra->cr',
    #                     c,
    #                     L.convolver + 0.j
    # )
    return L.fft(torch.einsum('ckk->ck', L.fft(L.ifft(c, axis=2), axis=1)), axis=1) / L.sites

@derived
def density_density_fluctuations(ensemble):
    r'''
    A derived quantity, :math:`\left\langle n*n \right\rangle - \left\langle n \right\rangle * \left\langle n \right\rangle`.

    Bootstraps first, then relative coordinate.

    '''

    L = ensemble.Action.Spacetime.Lattice

    # These two lines differ only in speed:
    #
    # return ensemble.nn - torch.einsum('ca,cb,bra->cr', ensemble.n('fermionic'), ensemble.n('fermionic'), 0.j+L.convolver)
    #
    return ensemble.nn - L.fft(L.fft(ensemble.n('fermionic'), axis=1) * L.ifft(ensemble.n('fermionic'), axis=1), axis=1) / L.sites
