import torch
import tdg
from tdg.observable import observable, derived

@observable
def ss(ensemble):
    r'''
    The (spatial) convolution of :math:`s^i * s^j`, which plays an important role in spin-spin fluctuation correlators.

    Configurations slowest, then (linearized) sites, then i, then j.

    .. todo::
       The convolution may be fourier accelerated.
    '''

    L = ensemble.Action.Spacetime.Lattice

    c = (
    # 1/4 δ_ab δ^ij < n_a >
    # nb: this takes the spatial average, by translation invariance
    + torch.einsum('c,ab,ij->cabij', ensemble.N('fermionic') , torch.eye(L.sites), torch.eye(3) / 4 / L.sites)

    + (
    # i/2 δ_ab ε^ijk < s^k_a >
    # nb: this takes the spatial average, by translation invariance
        + torch.einsum('ijk,ck,ab->cabij', 0.5j / L.sites * tdg.epsilon, ensemble.Spin, torch.eye(L.sites))
        if (ensemble.Action.h != torch.tensor([0.,0.,0.])).all()
        else 0
    )

    # 1/4 σ^i σ^j G G
    + torch.einsum('caamn,cbbop,imn,jop->cabij', ensemble.G, ensemble.G, tdg.PauliMatrix[1:], tdg.PauliMatrix[1:]/4)
    - torch.einsum('cabmp,cbaon,imn,jop->cabij', ensemble.G, ensemble.G, tdg.PauliMatrix[1:], tdg.PauliMatrix[1:]/4)
    )

    return torch.einsum('cabij,bra->crij',
                        c,
                        L.convolver + 0.j
    )

@derived
def spin_spin_fluctuations(ensemble):
    r'''
    A derived quantity, :math:`\left\langle (e^i*s^j)_r \right\rangle - \left(\left\langle s^i \right\rangle * \left\langle s^j \right\rangle\right)_r`.

    Bootstraps first, then relative coordinate :math:`r`.

    .. todo::
       The disconnected convolution may be fourier accelerated.
    '''

    L = ensemble.Action.Spacetime.Lattice

    return ensemble.ss - torch.einsum('cai,cbj,bra->crij', ensemble.spin, ensemble.spin, 0.j+L.convolver)

