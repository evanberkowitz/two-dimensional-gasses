import torch
import tdg
from tdg.observable import observable

@observable
def ss(ensemble):
    r'''
    The (spatial) convolution of :math:`s^i * s^j`, which plays an important role in spin-spin fluctuation correlators.

    Configurations slowest, then (linearized) sites, then i, then j.
    '''

    L = ensemble.Action.Spacetime.Lattice

    c = (
    # 1/4 δ_ab δ^ij < n_a >
    + torch.einsum('c,ab,ij->cabij', ensemble.N('fermionic') , torch.eye(L.sites), torch.eye(3) / 4 / L.sites)

    + (
    # i/2 δ_ab ε^ijk < s^k_a >
        + torch.einsum('ij,c,ab->cabij', 0.5j / L.sites * tdg.epsilon[:,:,0], ensemble.S(1), torch.eye(L.sites))
        + torch.einsum('ij,c,ab->cabij', 0.5j / L.sites * tdg.epsilon[:,:,1], ensemble.S(2), torch.eye(L.sites))
        + torch.einsum('ij,c,ab->cabij', 0.5j / L.sites * tdg.epsilon[:,:,2], ensemble.S(3), torch.eye(L.sites))
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

