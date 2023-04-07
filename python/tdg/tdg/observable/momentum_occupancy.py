import torch
from tdg.observable import observable
from tdg import PauliMatrix

@observable
def n_momentum(ensemble):
    r'''
    The expectation value of :math:`\frac{1}{V}\tilde{\psi}^\dagger_{k} \tilde{\psi}_{k}` summed over spins.

    Configurations first, then momentum :math:`k`.
    '''
    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('ckkss->ck', ensemble.G_momentum)

@observable
def spin_momentum(ensemble):
    r'''
    The expectation value of :math:`\frac{1}{2V} \tilde{\psi}^\dagger_{k} \sigma^i \tilde{\psi}_{k}`

    Configurations first, then momentum :math:`k`, then spin index :math:`i`.
    The spin direction matches the index of :code:`tdg.PauliMatrix[1:]` so that :code:`0` is in the x direction, for example.
    '''
    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('ckkst,ist->cki', ensemble.G_momentum, PauliMatrix[1:] / 2)
