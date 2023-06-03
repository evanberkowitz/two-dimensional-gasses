import torch
from tdg.observable import observable
from tdg import PauliMatrix

@observable
def n_momentum(ensemble):
    r'''
    The expectation value of :math:`\frac{1}{V}\tilde{\psi}^\dagger_{k} \tilde{\psi}_{k}` summed over spins.

    This is a lattice discretization of :math:`\frac{1}{L^2} \psi^\dagger(k) \psi(k)` and it has good continuum and infinite-volume limits and is dimensionless.

    Configurations first, then momentum :math:`k`.
    '''
    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('ckkss->ck', ensemble.G_momentum)

@observable
def spin_momentum(ensemble):
    r'''
    The expectation value of :math:`\frac{1}{2V} \tilde{\psi}^\dagger_{k} \sigma^i \tilde{\psi}_{k}`

    This is a lattice discretization of :math:`\frac{1}{2L^2} \psi^\dagger(k) \sigma^i \psi(k)` and it has good continuum and infinite-volume limits and is dimensionless.  It should be 0 unless the external field :math:`h` is nonzero.

    Configurations first, then momentum :math:`k`, then spin index :math:`i`.
    The spin direction matches the index of :code:`tdg.PauliMatrix[1:]` so that :code:`0` is in the x direction, for example.
    '''
    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('ckkst,ist->cki', ensemble.G_momentum, PauliMatrix[1:] / 2)
