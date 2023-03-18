import torch
from tdg.observable import observable

@observable
def Kinetic(ensemble):
    r'''
    The total kinetic energy,

    .. math ::
       \left\langle \texttt{Kinetic} \right\rangle
       =
       \left\langle \sum_{ab\sigma} \tilde{\psi}^{\dagger}_{a\sigma} \tilde{\kappa}_{ab} \tilde{\psi}_{b\sigma} \right\rangle,

    one per configuration.
    '''

    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('cabss,ab->c', ensemble.G, L.kappa)

@observable
def Potential(ensemble):
    r'''
    The total potential energy,

    .. math ::
       \left\langle \texttt{Potential} \right\rangle
       =
       \left\langle \frac{1}{2} \sum_{ab} \tilde{n}_a \tilde{V}_{ab} \tilde{n}_b - \frac{N_x^2 C_0}{2} \sum_a n_a \right\rangle,

    one per configuration.
    '''

    L = ensemble.Action.Spacetime.Lattice
    V = ensemble.Action.Potential

    return (
        torch.einsum('r,cr->c', 0.5 * L.sites * V.spatial(L)[0] + 0.j, ensemble.nn)
      - 0.5 * L.sites * V.C0 * ensemble.N('fermionic')
    )

@observable
def FreeEnergy(ensemble):
    r'''
    The total free energy,

    .. math ::
       \left\langle \texttt{FreeEnergy} \right\rangle
       =
       \left\langle \tilde{K} + \tilde{V} - \tilde{\mu} \tilde{N} - \tilde{h}\cdot \tilde{S} \right\rangle

    one per configuration.
    '''
    return (
            ensemble.Kinetic

          + ensemble.Potential

          - ensemble.Action.mu * ensemble.N('fermionic')

          - ensemble.Action.h[0] * ensemble.Spin(1)
          - ensemble.Action.h[1] * ensemble.Spin(2)
          - ensemble.Action.h[2] * ensemble.Spin(3)
            )
