import torch
from tdg.observable import observable, derived

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

@derived
def kinetic_by_kF4(ensemble):
    r'''
The baryon mass times the kinetic energy density normalized by :ref:`the Fermi momentum <fermi>`.
    
    .. math::
       \frac{k}{k_F^4} = \frac{MK}{k_F^4 L^2} = \frac{KML^2}{(k_F L)^4} = \frac{\texttt{Kinetic}}{(2\pi \texttt{N})^2}
    '''
    return ensemble.Kinetic / (2*torch.pi * ensemble.N)**2

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
      - 0.5 * L.sites * V.C0 * ensemble.N
    )

@derived
def potential_by_kF4(ensemble):
    r'''
    The baryon mass times potential energy density normalized by :`the Fermi momentum <fermi>`.
    
    .. math::
       \frac{v}{k_F^4} = \frac{MV}{k_F^4 L^2} = \frac{VML^2}{(k_F L)^4} = \frac{\texttt{Potential}}{(2\pi \texttt{N})^2}
    '''
    return ensemble.Potential / (2*torch.pi * ensemble.N)**2

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

          - ensemble.Action.mu * ensemble.N

          - torch.einsum('i,ci->c', 0.j+ensemble.Action.h, ensemble.Spin)
            )

@derived
def freeEnergy_by_kF4(ensemble):
    r'''
    The baryon mass times the free energy density normalized by :`the Fermi momentum <fermi>`.
    
    .. math::
       \frac{f}{k_F^4} = \frac{MF}{k_F^4 L^2} = \frac{FML^2}{(k_F L)^4} = \frac{\texttt{FreeEnergy}}{(2\pi \texttt{N})^2}
    '''
    return ensemble.FreeEnergy / (2*torch.pi * ensemble.N)**2
