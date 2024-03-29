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
       \left\langle \frac{1}{2} \sum_{ab} \tilde{n}_a \tilde{V}_{ab} \tilde{n}_b - \frac{N_x^2 C_0}{2} \sum_a \tilde{n}_a \right\rangle,

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
    The baryon mass times potential energy density normalized by :ref:`the Fermi momentum <fermi>`.
    
    .. math::
       \frac{v}{k_F^4} = \frac{MV}{k_F^4 L^2} = \frac{VML^2}{(k_F L)^4} = \frac{\texttt{Potential}}{(2\pi \texttt{N})^2}
    '''
    return ensemble.Potential / (2*torch.pi * ensemble.N)**2

@observable
def InternalEnergy(ensemble):
    r'''
    The total internal energy,

    .. math ::
       \left\langle \texttt{InternalEnergy} \right\rangle
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
def internalEnergy_by_kF4(ensemble):
    r'''
    The baryon mass times the internal energy density normalized by :ref:`the Fermi momentum <fermi>`.
    
    .. math::
       \frac{u}{k_F^4} = \frac{MU}{k_F^4 L^2} = \frac{UML^2}{(k_F L)^4} = \frac{\texttt{InternalEnergy}}{(2\pi \texttt{N})^2}
    '''
    return ensemble.InternalEnergy / (2*torch.pi * ensemble.N)**2
