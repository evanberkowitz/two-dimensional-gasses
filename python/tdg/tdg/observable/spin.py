import torch
import tdg
from tdg.observable import observable

####
#### Intensive
####

@observable
def spin(ensemble):
    r'''
    The local spin density.
    Configurations, then sites, then spin direction.  That makes it easy to do something with :code:`ensemble.s[...,1]`.
    The spin direction matches the index of :code:`tdg.PauliMatrix[1:]` so that :code:`0` is in the x direction, for example.
    '''

    return torch.einsum('cxxst,ist->cxi', ensemble.G, 0.5 * tdg.PauliMatrix[1:] + 0.j)

####
#### Extensive
####

@observable
def Spin(ensemble):
    r'''
    The total spin, summed over sites.  Configurations slowest, then spin direction.
    The spin direction matches the index of :code:`tdg.PauliMatrix[1:]` so that :code:`0` is in the x direction, for example.
    '''
    return ensemble.spin.sum(axis=1)

