import torch
from tdg.observable import observable, derived

@observable
def kFa_squared(ensemble):
    r'''
    :math:`(k_F a)^2 = N \tilde{a}^2 / 2\pi`.
    '''
    ere = ensemble.Action.Tuning.ere

    return ensemble.N * ere.a**2 / (2*torch.pi)

@derived
def kFa(ensemble):
    r'''
    :math:`\sqrt{\texttt{kFa_squared}}`.
    '''
    return ensemble.kFa_squared.sqrt()

@derived
def momentum_by_kF_squared(ensemble):
    '''
    :math:`(k/k_F)^2`, which is particularly useful for plotting as a function of momentum.

    Bootstraps first, then linearized momentum index :math:`k`.
    '''

    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('k,b->bk', 2*torch.pi * L.linearize(L.ksq), 1 / ensemble.N)

@derived
def alpha(ensemble):
    r'''
    In the language of Ref. :cite:`Beane:2022wcn`, :math:`\alpha(k_F)` is a dimensionless coupling constant that is the natural expansion parameter of the two-dimensional EFT.
    '''
    return - 1 / torch.log(ensemble.kFa)

@derived
def binding_by_EF(ensemble):
    r'''
    The binding energy :math:`-(Ma^2)^{-1}` divided by the Fermi energy :math:`k_F^2/2M`,

    .. math::
       \frac{\mathcal{E}_B}{E_F} = - \frac{2}{(k_F a)^2}
    '''
    return - 2 / ensemble.kFa_squared

@derived
def T_by_TF(ensemble):
    r'''
    The temperature in proportion to the Fermi temperature,
    
    .. math::
       \frac{T}{T_F} = \frac{1}{\pi \tilde{\beta} N}
    '''

    beta = ensemble.Action.beta

    return 1./(beta * torch.pi * ensemble.N)
