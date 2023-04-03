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
def alpha(ensemble):
    r'''
    In the language of Ref. :cite:`Beane:2022wcn`, :math:`\alpha(k_F)` is a dimensionless coupling constant that is the natural expansion parameter of the two-dimensional EFT.
    '''
    return - 1 / torch.log(ensemble.kFa)
