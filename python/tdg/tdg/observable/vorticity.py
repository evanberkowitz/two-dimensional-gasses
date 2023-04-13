import torch

import tdg
from tdg.observable import observable

import logging
logger = logging.getLogger(__name__)

@observable
def vorticity(ensemble):
    r'''
    The lattice-exact curl of the baryon number current, :math:`\tilde{\omega}_x^i`.

    Zero in expectation value by periodic boundary conditions.

    Configurations slowest, then space `x`, then direction `i` (in 2 dimensions, a singleton dimension).
    '''
    
    L = ensemble.Action.Spacetime.Lattice
    momenta = 2 * torch.pi / L.nx * L.coordinates
    q_cross_k = 0.j + L.cross(momenta, momenta)

    TiG = 1j*torch.einsum('qki,ckqss->ckqi', q_cross_k, ensemble.G_momentum)
    return L.sites * torch.einsum('cxxi->cxi', L.ifft(L.fft(TiG, axis=1), axis=2))

# A generic two-vorticity correlator is 
# The ω†x ωy operator can be written
#
#    volume^{-2} sum_{pj qk} e^{2πi[ (q-k)y - (p-j) x ]/Nx}
#                           (2π/Nx)^4 (2π/Nx)^2 (p × j)^i (2π/Nx)^2 (q × k)^i
#                           ψ†_{pτ} ψ_{jτ} ψ†_{kσ} ψ_{qσ}
#
# One way to compute this value is to see it as two independent double Fourier transforms of the last two lines element-wise multiplied,
#
#   TGG_{pj qk} =           (2π/Nx)^4 (2π/Nx)^2 (p × j)^i (2π/Nx)^2 (q × k)^i
#                           volume^{-2} ψ†_{pτ} ψ_{jτ} ψ†_{kσ} ψ_{qσ}
#
# This is the strategy we will take here.

# First, the cross product tensor
def _double_cross(ensemble):
    # _double_cross_{qkpj} = (2π/Nx)^4 (2π/Nx)^2 (p × j)^i (2π/Nx)^2 (q × k)^i
    L = ensemble.Action.Spacetime.Lattice
    momenta = 2*torch.pi / L.nx * L.coordinates
    cross = L.cross(momenta, momenta) # the momenta contain the 2π/Nx factors
    return torch.einsum('qki,pji->qkpj', cross, cross)
    # Equal to
    #
    # (
    #     torch.einsum('pa,jb,qa,kb->qkpj', momenta, momenta, momenta, momenta)
    #    -torch.einsum('pa,jb,qb,ka->qkpj', momenta, momenta, momenta, momenta)
    # )
    #

# Next the two-propagator contractions.
def _GG(ensemble):
    # _GG = volume^{-2} ψ†_{pτ} ψ_{jτ} ψ†_{kσ} ψ_{qσ}

    # The two-propagator contractions absorb two factors of inverse volume while the one-propagator
    # contraction that comes from normal-ordering the operators ALSO absorbs two, because the anticommutator of momentum-space
    # {ψ_{pσ}, ψ†_{kτ}} = V δ_{pk,στ}.

    L = ensemble.Action.Spacetime.Lattice
    return (
            torch.einsum('ckqss,cpjtt->cpjkq', ensemble.G_momentum, ensemble.G_momentum)    # from the unmixed contraction
          - torch.einsum('ckjst,cpqts->cpjkq', ensemble.G_momentum, ensemble.G_momentum)    # from the mixed contraction
          + torch.einsum('cpqss,kj->cpjkq',    ensemble.G_momentum, torch.eye(L.sites))     # from the anticommutator
            )

# Now we can do the element-wise multiplication
def _TGG(ensemble):
    return torch.einsum('pjqk,cpjkq->cpjkq', _double_cross(ensemble), _GG(ensemble))

# Finally, to get a function of just x and y we need two double Fourier transforms.
# Using our intermediate quantities the ω†x ωy operator is
#    sum_{pj qk} e^{2πi[ (q-k)x - (p-j) y ]/Nx} TGG_{kqpj}
# 
@observable
def _vorticity_vorticity(ensemble):
    # Each double FFT multiplies by a factor of 1/V.
    # But we don't want them; we've got to multiply them back in.

    L = ensemble.Action.Spacetime.Lattice

    # In the FT we have e^{2πi[ (q-k)y - (p-j) x ]/Nx}
    return torch.einsum('cxxyy->cxy',
                            L.ifft(    # q
                                L.fft( # k
                                L.ifft(# j
                                L.fft( # p
                                    _TGG(ensemble),
                                axis=1),# p
                                axis=2),# j
                                axis=3),# k
                            axis=4),    # q
                        ) * L.sites**2

@observable
def vorticity_squared(ensemble):
    r'''
    The square of the dimensionless vorticity :math:`\tilde{\omega}_x^2`.

    Configurations slowest, then the spatial coordinate :math:`x`.
    '''
    return torch.einsum('cxx->cx', ensemble._vorticity_vorticity)

@observable
def vorticity_vorticity(ensemble):
    r'''
    The (dimensionless) spatial convolution of the vorticity, :math:`\tilde{\Omega}_r = (\tilde{\omega}^i * \tilde{\omega}^i)_r` summed over :math:`i`.

    Configurations slowest, then the relative coordinate :math:`r`.
    '''

    # One expression we could use is
    #
    #   volume^{-1} sum_{xy} δ(y, x-r) ω†x ωy

    L = ensemble.Action.Spacetime.Lattice
    xy = ensemble._vorticity_vorticity

    # The convolver comes with one inverse power of volume anyway, so
    # return torch.einsum('cxy,yrx->cr', xy, 0.j+L.convolver)
    #
    # OR
    #
    return L.fft(torch.einsum('ckk->ck', L.fft(L.ifft(xy, axis=1), axis=2)), axis=1) / L.sites
    # Let's check the volume factor:
    #
    # the above code                            mathematical expression
    # L.fft(                                    sum_{k} e^{-2πikr/Nx} 
    #   torch.einsum('ckk->ck',                 sum_{q} δ_{kq}
    #       L.fft(                              1/V sum_{y} e^{-2πiqy/Nx}
    #           L.ifft(xy, axis=1),                 sum_{x} e^{+2πikx/Nx} T_{xy}
    #           axis=2)
    #       ),
    # axis=1) * L.sites                         1/V
    #
    #   Gathering the terms and simplifying,
    #
    #   1/V sum_{k} e^{-2πikr/Nx} sum_{q} δ_{kq} 1/V sum_{xy} e^{2πi(kx-qy)/Nx} T_{xy}
    # = 1/V sum_{xy} T_{xy} 1/V sum_{kq} δ_{kq} e^{2πi [ k(x-r) - qy ]}
    # = 1/V sum_{xy} T_{xy} 1/V sum_{k}         e^{2πik[  (x-r) -  y ]}
    # = 1/V sum_{xy} T_{xy} 1/V V δ_{y,x-r}
    # = 1/V sum_{xy} T_{xy} δ_{y,x-r}
    # = 1/V sum_{xy} T_{x,x-r}
