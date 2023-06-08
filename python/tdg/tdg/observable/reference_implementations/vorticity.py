import torch

import tdg
from tdg.observable import observable

import logging
logger = logging.getLogger(__name__)

logger.info(f'Importing reference implementations in {__name__}.')

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
            torch.einsum('ckqss,cpjtt->cpjqk', ensemble.G_momentum, ensemble.G_momentum)    # from the unmixed contraction
          - torch.einsum('ckjst,cpqts->cpjqk', ensemble.G_momentum, ensemble.G_momentum)    # from the mixed contraction
          + torch.einsum('cpqss,kj->cpjqk',    ensemble.G_momentum, torch.eye(L.sites))     # from the anticommutator
            )

# Now we can do the element-wise multiplication
def _TGG(ensemble):
    return torch.einsum('pjqk,cpjqk->cpjqk', _double_cross(ensemble), _GG(ensemble))

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
                            L.fft(      # k
                                L.ifft( # q
                                L.ifft( # j
                                L.fft(  # p
                                    _TGG(ensemble),
                                axis=1),# p
                                axis=2),# j
                                axis=3),# q
                            axis=4),    # k
                        ) * L.sites**2
