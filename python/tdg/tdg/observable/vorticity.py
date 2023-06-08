import torch

import tdg
from tdg.observable import observable, derived

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

@observable
def _vorticity_vorticity(ensemble):
    # This is the fast, memory-efficient implementation of ω†(x) ω(y).
    # It can be compared to the reference implementation, which scales poorly.
    #
    # ω†(x)•ω(y) = 1/4 V^2 sum_{pjkq στ} e^{2πi[ -(p-j)x + (q-k) y]/Nx} (2π/Nx)^2 (p×j)•(q×k) ψ†(p,σ) ψ(j,σ) ψ†(k,τ) ψ(q,τ)
    #
    # where on the right-hand side k, q, p, and j are momenta, σ and τ are spin indices,
    # and there's a dot product on the vector indices of the cross products.
    #
    # The leading 1/V^2 are absorbed by the contractions of the fermion operators into momentum-space propagators.

    L = ensemble.Action.Spacetime.Lattice
    p = (2*torch.pi / L.nx) * L.coordinates + 0.j
    
    # Our strategy is to write the three contractions and expand the dot product into two terms,
    #
    #   (p•q)(j•k) - (p•k)(j•q)
    #
    # yielding six things that must be summed.

    # What is nice about this approach is that we can combine propagators and momenta into tensors
    # that are not much bigger than a propagator, O(volume^2), and fourier back to position space.
    # Each fft/ifft pair brings a factor of 1/volume which we need to restore, 
    # but to save on arithmetic we delay until later.
    G  = ensemble.G_momentum
    xGx= L.ifft(L.fft(torch.einsum('ka,ckqst,qb->ckaqbst', p, G, p), axis=1), axis=3)
    
    d  = torch.eye(L.sites)
    xdx= L.ifft(L.fft(torch.einsum('ka,kq,qb->kaqb', p, d, p), axis=0), axis=2)

    # THERE IS AN EXTRAORDINARY SUBTLETY HERE
    # =======================================
    # | By Fourier transforming into position space first we mod all the momenta into the BZ.
    # | Therefore, for cross products of sums of vectors that could land outside of the BZ,
    # | this mods the sums into the BZ first, and then cross-multiplying.
    # |
    # | In constrast, if you do not mod first, you will find different answers.  In particular,
    # | in the free theory we can compute both ways and explicitly get different answers.
    # | However, this is an irrelevant UV choice: these differences vanish in the spatial continuum limit.
    # =======================================
    
    # Now we do the six (= 2 momentum structures * 3 Wick structures) tensor contractions
    # and multiply back one factor of volume per fft/ifft pair.
    return L.sites**2 * (
    +   torch.einsum('cxaxbss,cybyatt->cxy', xGx, xGx) #   unmixed Wick (p•q)(j•k)
    -   torch.einsum('cxaxbss,cyaybtt->cxy', xGx, xGx) #   unmixed Wick (p•k)(j•q)
    -   torch.einsum('cxayast,cybxbts->cxy', xGx, xGx) #     mixed Wick (p•q)(j•k)
    +   torch.einsum('cxaybst,cyaxbts->cxy', xGx, xGx) #     mixed Wick (p•k)(j•q)
    +   torch.einsum('cxayast,ybxb->cxy'   , xGx, xdx) # anticommutator (p•q)(j•k)
    -   torch.einsum('cxaybst,yaxb->cxy'   , xGx, xdx) # anticommutator (p•k)(j•q)
    )

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

    By periodic boundary conditions, should vanish when summed on the radial coordinate.

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

@derived
def b2_by_kF4(ensemble):
    r'''
    :math:`M^2 B_2(k=0) / k_F^4` which is a non-zero low-energy moment of :math:`\Omega`;

    .. math::
        
    '''

    L = ensemble.Action.Spacetime.Lattice
    rsq =   0.j + L.linearize(L.rsq)
    Omega = ensemble.vorticity_vorticity

    return torch.einsum('br,r->b', Omega, rsq) / (2*torch.pi * ensemble.N)**2

@derived
def b4_by_kF2(ensemble):
    r'''
    :math:`M^2 B_4(k=0) / k_F^2`.
    '''
    L     = ensemble.Action.Spacetime.Lattice
    r2    = 0.j+L.linearize(L.rsq)
    Omega = ensemble.vorticity_vorticity

    return torch.einsum('br,r->b', Omega, r2**2) / (2*torch.pi*ensemble.N)**1 / L.nx**2

@observable
def b6(ensemble):
    r'''
    :math:`M^2 B_6(k=0)`.
    '''

    L     = ensemble.Action.Spacetime.Lattice
    r2    = 0.j+L.linearize(L.rsq)
    Omega = ensemble.vorticity_vorticity

    return torch.einsum('br,r->b', Omega, r2**3) / L.nx**4
