import torch

import tdg
from tdg.observable import observable

import logging
logger = logging.getLogger(__name__)


@observable
def _baryon_number_current_tensor(ensemble):
    # A tensor with momentum indices k, q and direction i.
    #
    #   2π (k+q)^i / nx
    #

    L = ensemble.Action.Spacetime.Lattice
    k = L.coordinates[None].expand(L.sites, L.sites, 2)

    return 2*torch.pi * L.mod(k + k.transpose(1,0)) / L.nx


@observable
def current(ensemble):
    r'''
    The lattice-exact baryon number current :math:`ML^2 j^i_x` so that :math:`\nabla \cdot j_x = - \partial_t \texttt{n}`
    where the divergence is lattice-exact so that the total divergence is zero by periodic boundary conditions and
    conservation of baryon number.

    Configurations slowest, then space `x`, then direction `i`.

    The expectation value is 0 by translational and rotational invariance.
    But, more is true: the real part is zero configuration-by-configuration, and the imaginary part sums to zero.
    '''

    L = ensemble.Action.Spacetime.Lattice

    TiG = torch.einsum('kqi,ckqss->ckqi', ensemble._baryon_number_current_tensor, ensemble.G_momentum)
    return L.sites / 2 * torch.einsum('cxxi->cxi', L.ifft(L.fft(TiG, axis=1), axis=2))

@observable
def _current_current(ensemble):

    # This is the fast, memory-efficient implementation of j†(x) j(y).
    #
    # j†(x)•j(y) = 1/4 V^2 sum_{pjkq στ} e^{2πi[ -(p-j)x + (q-k) y]/Nx} (2π/Nx)^2 (j+p)•(k+q) ψ†(p,σ) ψ(j,σ) ψ†(k,τ) ψ(q,τ)
    #
    # where on the right-hand side k, q, p, and j (sorry) are momenta, σ and τ are spin indices,
    # and there's a dot product on the vector indices of the momenta.
    #
    # The leading 1/V^2 are absorbed by the contractions of the fermion operators into momentum-space propagators.

    L = ensemble.Action.Spacetime.Lattice
    p = (2*torch.pi / L.nx) * L.coordinates + 0.j
    
    # Our strategy is to write the three contractions and expand the dot product into four terms,
    #
    #   j•k, j•q, p•k, p•q
    #
    # yielding twelve things that must be summed.

    # What is nice about this approach is that we can combine propagators and momenta into tensors
    # that are not much bigger than a propagator, O(volume^2).  In contrast the reference implementation
    # requires O(volume^4) memory.

    # Moreover, we can fourier transform back to position space immediately.  The left indices (which correspond to ψ†)
    # always get an fft, the right indices always get an ifft. The fourier transform pairs ALSO have a 1/V;
    # our results have two of them.  Therefore, we need to multiply back in two factors of the volume. However, to save
    # on some arithmetic we delay these multiplications to the end.
    
    prop = ensemble.G_momentum
    G  = L.ifft(L.fft(prop,                                             axis=1), axis=2)
    pG = L.ifft(L.fft(torch.einsum('ki,ckqst->ckiqst', p, prop),        axis=1), axis=3)
    Gp = L.ifft(L.fft(torch.einsum('ckqst,qi->ckqist', prop, p),        axis=1), axis=2)
    pGp= L.ifft(L.fft(torch.einsum('ki,ckqst,qi->ckqst', p, prop, p),   axis=1), axis=2) 
    # for the last one we can do the ^ i  sum ^ immediately.
    
    delta  = torch.eye(L.sites)
    d  = L.ifft(L.fft(delta,                                            axis=0), axis=1)
    pd = L.ifft(L.fft(torch.einsum('ki,kq->kiq', p, delta),             axis=0), axis=2)
    dp = L.ifft(L.fft(torch.einsum('kq,qi->kqi', delta, p),             axis=0), axis=1)
    pdp= L.ifft(L.fft(torch.einsum('ki,kq,qi->kq', p, delta, p),        axis=0), axis=1)
    # similarly, we can dot product  ^ sum ^ immediately.

    # Having done the fourier transforms we can use these 8 objects and think of their indices as being
    # in position space.  We can then do the contractions as needed, simply replacing
    #
    #   k, q --> y
    #   j, p --> x
    #
    # based on the fourier transform above.
    #
    # Finally we reincorporate the volume factors from the fourier transforms, the leading 1/4.
    return (1/4 * L.sites**2) * (
        torch.einsum('cyiytt,cxxiss->cxy', pG, Gp)
    +   torch.einsum('cyiytt,cxixss->cxy', pG, pG)
    +   torch.einsum('cyyitt,cxxiss->cxy', Gp, Gp)
    +   torch.einsum('cyyitt,cxixss->cxy', Gp, pG)
    -   torch.einsum('cyxts,cxyst->cxy',   pGp, G)
    -   torch.einsum('cyxits,cxyist->cxy', Gp, Gp)
    -   torch.einsum('cyixts,cxiyst->cxy', pG, pG)
    -   torch.einsum('cyxts,cxyst->cxy',   G, pGp)
    +   torch.einsum('yx,cxyss->cxy',      pdp, G)
    +   torch.einsum('yxi,cxyiss->cxy',    dp, Gp)
    +   torch.einsum('yix,cxiyss->cxy',    pd, pG)
    +   torch.einsum('yx,cxyss->cxy',      d, pGp)
    )

@observable
def current_squared(ensemble):
    r'''
    The local square of the dimensionless current :math:`\tilde{\jmath}^2_x`, with a dot product of the vector indices.

    Configurations slowest, then spatial coordinate :math:`x`.
    '''

    return torch.einsum('cxy->cx', ensemble._current_current)

@observable
def current_current(ensemble):
    r'''
    The convolution of the dimensionless current with itself :math:`(\tilde{\jmath}^i*\tilde{\jmath}^i)_r`, with a dot product of the vector indices.

    Configurations slowest, then relative coordinate :math:`r`.
    '''

    L = ensemble.Action.Spacetime.Lattice


    # These two lines should differ only in speed:
    #
    #   return torch.einsum('cxy,yrx->cr', ensemble._current_current, 0.j+L.convolver)
    #
    return L.fft(torch.einsum('ckk->ck', L.fft(L.ifft(ensemble._current_current, axis=2), axis=1)), axis=1) / L.sites
