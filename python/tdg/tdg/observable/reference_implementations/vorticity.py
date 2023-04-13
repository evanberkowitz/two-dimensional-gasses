import torch

import tdg
from tdg.observable import observable

import logging
logger = logging.getLogger(__name__)

logger.info(f'Importing reference implementations in {__name__}.')

# 

# This was hoped to be the fast, memory-efficient implementation of ω†(x) ω(y).
# It uses the trick of combining momenta with propagators to construct useful intermediate quantities,
# like the production implementation of _current_current.
@observable
def _vorticity_vorticity(ensemble):
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
    # that are not much bigger than a propagator, O(volume^2).

    G  = ensemble.G_momentum
    pGp= torch.einsum('ka,ckqst,qb->ckaqbst', p, G, p) 
    
    d  = torch.eye(L.sites)
    pdp= torch.einsum('ka,kq,qb->kaqb', p, d, p)
    
    
    # Unfortunately, unlike the current-current correlator, it seems we still have to construct
    # an object of O(volume^4) in memory.  This stems from the fact that each momentum appears
    # in each term of (p•q)(j•k) - (p•k)(j•q); in the current case the terms had only 2 momenta.
    fouriered = (
    +   torch.einsum('cpajbss,ckbqatt->cpjkq', pGp, pGp)
    -   torch.einsum('cpajbss,ckaqbtt->cpjkq', pGp, pGp)
    -   torch.einsum('cpaqast,ckbjbts->cpjkq', pGp, pGp)
    +   torch.einsum('cpaqbst,ckajbts->cpjkq', pGp, pGp)
    +   torch.einsum('cpaqast,kbjb->cpjkq'   , pGp, pdp)
    -   torch.einsum('cpaqbst,kajb->cpjkq'   , pGp, pdp)
    )

    # Now we can fourier back.  The left indices (which correspond to ψ†) always get an fft,
    # the right indices always get an ifft. The fourier transform pairs ALSO have a 1/V;
    # our results have two of them.  Therefore, we need to multiply back in two factors of the volume. However, to save
    # on some arithmetic we delay these multiplications to the end.

    position = L.fft(L.ifft(
                    L.fft(L.ifft(
                        fouriered,
                        axis=4), axis=3),
                    axis=2), axis=1)
    
    # Now we've got something with four position indices.  We want to evaluate the piece that has 
    # p, j talking to x and q, k talking to y.  We simply replace
    #
    #   k, q --> y
    #   j, p --> x
    #
    # based on the fourier transform above.  Finally we multiply in the two factors of volume.
    
    return L.sites**2 * torch.einsum('cxxyy->cxy', position)
