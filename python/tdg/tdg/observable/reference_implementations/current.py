import torch

import tdg
from tdg.observable import observable

import logging
logger = logging.getLogger(__name__)

logger.info(f'Importing reference implementations in {__name__}.')

@observable
def _current_current(ensemble):
    
    # This is the slow, memory-hungry implementation of j†(x) j(y).
    #
    # j†(x)•j(y) = 1/4 V^2 sum_{pjkq στ} e^{2πi[ -(p-j)x + (q-k) y]/Nx} (2π/Nx)^2 (j+p)•(k+q) ψ†(p,σ) ψ(j,σ) ψ†(k,τ) ψ(q,τ)
    #
    # where on the right-hand side k, q, p, and j (sorry) are momenta, σ and τ are spin indices,
    # and there's a dot product on the vector indices of the momenta.

    L = ensemble.Action.Spacetime.Lattice
    momenta = L.coordinates
    
    # An obviously correct implementation of the momentum sums is
    #
    #   momentum_sum = torch.stack(tuple(k+momenta for k in momenta))
    #
    # but this can be done faster using pytorch trickery.
    momentum_sum = torch.reshape(momenta, (-1,1,2)) + momenta
    #
    # Should you mod the sum, like
    #
    #   momentum_sum = torch.stack(tuple(L.mod(k+momenta) for k in momenta))
    #
    #                                    ^^^^^
    #
    # ?  NO.  The sum does NOT represent the momentum of a single particle (operator).
    # Therefore L.mod does not take the correct mod!
    
    # Includes the leading 1/4:
    dot_product = 0.25 * (2 * torch.pi / L.nx)**2 * torch.einsum('jpi,kqi->kqjp', momentum_sum, momentum_sum)
    
    # The contractions capture the 1/V^2.
    contractions = (
        torch.einsum('ckqtt,cpjss->ckqpj', ensemble.G_momentum, ensemble.G_momentum)
    -   torch.einsum('ckjts,cpqst->ckqpj', ensemble.G_momentum, ensemble.G_momentum)
    +   torch.einsum('cpqss,kj->ckqpj',    ensemble.G_momentum, torch.eye(L.sites))
    )
    
    return torch.einsum('cyyxx->cxy', L.ifft(L.fft(L.ifft(L.fft(
        torch.einsum('ckqpj,kqjp->ckqpj', contractions, dot_product),
        axis=1), axis=2), axis=3), axis=4)) * L.sites**2
