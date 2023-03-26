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

    return 2*torch.pi * L.mod(k + k.transpose(1,0)).transpose(0,2) / L.nx


@observable
def current(ensemble):
    r'''
    The lattice-exact baryon number current :math:`ML^2 j^i_x` so that :math:`\nabla \cdot j_x = - \partial_t \texttt{n}`
    where the divergence is lattice-exact so that the total divergence is zero by periodic boundary conditions and
    conservation of baryon number.

    Configurations slowest, then space `x`, then direction `i`.

    .. todo::
       The expectation value is 0 by translational and rotational invariance.
       However, the real (observable) part is zero configuration-by-configuration, currently a mystery.
    '''

    L = ensemble.Action.Spacetime.Lattice

    TiG = torch.einsum('kqi,ckqss->ckqi', ensemble._baryon_number_current_tensor, ensemble.G_momentum)
    return L.sites / 2 * torch.einsum('cxxi->cxi', L.ifft(L.fft(TiG, axis=1), axis=2))

