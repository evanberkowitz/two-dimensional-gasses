import torch
import functorch

import tdg
from tdg.observable import observable

import logging
logger = logging.getLogger(__name__)


@observable
def doubleOccupancy(ensemble):
    r'''
    The double occupancy of a site is :math:`n_{\uparrow} n_{\downarrow}` on that site (or the equivalent along any direction; not just the :math:`z`-axis),
    which as an operator is equal to :math:`\frac{1}{2}(n^2-n)`, where :math:`n` is the total number operator :func:`~.n`.

    Configuration slowest, then space.
    '''

    # The double occupancy on site a is given by the expectation value of
    #
    #   2 * doubleOccupancy =
    #   sum_{st}
    #     + [ (1+UU)^-1 U ]^{ss}_{aa} [ (1+UU)^-1 U ]^{tt}_{aa}
    #     - [ (1+UU)^-1 U ]^{st}_{aa} [ (1+UU)^-1 U ]^{ts}_{aa}
    #
    # where the lower indices are spatial and the upper indices are spin.
    #
    # The first term is the square of the contraction of the fermionic n operator.
    first = ensemble.n**2
    # The second term is a bit more annoying;
    second = torch.einsum('caast,caats->ca',
                            ensemble.G,
                            ensemble.G,
                         )

    # To get the double occupancy itensemble, take half the difference.
    return 0.5*(first - second)

@observable
def DoubleOccupancy(ensemble):
    r'''
    The spatial sum of the :func:`~.doubleOccupancy`; one per configurtion.
    '''
    return ensemble.doubleOccupancy.sum(axis=1)

@observable
def Contact(ensemble):
    r'''
    The `contact`, :math:`C\Delta L^2 = 2\pi\frac{d\tilde{H}}{d\log a}`.

    This observable is much less noisy than :func:`~.Contact_bosonic`.

    In the case where the only :class:`~.LegoSphere` in the interaction is the on-site interaction, the ``fermionic`` method is accelerated by computing the :func:`~.DoubleOccupancy`.

    .. note::
        The contact :math:`C` is extensive, and :math:`L^2` is extensive, so this observable is *doubly extensive*!
        You may want to compute something intensive, such as the contact density :math:`c` normalized by :math:`k_F^4` :cite:`Beane:2022wcn`,
        :math:`c/k_F^4 = \texttt{Contact} / (2\pi \texttt{N})^2`, where this observable is divided by two powers of an extensive observable!

    '''
    # For the on-site interaction we have a shortcut, which is a good acceleration because the LegoSphere is a delta function
    # and thus we can do contractions that don't cost volume^2 but simply volume.
    if len(ensemble.Action.Tuning.radii) == 1 and all(r==0 for r in ensemble.Action.Tuning.radii[0]):
        # This shortcut was implemented and tested AFTER the remaining portion of this routine was,
        # so that what is below was checked to be correct for the on-site interaction.
        logger.info('Calculating the fermionic Contact via the double occupancy.')
        return 2*torch.pi * ensemble.Action.Spacetime.Lattice.sites * ensemble.Action.Tuning.dC_dloga[0] * ensemble.DoubleOccupancy

    logger.info('Using the general form of the fermionic Contact.')

    L = ensemble.Action.Spacetime.Lattice
    S = torch.stack(tuple(tdg.LegoSphere(r, c).spatial(L) for c,r in zip(ensemble.Action.Tuning.dC_dloga, ensemble.Action.Tuning.radii) )).sum(axis=0).to(ensemble.G.dtype)

    # The contractions looks just like the doubleOccupancy contractions, but with two spatial indices tied together just like the spins are,
    # and then summed with the derivative LegoSphere stencil.
    first = torch.einsum(
            'ab,caass,cbbtt->c',
            S,
            ensemble.G,
            ensemble.G,
            )
    second = torch.einsum(
            'ab,cbats,cabst->c',
            S,
            ensemble.G,
            ensemble.G,
            )
    return torch.pi * L.sites * (first-second)

@observable
def Contact_bosonic(ensemble):
    r'''
    The same expectation value as :func:`~.Contact` using automatic differentiation and the chain rule, evaluating :math:`dH/dC_R` and the ensemble's :class:`Tuning` to compute :math:`dC_R / d\log a`.
    Just as :func:`~.n_bosonic` is extremely noisy in comparison to :func:`~.n`, so too is this noisy compared to :func:`~.Contact`.
    
    .. todo::
        
        In fact, it is SO NOISY that it has not been checked for correctness by comparing with an exact Trotterized two-body calcuation.

    '''
    # This is a "straightforward" application of differentiating Z with respect to log a.
    # We implement it using forward-mode automatic differentiation, promoting the LegoSphere
    # coefficients to dual numbers whose derivatives are given by the derivative of the tuning.
    with torch.autograd.forward_ad.dual_level():
        C0_dual = torch.autograd.forward_ad.make_dual(ensemble.Action.Tuning.C, ensemble.Action.Tuning.dC_dloga)
        V_dual  = tdg.Potential(*[c * tdg.LegoSphere(r) for c,r in zip(C0_dual, ensemble.Action.Tuning.radii)])
        S_dual  = tdg.Action(ensemble.Action.Spacetime, V_dual, ensemble.Action.beta, ensemble.Action.mu, ensemble.Action.h, ensemble.Action.fermion)

        s_dual  = functorch.vmap(S_dual)(ensemble.configurations)
        return  (2*torch.pi / ensemble.Action.beta)* torch.autograd.forward_ad.unpack_dual(s_dual).tangent

