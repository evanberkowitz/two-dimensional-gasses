import torch
from tdg.observable import observable, callable_observable

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
    UUPlusOneInverseUU = ensemble._matrix_to_tensor(ensemble._UUPlusOneInverseUU)
    second = torch.einsum('caast,caats->ca',
                            UUPlusOneInverseUU,
                            UUPlusOneInverseUU,
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
def contact(ensemble, method='bosonic'):
    r'''
    The `contact`, :math:`C\Delta x^2 = 2\pi\frac{d\tilde{H}}{d\log a}`.

    The ``bosonic`` method uses automatic differentiation to compute :math:`dH/dC_R` and the ensemble's :class:`Tuning` to compute :math:`dC_R / d\log a`.
    Just as the `bosonic` method for :func:`~.n` is extremely noisy compared to the ``fermionic`` method, so too is the ``bosonic`` action noisy.
    
    .. todo::
        
        In fact, it is SO NOISY that it has not been checked for correctness by comparing with an exact Trotterized two-body calcuation.

    The ``fermionic`` method is much less noisy by comparison, computing tensor contractions.
    In the case where the only :class:`~.LegoSphere` in the interaction is the on-site interaction, the ``fermionic`` method is accelerated by computing the :func:`~.DoubleOccupancy`.

    Parameters
    ----------
        method: str
            The approach for calculating the number densities ``fermionic`` or ``bosonic``.

    Returns
    -------
        torch.tensor
            One per configuration.

    .. note::

        The method combines matrix elements with the derivative of the Wilson coefficients with respect to :math:`\log a` through :meth:`~.Tuning.dC_dloga`.
        Therefore the ``ensemble.Action`` must have a tuning!

    '''
    # For the on-site interaction we have a shortcut, which is a good acceleration because the LegoSphere is a delta function
    # and thus we can do contractions that don't cost volume^2 but simply volume.
    if len(ensemble.Action.Tuning.radii) == 1 and all(r==0 for r in ensemble.Action.Tuning.radii[0]):
        # This shortcut was implemented and tested AFTER the remaining portion of this routine was,
        # so that what is below was checked to be correct for the on-site interaction.
        logger.info('Calculating the contact via the double occupancy.')
        return 2*torch.pi * ensemble.Action.Tuning.dC_dloga[0] * ensemble.DoubleOccupancy

    logger.info('Using the general form of the fermionic contact.')

    L = ensemble.Action.Spacetime.Lattice
    S = torch.stack(tuple(tdg.LegoSphere(r, c).spatial(L) for c,r in zip(ensemble.Action.Tuning.dC_dloga, ensemble.Action.Tuning.radii) )).sum(axis=0).to(ensemble._UUPlusOneInverseUU.dtype)

    # The contractions looks just like the doubleOccupancy contractions, but with two spatial indices tied together just like the spins are,
    # and then summed with the derivative LegoSphere stencil.
    UUPlusOneInverseUU = ensemble._matrix_to_tensor(ensemble._UUPlusOneInverseUU)
    first = torch.einsum(
            'ab,caass,cbbtt->c',
            S,
            UUPlusOneInverseUU,
            UUPlusOneInverseUU,
            )
    second = torch.einsum(
            'ab,cbats,cabst->c',
            S,
            UUPlusOneInverseUU,
            UUPlusOneInverseUU,
            )
    return torch.pi * (first-second)

@observable
def contact_bosonic(ensemble):
    with torch.autograd.forward_ad.dual_level():
        C0_dual = torch.autograd.forward_ad.make_dual(ensemble.Action.Tuning.C, ensemble.Action.Tuning.dC_dloga)
        V_dual  = tdg.Potential(*[c * tdg.LegoSphere(r) for c,r in zip(C0_dual, ensemble.Action.Tuning.radii)])
        S_dual  = tdg.Action(ensemble.Action.Spacetime, V_dual, ensemble.Action.beta, ensemble.Action.mu, ensemble.Action.h, ensemble.Action.fermion)

        s_dual  = functorch.vmap(S_dual)(ensemble.configurations)
        return  (2*torch.pi / ensemble.Action.beta)* torch.autograd.forward_ad.unpack_dual(s_dual).tangent

