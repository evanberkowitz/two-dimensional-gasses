import torch
from tdg.observable import observable

@observable
def _n_fermionic(ensemble):
    return torch.einsum('caass->ca',
                        ensemble.G
    )

@observable
def _n_bosonic(ensemble):
    Vinv = ensemble.Action.Potential.inverse(ensemble.Action.Spacetime.Lattice)
    return -torch.einsum(
                'ab,ctb->cta',
                Vinv + 0j,
                ensemble.configurations
                ).mean(1)/ (ensemble.Action.beta/ensemble.Action.Spacetime.nt)

@observable
def _N_fermionic(ensemble):
    return ensemble._n_fermionic.sum(1)

@observable
def _N_bosonic(ensemble):
    return ensemble._n_bosonic.sum(1)

@observable
def n(ensemble, method='fermionic'):
    r'''
    The local number density, one per site per configuration.

    Parameters
    ----------
        method: str
            The approach for calculating the local number densities, ``'fermionic'`` or ``'bosonic'``.

    Returns
    -------
        torch.tensor
            Configurations slowest, then sites.

    .. note ::
        The ``'fermionic'`` method computes a derivative of the fermion determinant, and seems to be positive-(semi?)definite.
        The ``'bosonic'`` method computes the derivative of the gauge action and is not positive-definite.

    '''
    if method == 'fermionic':
        return ensemble._n_fermionic
    if method == 'bosonic':
        return ensemble._n_bosonic
    raise NotImplemented()

@observable
def N(ensemble, method='fermionic'):
    r'''
    The total number, one per configuration.

    Parameters
    ----------
        method: str
            The approach for calculating the number, as in :func:`~.number.n`.

    Returns
    -------
        torch.tensor
    '''
    if method == 'fermionic':
        return ensemble._N_fermionic
    if method == 'bosonic':
        return ensemble._N_bosonic
    raise NotImplemented()

