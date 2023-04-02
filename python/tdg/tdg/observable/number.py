import torch
from tdg.observable import observable

@observable
def n_bosonic(ensemble):
    r'''
    Using Ward-Takahashi identities one finds the bosonic estimator

    .. math::
       \left\langle\tilde{n} = - \frac{1}{\tilde{\beta}} \tilde{V}^{-1} \sum_t A_t\right\rangle

    of the local number density, one per site per configuration.

    .. note::
       Has substantially greater variance than the fermionic estimator :func:`~.n`, especially at low particle numbers.
    '''
    Vinv = ensemble.Action.Potential.inverse(ensemble.Action.Spacetime.Lattice)
    return -torch.einsum(
                'ab,ctb->cta',
                Vinv + 0j,
                ensemble.configurations
                ).mean(1)/ (ensemble.Action.beta/ensemble.Action.Spacetime.nt)

@observable
def N_bosonic(ensemble):
    r'''
    The total number, one per configuration.

    The sum of :func:`n_bosonic`.
    '''
    return ensemble._n_bosonic.sum(1)

@observable
def n(ensemble):
    r'''
    The fermionic estimator of the local number density, one per site per configuration.

    Seems to be positive (semi-)definite.
    '''
    return torch.einsum('caass->ca',
                        ensemble.G
    )

@observable
def N(ensemble):
    r'''
    The total number, one per configuration.
    Computed via fermion contractions; the sum of :func:`~.n` on sites.
    '''
    if method == 'fermionic':
        return ensemble._N_fermionic
    if method == 'bosonic':
        return ensemble._N_bosonic
    raise NotImplemented()

