import torch
from tdg.h5 import H5able
from tdg.performance import Timer

from functools import lru_cache as cached

import logging
logger = logging.getLogger(__name__)

class Bootstrap(H5able):
    r'''
    The bootstrap is a resampling technique for estimating uncertainties.

    For samples with weights :math:`w` the expectation value of an observable is

    .. math::
        \left\langle O \right\rangle = \frac{\left\langle O w \right\rangle}{\left\langle w \right\rangle}

    and an accurate bootstrap estimate of the left-hand side requires tracking the correlations between the numerator and denominator.
    Moreover, quoting correlated uncertainties requires resampling different observables in the same way.

    Parameters
    ----------
        ensemble:   :class:`~.GrandCanonical` or :class:`~.Binned`
            The ensemble to resample.
        draws:      int
            The number of times to resample.

    Any observable that :code:`ensemble` supports can be called from the :class:`~.Bootstrap`.
    :class:`~.Bootstrap` uses :code:`__getattr__` trickery under the hood to intercept calls and perform the weighted average transparently.

    Each observable returns a :code:`torch.tensor` of the same dimension as the ensemble's observable.  However, rather than configurations first, :code:`draws` are first.

    Each draw is a weighted average over the resampled weight, as shown above, and is therefore an estimator for the expectation value.
    These are guaranteed (by the `central limit theorem`_) to be normally distributed.
    To get an uncertainty estimate one need only take the :code:`.mean()` for a central value and :code:`.std()` for the uncertainty on the mean.

    .. _central limit theorem: https://en.wikipedia.org/wiki/Central_limit_theorem
    '''

    _observables = set()
    # The _observables are populated by the @observable and @derived decorators.

    _intermediates = set()
    # The _intermediates are populated by the @intermediate and @derived_intermediate decorator.

    def __init__(self, ensemble, draws=100):
        self.Ensemble = ensemble
        r'''The ensemble from which to resample.'''
        self.Action = ensemble.Action
        r'''The action underlying the ensemble.'''
        self.draws = draws
        r'''The number of resamplings.'''
        cfgs = ensemble.configurations.shape[0]
        self.indices = torch.randint(0, cfgs, (cfgs, draws))
        r'''The random draws themselves; configurations × draws.'''
        
    def __len__(self):
        return self.draws

    def _resample(self, obs):
        # Each observable should be multiplied by its respective weight.
        # Each draw should be divided by its average weight.
        w = self.Ensemble.weights[self.indices]

        # This index ordering is needed to broadcast the weights division correctly.
        # See https://github.com/evanberkowitz/two-dimensional-gasses/issues/55
        # We return the bootstrap axis to the front to provide an analogous interface for Bootstrap and GrandCanonical quantities.
        return torch.einsum('...d->d...', torch.einsum('cd,cd...->c...d', w, obs[self.indices]).mean(axis=0) / w.mean(axis=0))
    
    @cached
    def __getattr__(self, name):
        
        try:    forward = self.Ensemble.__getattribute__(name)
        except: forward = self.Ensemble.__getattr__(name)
        
        if callable(forward):
            @cached
            def curry(*args, **kwargs):
                return self._resample(forward(*args, **kwargs))
            return curry
        return self._resample(forward)
