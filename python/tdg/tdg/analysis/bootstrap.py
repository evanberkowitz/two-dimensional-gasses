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

    def measure(self, *observables):
        r'''
        Compute each :ref:`@observable <observables>` and @derived quantity in `observables`; log an error for any :ref:`unregistered observable <custom observables>` or derived quantity.

        Parameters
        ----------
            observables: strings
                Names of observables or derived quantities.

        Returns
        -------
            :class:`~.Bootstrap`; itself, now with some observables and derived quantities bootstrapped.

        .. note::

            If no `observables` are passed, bootstraps **every** registered `@observable` and `@derived` quantity.

        '''
        if not observables:
            observables = self._observables

        with Timer(logger.info, f'Bootstrap of {len(self)} draws', per=len(self)):
            for observable in observables:
                if observable not in self._observables | self._intermediates:
                    logger.error(f'No registered observable "{observable}"')
                    continue
                try:
                    getattr(self, observable)
                except AttributeError as error:
                    logger.error(str(error))
        return self

    def _resample(self, obs):
        # Each observable should be multiplied by its respective weight.
        # Each draw should be divided by its average weight.
        w = self.Ensemble.weights[self.indices]

        # This index ordering is needed to broadcast the weights division correctly.
        # See https://github.com/evanberkowitz/two-dimensional-gasses/issues/55
        # We return the bootstrap axis to the front to provide an analogous interface for Bootstrap and GrandCanonical quantities.
        return torch.einsum('...d->d...', torch.einsum('cd,cd...->c...d', w, obs[self.indices]).mean(axis=0) / w.mean(axis=0))
    
    def __getattr__(self, name):
        
        if name not in self._observables | self._intermediates:
            raise AttributeError(name)

        with Timer(logger.info, f'Bootstrapping {name}', per=len(self)):

            try:    forward = self.Ensemble.__getattribute__(name)
            except: forward = self.Ensemble.__getattr__(name)

            ## I believe this code block is a remnant from callable observables.
            ## Callable observables were eliminated in #58 https://github.com/evanberkowitz/two-dimensional-gasses/pull/58
            ## to keep the interface much simpler.
            ##
            ## Normally I'd just delete this block of code.  But since I'm not POSITIVE this __getattr__ isn't doing something
            ## else clever I'll just comment it out for now and remove it permanently in the future.
            ##
            ## TODO: remove this if its absence hasn't caused a problem.
            ##       If it is removed, we can also remove the importing of lru_cache as cached above.
            ##       Marked for deletion on 21 July 2023.
            #
            # if callable(forward):
            #     @cached
            #     def curry(*args, **kwargs):
            #         return self._resample(forward(*args, **kwargs))
            #     return curry
            #
            ## In fact, it may even be that the try/except above can go too; that was also about intercepting calls.

            self.__dict__[name] = self._resample(forward)
            return self.__dict__[name]
