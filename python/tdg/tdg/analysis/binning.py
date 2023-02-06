import torch

class Binning:
    r'''
    A binning is built from an ensemble, on which observables are computed, and a width, over which observables are averaged.

    If the width does not evenly divide the length of the ensemble, some configurations are dropped from the front of the ensemble.

    For samples with weights :math:`w`, the new weights are given by the mean weight of the bin, while the new observable value is given by the weighted mean.

    Parameters
    ----------
        ensemble: :class:`~.ensemble.GrandCanonical`
            The ensemble from which sample observables are drawn.
        width: int
            The strides over which to average observables.

    Any observable that :code:`ensemble` supports can be called from the :class:`~.Binning`.
    :class:`~.Binning` uses :code:`__getattr__` trickery under the hood to intercept calls and perform the average transparently.
    '''

    def __init__(self, ensemble, width):
        self.Ensemble = ensemble
        r'''The ensemble underlying the binning.'''
        self.width = width
        r'''The width over which to average'''
        cfgs  = ensemble.configurations.shape[0]
        self.drop  = cfgs % self.width
        r'''How many configurations are dropped from the start of the ensemble.'''
        self.bins  = (cfgs - self.drop) // self.width
        r'''How many bins are in the binning.'''
        self.weights = torch.stack([w.mean(axis=0) for w in self.Ensemble.weights[self.drop:].split(self.width, dim=0)])
        r'''The weight of each bin.'''
        self.index = torch.stack([(0.+i).mean(axis=0) for i in self.Ensemble.index[self.drop:].split(self.width, dim=0)])

    def __len__(self):
        return self.bins

    def _bin(self, obs):
        O = torch.einsum('c,c...->c...',
                         self.Ensemble.weights[self.drop:],
                         obs[self.drop:]
                         ).split(self.width, dim=0)
        averaged = torch.stack([o.mean(axis=0) for o in O])
        return averaged

    def __getattr__(self, name):

        try:    forward = self.Ensemble.__getattribute__(name)
        except: forward = self.Ensemble.__getattr__(name)

        if callable(forward):
            def curry(*args, **kwargs):
                return self._bin(forward(*args, **kwargs))
            return curry
        return self._bin(forward)

