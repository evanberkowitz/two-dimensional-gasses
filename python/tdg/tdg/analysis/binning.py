import torch

class Binning:

    def __init__(self, ensemble, width):
        self.Ensemble = ensemble
        self.width = width
        cfgs  = ensemble.configurations.shape[0]
        self.drop  = cfgs % self.width
        self.bins  = (cfgs - self.drop) // self.width

    def _bin(self, obs):
        O = obs[self.drop:].split(self.width, dim=0)
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

