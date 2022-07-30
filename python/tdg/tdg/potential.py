#!/usr/bin/env python3

from functools import cached_property
import numpy as np

class Potential:

    def __init__(self, Lattice, *spheres):
        self.Lattice = Lattice
        self.spheres = [*spheres]

    def __str__(self):
        return f"Potential({self.Lattice} with {', '.join([str(s) for s in self.spheres])})"

    def __repr__(self):
        return str(self)

    @cached_property
    def spatial(self):
        return np.sum([s.spatial(self.Lattice) for s in self.spheres], axis=0)

    @cached_property
    def inverse(self):
        return self.Lattice.linearized_tensor(np.linalg.inv(self.Lattice.tensor_linearized(self.spatial)))

    @cached_property
    def eigvals(self):
        return np.linalg.eigvals(self.Lattice.tensor_linearized(self.spatial))

