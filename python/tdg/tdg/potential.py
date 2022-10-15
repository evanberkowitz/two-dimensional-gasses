#!/usr/bin/env python3

from functools import cached_property
from functools import lru_cache as cached
import numpy as np
import torch

class Potential:

    def __init__(self, *spheres):
        self.spheres = [*spheres]

    def __str__(self):
        return f"Potential({', '.join([str(s) for s in self.spheres])})"

    def __repr__(self):
        return str(self)

    @cached
    def spatial(self, lattice):
        return torch.sum(torch.stack([s.spatial(lattice) for s in self.spheres]), axis=0)

    @cached
    def inverse(self, lattice):
        return torch.linalg.inv(self.spatial(lattice))

    @cached
    def eigvals(self, lattice):
        e = torch.linalg.eigvals(self.spatial(lattice))
        if (e.imag != 0).all():
            raise TypeError(f"{self} yields imaginary eigenvalues.")
        if (e.real >= 0).any():
            raise ValueError(f"{self} yields positive eigenvalues; we require the attractive channel.")
        #return e.real ?
        return e

    @cached_property
    def C0(self):
        c0 = 0
        for s in self.spheres:
            if (0 == s.r).all() :
                c0 += s.c
        return c0
