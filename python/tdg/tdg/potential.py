#!/usr/bin/env python3

from functools import cached_property
import torch

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
        return torch.sum(torch.stack([s.spatial(self.Lattice) for s in self.spheres]), axis=0)

    @cached_property
    def inverse(self):
        return self.Lattice.linearized_tensor(torch.linalg.inv(self.Lattice.tensor_linearized(self.spatial)))

    @cached_property
    def eigvals(self):
        e = torch.linalg.eigvals(self.Lattice.tensor_linearized(self.spatial))
        if (e.imag != 0).all():
            raise TypeError(f"{self} yields imaginary eigenvalues.")
        if (e.real <= 0).any():
            raise ValueError(f"{self} yields negative eigenvalues.")
        #return e.real ?
        return e

    @cached_property
    def C0(self):
        c0 = 0
        for s in self.spheres:
            if( torch.tensor([0,0]) == s.r ):
                c0 += s.c
        return c0
