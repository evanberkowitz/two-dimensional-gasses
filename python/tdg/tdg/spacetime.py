#!/usr/bin/env python

import torch
import numpy as np

class Spacetime:

    def __init__(self, nt, lattice):
        self.nt   = nt
        self.Lattice = lattice

        self.t    = torch.arange(0, nt)
        self.sites= nt * lattice.sites

        self.dims = torch.Size([nt, *self.Lattice.dims])

        # A linearized list of coordinates.
        # Each timeslice matches lattice.coordinates
        self.coordinates = torch.cat(tuple(
            torch.cat((
                t * torch.ones(self.Lattice.sites,1, dtype=torch.int),
                self.Lattice.coordinates),
                1)
            for t in range(self.nt))
            )

        self.TX = torch.stack(tuple(self.coordinates[:,x].reshape(self.dims) for x in range(len(self.dims))))

    def __str__(self):
        return f'Spacetime(nt={self.nt}, {str(self.Lattice)})'

    def __repr__(self):
        return str(self)

    def vector(self, *dims):
        return self.Lattice.vector(*dims, self.nt)
