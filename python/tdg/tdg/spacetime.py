#!/usr/bin/env python

import torch
import numpy as np

class Spacetime:

    def __init__(self, nt, lattice):
        self.nt   = nt
        self.Lattice = lattice

        self.t    = torch.arange(0, nt)
        self.sites= nt * lattice.sites

        # These are chosen so that they have shape (nt, nx, ny)
        self.T = torch.tile( self.t, (self.Lattice.nx, self.Lattice.ny, 1)).permute(2,0,1)
        self.X = torch.tile( self.Lattice.X, (1, self.nt, 1)).reshape(nt, *self.Lattice.X.shape)
        self.Y = torch.tile( self.Lattice.Y, (1, self.nt, 1)).reshape(nt, *self.Lattice.Y.shape)
        # and each timeslice matches the X and Y of the underlying lattice,
        #print( (self.X[0] == self.Lattice.X).all() and (self.Y[0] == self.Lattice.Y).all() )

        # A linearized list of coordinates.
        # Each timeslice matches lattice.coordinates
        self.coordinates = torch.stack((self.T.flatten(), self.X.flatten(), self.Y.flatten())).T

        self.dims = self.T.shape

    def __str__(self):
        return f'Spacetime(nt={self.nt}, {str(self.Lattice)})'

    def __repr__(self):
        return str(self)

    def vector(self):
        return torch.zeros(self.dims)
