#!/usr/bin/env python

import numpy as np

class Spacetime:

    def __init__(self, beta, nt, lattice):
        self.beta = beta
        self.nt   = nt
        self.dt   = beta / nt
        self.Lattice = lattice

        self.dims = np.concatenate(([self.nt], self.Lattice.dims))
        self.t    = np.arange(0, nt)

        # These are chosen so that they have shape (nt, nx, ny)
        self.T = np.tile( self.t, (len(self.Lattice.x), len(self.Lattice.y), 1)).transpose([2,0,1])
        self.X = np.tile( self.Lattice.X, (1, self.nt, 1)).reshape(nt, self.Lattice.X.shape[0], self.Lattice.X.shape[1])
        self.Y = np.tile( self.Lattice.Y, (1, self.nt, 1)).reshape(nt, self.Lattice.Y.shape[0], self.Lattice.Y.shape[1])
        # and each timeslice matches the X and Y of the underlying lattice,
        #print( (self.X[0] == self.Lattice.X).all() and (self.Y[0] == self.Lattice.Y).all() )

        # A linearized list of coordinates.
        self.coordinates = [(t,x,y) for t,x,y in zip(self.T.flat, self.X.flat, self.Y.flat)]
        # Each timeslice matches lattice.coordinates,
        #self.coordinates = np.array(self.coordinates)
        #print((self.coordinates[:(self.dims[1]*self.dims[2])][:,[1,2]] == self.Lattice.coordinates).all())

    def __str__(self):
        return f'Spacetime(β={self.beta}, nt={self.nt}, {str(self.Lattice)})'

    def __repr__(self):
        return str(self)

    def vector(self, dtype=complex):
        return np.zeros(self.dims, dtype=dtype)