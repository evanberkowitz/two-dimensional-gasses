#!/usr/bin/env python

import torch
import numpy as np

class Spacetime:

    def __init__(self, nt, lattice):
        self.nt   = nt
        r'''The number of timeslices.'''
        self.Lattice = lattice
        r'''The spatial lattice.'''

        self.t    = torch.arange(0, nt)
        r'''The coordinates in the time direction.'''
        self.sites= nt * lattice.sites
        r'''The total spacetime (integer) volume.'''

        self.dims = torch.Size([nt, *self.Lattice.dims])
        r'''The spacetime dimensions.'''

        # A linearized list of coordinates.
        # Each timeslice matches lattice.coordinates
        self.coordinates = torch.cat(tuple(
            torch.cat((
                t * torch.ones(self.Lattice.sites,1, dtype=torch.int),
                self.Lattice.coordinates),
                1)
            for t in range(self.nt))
            )
        r'''A tensor of size ``[sites, len(dims)]``.  Each row contains a set of coordinates.
        Time is slowest and each timeslice matches Lattice.coordinates in order.
        '''

        self.TX = torch.stack(tuple(self.coordinates[:,x].reshape(self.dims) for x in range(len(self.dims))))

    def __str__(self):
        return f'Spacetime(nt={self.nt}, {str(self.Lattice)})'

    def __repr__(self):
        return str(self)

    def vector(self, *dims):
        r'''
        Parameters
        ----------
            dims:   tuple
                Specifies how how many spacetime vectors to produce.
                
        Returns
        -------
            torch.tensor:
                A ``dims``-dimensional stack of zero vectors.  Each vector is of ``.shape == [nt, Lattice.vector().shape]``.
        '''

        return self.Lattice.vector(*dims, self.nt)
