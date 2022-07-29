#!/usr/bin/env python

from functools import cached_property
import numpy as np

class Lattice:

    def __init__(self, nx, ny=None):
        self.nx = nx

        if ny is None:
            self.ny = nx
        else:
            self.ny = ny

        self.dims = np.array([self.nx, self.ny])
        self.sites = self.nx * self.ny

        # We want to go from -n/2 to +n/2
        self.x = np.arange( - (self.nx // 2), self.nx // 2 + 1)
        self.y = np.arange( - (self.ny // 2), self.ny // 2 + 1)
        # (up to even/odd issues)
        if self.nx % 2 == 0:
            self.x = self.x[1:]
        if self.ny % 2 == 0:
            self.y = self.y[1:]

        # However, we want to match the FFT convention of numpy
        # https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details
        # where the lowest coordinate / frequency is at 0, we increase to the max,
        # and then go in decreasingly-negative order.
        origin = np.array([
            np.where(self.x==0)[0][0],
            np.where(self.y==0)[0][0]
            ])
        self.x = np.roll(self.x, -origin[0])
        self.y = np.roll(self.y, -origin[1])

        # These are chosen so that Lattice(nx, ny)
        # has coordinate matrices of size (nx, ny)
        self.X = np.tile( self.x, (len(self.y), 1)).T
        self.Y = np.tile( self.y, (len(self.x), 1))

        # Wavenumbers are the same
        self.kx = self.x
        self.ky = self.y
        self.KX = self.X
        self.KY = self.Y
        # To get the dimensionless wavenumber^2 in the fourier basis we can simply
        self.ksq = self.KX**2 + self.KY**2

        # We also construct a linearized list of coordinates.
        # The order matches self.X.ravel() and self.Y.ravel()
        self.coordinates = [tuple((x,y)) for x,y in zip(self.X.flat, self.Y.flat)]
        self.coordinate_lookup= {tuple(x):i for i,x in enumerate(self.coordinates)}

    def __str__(self):
        return f'Lattice({self.nx},{self.ny})'

    def __repr__(self):
        return str(self)

    def mod_x(self, x):
        # Mods into the x coordinate.
        # Assumes periodic boundary conditions.
        mod = np.mod(x, self.nx)
        if type(mod) is np.ndarray:
            return np.where(mod < 1+self.nx // 2, mod, mod - self.nx)
        return mod if mod < 1+self.nx // 2 else mod - self.nx

    def mod_y(self, y):
        # Mods into the y coordinate.
        # Assumes periodic boundary conditions.
        mod = np.mod(y, self.ny)
        if type(mod) is np.ndarray:
            return np.where(mod < 1+self.ny // 2, mod, mod - self.ny)
        return mod if mod < 1+self.ny // 2 else mod - self.ny

    def mod(self, x):
        # Mod an [x,y] pair into lattice coordinates.
        # Assumes periodic boundary conditions
        mod = np.mod(x, self.dims)
        return np.where(mod < 1+self.dims // 2, mod, mod - self.dims)

    def distance_squared(self, a, b):
        d = self.mod(np.array(a)-np.array(b))
        return np.dot(d,d)

    def tensor(self, n=2, dtype=complex):
        # can do matrix-vector via
        #   np.einsum('ijab,ab',matrix,vector)
        # to get a new vector (with indices ij).
        return np.zeros(np.tile(self.dims, n), dtype=dtype)

    def tensor_linearized(self, tensor):
        repeats = len(tensor.shape)
        return tensor.reshape(*np.tile(self.sites, int(repeats / len(self.dims))))

    def linearized_tensor(self, linearized):
        return linearized.reshape(*np.tile(self.dims, len(linearized.shape)))

    def vector(self, dtype=complex):
        return self.tensor(1, dtype)

    def matrix(self, dtype=complex):
        return self.tensor_linearized(self.tensor(2, dtype))

    def fft(self, vector, axes=(-2,-1), norm='ortho'):
        return np.fft.fft2(vector, axes=axes, norm=norm)

    def ifft(self, vector, axes=(-2,-1), norm='ortho'):
        return np.fft.ifft2(vector, axes=axes, norm=norm)

    @cached_property
    def adjacency_tensor(self):
        # Creates an (nx, ny, nx, ny) adjacency matrix, where the
        # first two indices form the "row" and the
        # second two indices form the "column"
        A = self.tensor(2, dtype=np.int64)
        for i,x in enumerate(self.x):
            for k,z in enumerate(self.x):
                if np.abs(self.mod_x(x-z)) not in [0,1]:
                    continue
                for j,y in enumerate(self.y):
                    for l,w in enumerate(self.y):
                        if np.abs(self.mod_y(y-w)) not in [0,1]:
                            continue
                        if self.distance_squared([x,y],[z,w]) == 1:
                            A[i,j,k,l] = 1
                            
        return A

    @cached_property
    def adjacency_matrix(self):
        return self.tensor_linearized(self.adjacency_tensor)
