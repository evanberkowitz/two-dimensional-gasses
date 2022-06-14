#!/usr/bin/env python

import numpy as np

class Lattice:

    def __init__(self, nx, ny=None):
        self.nx = nx

        if ny is None:
            self.ny = nx
        else:
            self.ny = ny

        self.dims = np.array([self.nx, self.ny])

        # We want to go from -n/2 to +n/2
        self.x = np.arange( - (self.nx // 2), self.nx // 2 + 1)
        self.y = np.arange( - (self.ny // 2), self.ny // 2 + 1)
        # (up to even/odd issues)
        if self.nx % 2 == 0:
            self.x = self.x[1:]
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

        # We also construct a linearized list of coordinates.
        # The order matches self.X.ravel() and self.Y.ravel()
        self.coordinates = np.array([[x,y] for x,y in zip(self.X.flat, self.Y.flat)])

    def __str__(self):
        return f'Lattice({self.nx},{self.ny})'

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

    def vector(self, dtype=complex):
        return np.zeros([self.nx, self.ny], dtype=dtype)

    def matrix(self, dtype=complex):
        return np.zeros([self.nx, self.ny, self.nx, self.ny], dtype=dtype)

    def fft(self, vector, axes=(-2,-1), norm='ortho'):
        return np.fft.fft2(vector, axes=axes, norm=norm)

    def ifft(self, vector, axes=(-2,-1), norm='ortho'):
        return np.fft.ifft2(vector, axes=axes, norm=norm)

    # Private, call-once functions to set some properties in the constructor
    def _dx(self):
        pass

