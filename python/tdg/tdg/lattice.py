#!/usr/bin/env python

from functools import cached_property
import numpy as np
import torch

class Lattice:

    def __init__(self, nx, ny=None):
        self.nx = nx

        if ny is None:
            self.ny = nx
        else:
            self.ny = ny

        if(self.nx != self.ny):
            # For the time being I will restrict to nx = ny.
            # The reason is that when nx ≠ ny more care about the kinetic matrix κ is required.
            # The main issue is that when nx ≠ ny, self.nx**2 ≠ self.sites and so the different
            # momentum components need different normalizations.
            # When nx == ny this is coincidentally correct.
            raise ValueError("Anisotropic lattices not currently supported")

        self.dims = torch.tensor([self.nx, self.ny])
        self.sites = self.nx * self.ny

        # We want to go from -n/2 to +n/2
        self.x = torch.arange( - (self.nx // 2), self.nx // 2 + 1)
        self.y = torch.arange( - (self.ny // 2), self.ny // 2 + 1)
        # (up to even/odd issues)
        if self.nx % 2 == 0:
            self.x = self.x[1:]
        if self.ny % 2 == 0:
            self.y = self.y[1:]

        # However, we want to match the FFT convention of numpy
        # https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details
        # where the lowest coordinate / frequency is at 0, we increase to the max,
        # and then go in decreasingly-negative order.
        self.x = torch.roll(self.x, -((self.nx-1) // 2))
        self.y = torch.roll(self.y, -((self.ny-1) // 2))

        # These are chosen so that Lattice(nx, ny)
        # has coordinate matrices of size (nx, ny)
        self.X = torch.tile( self.x, (self.ny, 1)).T
        self.Y = torch.tile( self.y, (self.nx, 1))

        # Wavenumbers are the same
        self.kx = self.x
        self.ky = self.y
        self.KX = self.X
        self.KY = self.Y
        # To get the dimensionless wavenumber^2 in the fourier basis we can simply
        self.ksq = self.KX**2 + self.KY**2

        # We also construct a linearized list of coordinates.
        # The order matches self.X.ravel() and self.Y.ravel()
        self.coordinates = torch.stack((self.X.flatten(), self.Y.flatten())).T
        self.coordinate_lookup= {tuple(x):i for i,x in enumerate(self.coordinates)}

    def __str__(self):
        return f'SquareLattice({self.nx},{self.ny})'

    def __repr__(self):
        return str(self)

    def mod_x(self, x):
        # Mods into the x coordinate.
        # Assumes periodic boundary conditions.
        mod = torch.remainder(x, self.nx)
        return torch.where(mod < 1+self.nx // 2, mod, mod - self.nx)

    def mod_y(self, y):
        # Mods into the y coordinate.
        # Assumes periodic boundary conditions.
        mod = torch.remainder(y, self.ny)
        return torch.where(mod < 1+self.ny // 2, mod, mod - self.ny)

    def mod(self, x):
        # Mod an [x,y] pair into lattice coordinates.
        # Assumes periodic boundary conditions
        if len(x.shape) == 1:
            X = x[None,:]
        else:
            X = x
        modx = self.mod_x(X[:,0])
        mody = self.mod_y(X[:,1])

        mod = torch.stack((modx, mody)).T

        if len(x.shape) == 1:
            return mod[0]
        else:
            return mod

    def distance_squared(self, a, b):
        d = self.mod(torch.tensor(a)-torch.tensor(b))
        return torch.dot(d,d)

    def tensor(self, n=2):
        # can do matrix-vector via
        #   torch.einsum('ijab,ab',matrix,vector)
        # to get a new vector (with indices ij).
        return torch.zeros(np.tile(self.dims, n).tolist())

    def tensor_linearized(self, tensor):
        repeats = len(tensor.shape)
        return tensor.reshape(*np.tile(self.sites, int(repeats / len(self.dims))))

    def linearized_tensor(self, linearized):
        return linearized.reshape(*np.tile(self.dims, len(linearized.shape)))

    def vector(self):
        return self.tensor(1)

    def matrix(self):
        return self.tensor_linearized(self.tensor(2))

    def fft(self, vector, axes=(-2,-1), norm='ortho'):
        return torch.fft.fft2(vector, dim=axes, norm=norm)

    def ifft(self, vector, axes=(-2,-1), norm='ortho'):
        return torch.fft.ifft2(vector, dim=axes, norm=norm)

    @cached_property
    def adjacency_tensor(self):
        # Creates an (nx, ny, nx, ny) adjacency matrix, where the
        # first two indices form the "row" and the
        # second two indices form the "column"
        A = torch.zeros(np.tile(self.dims, 2).tolist(), dtype=torch.int)
        for i,x in enumerate(self.x):
            for k,z in enumerate(self.x):
                a = torch.abs(self.mod_x(x-z))
                if a not in [0,1]:
                    continue
                for j,y in enumerate(self.y):
                    for l,w in enumerate(self.y):
                        b = torch.abs(self.mod_y(y-w))
                        #if self.distance_squared( [x,y], [z,w] ) == 1: # nearest-neighbors
                        if a+b == 1:
                            A[i,j,k,l] = 1
                            
        return A

    @cached_property
    def adjacency_matrix(self):
        return self.tensor_linearized(self.adjacency_tensor)

    @cached_property
    def kappa(self):
        # Makes an assumption that nx = ny

        # We know kappa_ab = 1/V Σ(k) (2πk)^2/2V exp( -2πik•(a-b)/Nx )
        # where a are spatial coordinates
        a = self.coordinates.clone().detach().to(torch.complex128)
        # and k are also integer doublets;
        # in the exponent we group the -2πi / Nx into the momenta
        p = (-2*torch.pi*1j / self.nx) * a
        # Separating out the unitary U_ak = 1/√V exp( - 2πik•a )
        U = torch.exp(torch.einsum('ax,kx->ak', a, p)) / np.sqrt(self.sites)

        # we can write isolate the eigenvalues
        #   kappa_kq = δ_kq ( 2π k / Nx )^2 / 2
        eigenvalues = ((2*torch.pi)**2 / self.sites) * self.ksq.flatten() / 2

        # via the unitary transformation
        # kappa_ab = Σ(kq) U_ak kappa_kq U*_qb
        #          = Σ(k)  U_ak [(2πk/Nx)^2/2] U_kb
        return torch.einsum('ak,k,kb->ab', U, eigenvalues, U.conj().transpose(0,1))
        #
        # Can be checked by eg.
        # ```
        # lattice = tdg.Lattice(11)
        # left = torch.linalg.eigvalsh(lattice.kappa).sort().values 
        # right = torch.tensor(4*np.pi**2 * lattice.ksq.ravel() / lattice.sites / 2).sort().values
        # (torch.abs(left-right) < 1e-6).all()
        # ```
