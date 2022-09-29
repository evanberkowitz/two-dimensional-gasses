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
        return f'SquareLattice({self.nx},{self.ny})'

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

    def tensor(self, n=2):
        # can do matrix-vector via
        #   np.einsum('ijab,ab',matrix,vector)
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
        A = np.zeros(np.tile(self.dims, 2).tolist())
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

    @cached_property
    def kappa(self):
        # Makes an assumption that nx = ny

        # We know kappa_ab = 1/V Σ(k) (2πk)^2/2V exp( -2πik•(a-b)/Nx )
        # where a are spatial coordinates
        a = torch.tensor(self.coordinates).to(torch.complex128)
        # and k are also integer doublets;
        # in the exponent we group the -2πi / Nx into the momenta
        p = (-2*np.pi*1j / self.nx) * torch.tensor(self.coordinates).to(torch.complex128)
        # Separating out the unitary U_ak = 1/√V exp( - 2πik•a )
        U = torch.exp(torch.einsum('ax,kx->ak', a, p)) / np.sqrt(self.sites)

        # we can write isolate the eigenvalues
        #   kappa_kq = δ_kq ( 2π k / Nx )^2 / 2
        eigenvalues = ((2*np.pi)**2 / self.sites) * torch.tensor(self.ksq.ravel()) / 2

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
