#!/usr/bin/env python3

from functools import cached_property
from functools import lru_cache as cached
import numpy as np
import torch
from tdg.h5 import H5able

class Potential(H5able):
    r'''
    A potential encodes a term in the many-body Hamiltonian like :math:`nVn`, where :math:`n` are number operators and :math:`V` can connect
    different sites.

    A potential is built up from one or more LegoSpheres :math:`\mathcal{S}^{\vec{R}}`, each which carries its own Wilson coefficient :math:`C_\vec{R}`.
    '''

    def __init__(self, *spheres):
        self.spheres = [*spheres]

    def __str__(self):
        return f"Potential({', '.join([str(s) for s in self.spheres])})"

    def __repr__(self):
        return str(self)

    @cached
    def spatial(self, lattice):
        r'''
        Parameters
        ----------
            lattice:    tdg.Lattice

        Returns
        -------
            torch.tensor:
                A matrix encoding :math:`V` on the given lattice.  The two axes of the matrix are in the order of `lattice.coordinates`.
        '''
        return torch.sum(torch.stack([s.spatial(lattice) for s in self.spheres]), axis=0)

    @cached
    def inverse(self, lattice):
        r'''
        Parameters
        ----------
            lattice:    tdg.Lattice

        Returns
        -------
            torch.tensor:
                The inverse matrix of ``spatial(spatial)``.
        '''
        return torch.linalg.inv(self.spatial(lattice))

    @cached
    def eigvals(self, lattice):
        r'''
        Parameters
        ----------
            lattice:    tdg.Lattice

        Returns
        -------
            torch.tensor:
                An array of eigenvalues of ``spatial(lattice)``.  If any of the eigenvalues are imaginary, raises a TypeError; the potential should be Hermitian.
                Raises a ValueError if any of the eigenvalues are positive; we require the attractive channel.
        '''
        e = torch.linalg.eigvals(self.spatial(lattice))
        if (e.imag != 0).all():
            raise TypeError(f"{self} yields imaginary eigenvalues.")
        if (e.real >= 0).any():
            raise ValueError(f"{self} yields positive eigenvalues; we require the attractive channel.")
        #return e.real ?
        return e

    @cached_property
    def C0(self):
        r'''
        The Wilson coefficient of the :math:`\mathcal{S}^0` piece of the potential.
        '''
        c0 = 0
        for s in self.spheres:
            if (0 == s.r).all() :
                c0 += s.c
        return c0
