#!/usr/bin/env python

from functools import cached_property, lru_cache as cached
import torch

class D4A1:

    r'''The D4 symmetry has an A1 representation that's "like the S-wave" in that it's rotationally symmetric.
    
    For functions of a relative coordinate (or of a total momentum), we can take a straight average over the whole group orbit of every point on the lattice (or Brillouin zone).

    Or, if we need, we can use only the subgroup that holds one or many vectors fixed.

    Parameters
    ----------
        Lattice: tdg.Lattice
            Used to construct the permutations that represent the group elements.
        fixed: iterable of torch.Tensor
            These are held fixed.

    .. plot:: examples/plot/symmetry_d4a1.py
       :include-source:


    '''

    def __init__(self,
                 Lattice,
                 fixed={}
                ):

        self.Lattice = Lattice
        self.fixed   = fixed

        self.operations = torch.tensor((
            ((+1,0),(0,+1)), # identity
            ((0,-1),(+1,0)), # rotate(π/2)
            ((-1,0),(0,-1)), # rotate(π) = inversion
            ((0,+1),(-1,0)), # rotate(3π/2)
            ((+1,0),(0,-1)), # reflect across x-axis
            ((0,-1),(-1,0)), # reflect across y=-x
            ((-1,0),(0,+1)), # reflect across y-axis
            ((0,+1),(+1,0)), # reflect across y=+x
        ))

        for f in fixed:
            modded = self.Lattice.mod(f)
            self.operations = self.operations[(torch.matmul(self.operations, modded) == modded).all(axis=1)]


    @cached
    def _permutation(self, op):
        # Since the operations map lattice points to lattice points we know that they are a permutation
        # on the set of coordinates.
        permutation = []
        L = self.Lattice
        for i in range(L.sites):
            for j in range(L.sites):
                if (L.coordinates[i] == torch.matmul(op, L.coordinates[j])).all():
                    permutation += [j]
                    continue # since a permutation is one-to-one
        return torch.tensor(permutation)

    def __len__(self):
        return self.operations.shape[0]

    def __call__(self, data, axis=(-1,)):
        r'''

        Parameters
        ----------
            data: torch.tensor
                Data whose `axis` should be symmetrized.
            axis: iterable of ints
                Axis or axes on which to act.  Those axes should be linearized spatial indices.

        Returns
        -------
            A torch.tensor of the same shape as data, but with the axis symmetrized.
        '''
        shape = data.shape
        dims  = len(shape)

        temp = torch.zeros_like(data)

        for o in self.operations:
            p = self._permutation(o)
            for i in axis:
                temp += torch.index_select(data, i, p)

        temp /= len(self.operations)**len(axis)
        return temp
