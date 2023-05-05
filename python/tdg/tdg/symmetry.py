#!/usr/bin/env python

from functools import cached_property, lru_cache as cached
import torch

class D4:

    r'''The D4 symmetry has an A1 representation that's "like the S-wave" in that it's rotationally symmetric.
    
    For functions of a relative coordinate (or of a total momentum), we can take a straight average over the whole group orbit of every point on the lattice (or Brillouin zone).

    Parameters
    ----------
        Lattice: tdg.Lattice
            Used to construct the permutations that represent the group elements.

    .. plot:: examples/plot/D4_irreps.py
       :include-source:


    '''

    def __init__(self,
                 Lattice,
                ):

        self.Lattice = Lattice

        self.operations = torch.tensor((
            # Matches the order of the (a,b) orbit in docs/D4.rst
            # That makes it easy to read off the weights
            ((+1,0),(0,+1)), # identity
            ((0,+1),(+1,0)), # reflect across y=+x
            ((0,-1),(+1,0)), # rotate(π/2)
            ((-1,0),(0,+1)), # reflect across y-axis
            ((-1,0),(0,-1)), # rotate(π) = inversion
            ((0,-1),(-1,0)), # reflect across y=-x
            ((0,+1),(-1,0)), # rotate(3π/2)
            ((+1,0),(0,-1)), # reflect across x-axis
        ))

        one_over_two_sqrt_two = torch.tensor(1/8).sqrt()
        self.weights = {
            'A1': one_over_two_sqrt_two * torch.tensor((+1,+1,+1,+1,+1,+1,+1,+1)) + 0.j,
            'A2': one_over_two_sqrt_two * torch.tensor((+1,-1,+1,-1,+1,-1,+1,-1)) + 0.j,
            'B1': one_over_two_sqrt_two * torch.tensor((+1,-1,-1,+1,+1,-1,-1,+1)) + 0.j,
            'B2': one_over_two_sqrt_two * torch.tensor((+1,+1,-1,-1,+1,+1,-1,-1)) + 0.j,
            ("E", +1): one_over_two_sqrt_two * torch.tensor((+1,+1j,+1j,-1,-1,-1j,-1j,+1)),
            ("E", -1): one_over_two_sqrt_two * torch.tensor((+1,-1j,-1j,-1,-1,+1j,+1j,+1)),
            ("E'", +1): one_over_two_sqrt_two * torch.tensor((+1,-1j,+1j,+1,-1,+1j,-1j,-1)),
            ("E'", -1): one_over_two_sqrt_two * torch.tensor((+1,+1j,-1j,+1,-1,-1j,+1j,-1)),
        }

        self.irreps = self.weights.keys()

        self.permutations = torch.stack(tuple(self._permutation(o) for o in self.operations))

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

    def __call__(self, data, irrep='A1', conjugate=False, axis=-1):
        r'''

        Parameters
        ----------
            data: torch.tensor
                Data whose `axis` should be symmetrized.
            irrep: one of `.irreps`
                The irrep to project to.
            conjugate: `True` or `False`
                The weights are conjugated, which only affects the E representations.
            axis: int
                The linearized index on which to act.

        Returns
        -------
            A complex-valued torch.tensor of the same shape as data, but with the axis projected to the requested irrep.
        '''
        shape = data.shape
        dims  = len(shape)

        temp = torch.zeros_like(data) + 0.j

        for p, w in zip(
                self.permutations,
                self.weights[irrep] if not conjugate else self.weights[irrep].conj()
                ):
            temp += w * torch.index_select(data, axis, p)

        return temp
