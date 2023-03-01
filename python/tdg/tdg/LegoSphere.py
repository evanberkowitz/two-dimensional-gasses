#!/usr/bin/env python3

import torch
from tdg.h5 import H5able

class LegoSphere(H5able):
    r'''
    Translationally-invariant potentials in the :math:`A_1` representations of the lattice :math:`D_4` symmetry can be written as a sum of LegoSpheres,

    .. math::
        \begin{align}
            \tilde{V}_{a,a+r} &= \tilde{V}_{0,r}
        &   \tilde{V}_{0,r} &= \mathcal{V} \sum_R \tilde{C}_R \mathcal{S}^R_{0,r}
        \end{align}

    where each LegoSphere :math:`\mathcal{S}` has a radius :math:`R`.

    To be in the :math:`A_1` representation each LegoSphere is a uniformly-weighted stencil with no phases,

    .. math::
        \begin{align}
            \mathcal{S}^R_{0,r} &= \frac{1}{\mathcal{N}_R^2} \sum_{g\in D_4} \delta_{r,gR}
        &   \mathcal{S}^R_{ab}  &= \frac{1}{\mathcal{N}_R^2} \sum_{g\in D_4} \delta_{b-a,gR}
        \end{align}

    where :math:`g` acts to rotate spatial displacements on the lattice.

    Parameters
    ----------
        r:  list or tuple
            the radius given as a vector ``[x, y]``.
            The magnitude is insufficient information, because two radii with the same magnitude
            might yield different LegoSpheres (consider ``[3, 4]`` and ``[0, 5]``, for example).
        c:  float or a torch.tensor-wrapped number
            the Wilson coefficient :math:`\tilde{C}_R`.

    LegoSpheres may be multiplied by coefficients on either side to give new LegoSpheres with c
    changed appropriately,

    >>> sphere = LegoSphere([0,1], 1.23)
    >>> stronger = 2*sphere
    >>> stronger.c
    tensor(2.4600)
    '''


    def __init__(self, r, c=1):

        x = r[0]
        y = r[1]

        self.r = torch.tensor([x, y])

        # Precompute the D4-symmetric set of points.
        # Use the set built-in to eliminate duplicates.
        if x ==0 and y==0:
            self.points = torch.tensor([[0,0]])
        elif x == 0:
            self.points = torch.tensor([
                [+x,+y], [+x,-y],
                [+y,+x], [-y,+x]
                ])
        elif x == y:
            self.points = torch.tensor([
                [+x,+y], [+x,-y],
                [-x,+y], [-x,-y],
                ])
        else:
            self.points = torch.tensor([
                [+x,+y], [+x,-y],
                [-x,+y], [-x,-y],
                [+y,+x], [+y,-x],
                [-y,+x], [-y,-x],
                ])

        # The canonical normalization is 1/(the number of points)
        self.norm = 1./len(self.points)
        self.c = torch.tensor(1) * c

    def __str__(self):
        return f'LegoSphere({self.r}, {self.c})'

    def __repr__(self):
        return str(self)

    def spatial(self, Lattice):
        r'''

        Parameters
        ----------
            lattice: tdg.Lattice
                a spatial lattice on which to construct :math:`\tilde{C}_R \mathcal{S}^R_{ab}`

        Returns
        -------
            torch.tensor:
                a square matrix of dimension ``[lattice.sites, lattice.sites]`` where the first index is the superindex :math:`a`
                and the second index  is the superindex :math:`b`.
        '''
        S = torch.zeros(Lattice.sites, Lattice.sites)

        for i,x in enumerate(Lattice.coordinates):
            for j,y in enumerate(Lattice.coordinates):
                for p in self.points:
                    if Lattice.distance_squared(x-y, p) == 0:
                        S[i,j] += self.c * self.norm

        return S

    def __mul__(self, c):
        return LegoSphere(self.r, self.c * c)

    __rmul__ = __mul__

if __name__ == "__main__":
    import doctest
    doctest.testmod()
