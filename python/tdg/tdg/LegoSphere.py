#!/usr/bin/env python3

import torch

class LegoSphere:

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
        S = Lattice.tensor(2)

        for i,x in enumerate(Lattice.x):
            for j,y in enumerate(Lattice.y):
                for k,z in enumerate(Lattice.x):
                    for l,w in enumerate(Lattice.y):
                        for p in self.points:
                            if Lattice.distance_squared(torch.tensor([x-z,y-w]), p) == 0:
                                S[i,j,k,l] += self.c * self.norm

        return S

    def __mul__(self, c):
        return LegoSphere(self.r, self.c * c)

    __rmul__ = __mul__
