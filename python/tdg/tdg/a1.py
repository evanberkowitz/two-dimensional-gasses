#!/usr/bin/env python3

from functools import cached_property
import torch

class ReducedTwoBodyA1Hamiltonian:

    def __init__(self, lattice, legoSpheres):
        self.Lattice = lattice
        self.LegoSpheres = legoSpheres
        self.spheres = len(self.LegoSpheres)

        self.spherical_operators = []

        norms = 1/(torch.sqrt(torch.einsum('i,j->ij',self.shellSizes, self.shellSizes))*self.Lattice.nx**2)
        for sphere in self.LegoSpheres:
            expdot = [torch.exp(2j*torch.pi/self.Lattice.nx * torch.einsum('sx,x->s',shell,sphere.r)) for shell in self.shells]
            operator = torch.tensor([[torch.sum(torch.outer(m,torch.conj(n))) for n in expdot] for m in expdot])*norms
            self.spherical_operators+=[operator]

    @cached_property
    def states(self):
        return self.Lattice.coordinates[
            torch.where(
                (0 <= self.Lattice.coordinates[:, 0]) &
                (self.Lattice.coordinates[:,0] <= self.Lattice.coordinates[:,1])
            )
        ]

    @cached_property
    def shells(self):
        shells = []
        for state in self.states:
            if (state == torch.tensor([0,0])).all():
                shell = torch.tensor([[0,0]])
            elif state[0] == 0:
                shell = torch.tensor([
                    [+state[0],+state[1]], [+state[0],-state[1]],
                    [+state[1],+state[0]], [-state[1],+state[0]]
                ])
            elif state[0] == state[1]:
                shell = torch.tensor([
                    [+state[0],+state[1]], [+state[0],-state[1]],
                    [-state[0],+state[1]], [-state[0],-state[1]],
                ])
            else:
                shell = torch.tensor([
                    [+state[0],+state[1]], [+state[0],-state[1]],
                    [-state[0],+state[1]], [-state[0],-state[1]],
                    [+state[1],+state[0]], [+state[1],-state[0]],
                    [-state[1],+state[0]], [-state[1],-state[0]],
                ])
            shells += [shell]

        return shells

    @cached_property
    def shellSizes(self):
        boundary = torch.floor(torch.tensor(self.Lattice.nx +1)/2).to(torch.int)
        shells = []
        for state in self.states:
            if   (state == torch.tensor([0,0])).all():
                count = 1
            elif state[0] == 0:
                count = 4
            elif state[0] == state[1]:
                count = 4
            else:
                count = 8
            shells += [(count  / 2**torch.count_nonzero(state == boundary)).to(torch.int) ]
        return torch.tensor(shells)

    @cached_property
    def kinetic(self):
        # A diagonal matrix with entries = 2 (reduced mass) * 1/2 * (2π/nx)^2 * n^2
        # Don't bother computing 2 * 1/2 = 1.
        return torch.diag((2*torch.pi/self.Lattice.nx)**2 * torch.einsum('np,np->n', self.states, self.states))

    def potential(self, C):
        V = torch.zeros_like(self.spherical_operators[0])
        for c, o in zip(C, self.spherical_operators):
            V += c * o
        return V

    def operator(self, C):
        return self.kinetic + self.potential(C)

    def eigenenergies(self, C):
        return torch.linalg.eigvalsh(self.operator(C))

    def tuning(self, target_energies, start=None, epochs=10000, lr=0.05):
        loss = torch.nn.MSELoss()
        coefficients = torch.ones_like(target_energies, requires_grad=True) if start is None else start
        assert coefficients.shape == target_energies.shape

        optimizer = torch.optim.AdamW(
            [coefficients],
            lr = lr,
        )

        for epoch in range(epochs):
            optimizer.zero_grad()
            energies = self.eigenenergies(coefficients)[:len(coefficients)]
            loss(energies-target_energies, torch.zeros_like(target_energies)).backward()
            optimizer.step()

        return coefficients
