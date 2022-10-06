#!/usr/bin/env python3

import torch

from tdg.fermionMatrix import FermionMatrix


class Action:

    def __init__(self, spacetime, potential, beta, mu=torch.tensor(0, dtype=torch.float), h=torch.tensor([0,0,0], dtype=torch.float), fermion=FermionMatrix):
        self.Spacetime = spacetime
        self.Potential = potential

        self.beta = beta
        self.dt = beta / self.Spacetime.nt

        self.mu = mu
        self.h  = h

        self.V = self.Potential.spatial(self.Spacetime.Lattice)
        self.Vinverse = self.Potential.inverse(self.Spacetime.Lattice)

        # Recall that requiring the contact interaction
        # be written as the quadratic nVn induces a term in the Hamiltonian
        # proportional to n itself; a chemical potential equal to - C0/2.
        #
        # Since we work with H-µN-hS this ADDS to the physical chemical potential.
        self.FermionMatrix = fermion(self.Spacetime, self.beta, mu=self.mu + potential.C0/2, h=self.h)

        self.normalizing_offset = self.Spacetime.nt / 2 * torch.sum( torch.log(-2*torch.pi*self.dt * self.Potential.eigvals(self.Spacetime.Lattice)))

    def __str__(self):
        return f"Action(β={self.beta}, µ={self.mu}, h={self.h}, {self.Spacetime}, {self.Potential})"

    def __repr__(self):
        return str(self)

    def __call__(self, A):
        # S = 1/2 Σ(t) A(t) inverse(- ∆t V) A(t) - log det d + nt/2 tr log(-2π ∆t V )
        gauss = -0.5 * torch.einsum('txy,xyab,tab->', A, self.Vinverse.to(A.dtype), A) / self.dt

        fermionic = - self.FermionMatrix.logdet(A)

        return gauss + fermionic + self.normalizing_offset

