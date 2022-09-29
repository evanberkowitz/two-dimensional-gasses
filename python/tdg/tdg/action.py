#!/usr/bin/env python3

from functools import cached_property
import numpy as np
import torch

from tdg.fermionMatrix import FermionMatrix


class Action:

    def __init__(self, spacetime, potential, beta, mu=0, h=[0,0,0], fermion=FermionMatrix):
        self.Spacetime = spacetime
        self.Potential = potential

        self.V = self.Potential.spatial(self.Spacetime.Lattice)
        self.Vinverse = self.Potential.inverse(self.Spacetime.Lattice)

        self.beta = beta
        self.dt = beta / self.Spacetime.nt

        self.mu = mu
        self.h  = torch.tensor(h)

        # Recall that requiring the contact interaction
        # be written as the quadratic nVn induces a term in the Hamiltonian
        # proportional to n itself; a chemical potential equal to - C0/2.
        #
        # Since we work with H-µN-hS this ADDS to the physical chemical potential.
        self.FermionMatrix = fermion(self.Spacetime, self.beta, mu=self.mu + potential.C0/2, h=self.h)

    def __str__(self):
        return f"Action(β={self.beta}, µ={self.mu}, h={self.h}, {self.Spacetime}, {self.Potential})"

    def __repr__(self):
        return str(self)

    def __call__(self, A):
        # compute the action here
        gauss = -0.5 * torch.einsum('txy,xyab,tab->', A, -self.Vinverse, A) / self.dt

        fermionic = - self.FermionMatrix.logdet(A)

        return gauss + fermionic

