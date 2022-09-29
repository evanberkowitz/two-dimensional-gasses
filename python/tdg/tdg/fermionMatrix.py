#!/usr/bin/env python3

from functools import cached_property
import numpy as np
import torch
import tdg

class FermionMatrix:

    def __init__(self, spacetime, beta, mu=0, h=torch.tensor([0,0,0])):
        self.Spacetime = spacetime

        self.beta = beta
        self.dt = beta / self.Spacetime.nt

        self.mu = mu
        self.h  = h
        self.absh = torch.sqrt(torch.dot(self.h, self.h))

        self.z = torch.exp(self.beta * self.mu)

        if self.absh == 0:
            self.exp_h = tdg.PauliMatrix[0]
        else:
            self.exp_h = np.cosh( self.absh * self.dt ) * tdg.PauliMatrix[0]
            for h, sigma in zip(self.h, tdg.PauliMatrix[1:]):
                self.exp_h += np.sinh( self.absh * self.dt ) * h / self.absh * sigma

        self.B = torch.matrix_exp( self.dt * self.Spacetime.Lattice.kappa)

    def __str__(self):
        return f"d(β={self.beta}, µ={self.mu}, h={self.h}, {self.Spacetime})"

    def __repr__(self):
        return str(self)

    def __self__(self, A):
        # Should do mat-vec dA
        # TODO: implement
        return A

    def logdet(self, A):
        # TODO: implement a proper evaluation of logdet(d)
        # This will require computing the sausage.
        return 0

