#!/usr/bin/env python3

import torch
import tdg

class FermionMatrix:

    def __init__(self, spacetime, beta, mu=torch.tensor(0, dtype=torch.float), h=torch.tensor([0,0,0], dtype=torch.float)):
        self.Spacetime = spacetime

        self.beta = beta
        self.dt = beta / self.Spacetime.nt

        self.mu = mu
        self.h  = h
        self.absh = torch.linalg.vector_norm(self.h)

        self.z = torch.exp(self.beta * self.mu)
        self.zh = torch.exp(self.beta * self.absh)

        if self.absh == 0:
            self.exp_h_dt = tdg.PauliMatrix[0]
        else:
            self.exp_h_dt = torch.cosh( self.absh * self.dt ) * tdg.PauliMatrix[0]
            for h, sigma in zip(self.h, tdg.PauliMatrix[1:]):
                self.exp_h_dt += torch.sinh( self.absh * self.dt) * h / self.absh * sigma

        self.B = torch.matrix_exp( self.dt * self.Spacetime.Lattice.kappa)
        self.Binverse = torch.matrix_exp( -self.dt * self.Spacetime.Lattice.kappa)

    def __str__(self):
        return f"d(β={self.beta}, µ={self.mu}, h={self.h}, {self.Spacetime})"

    def __repr__(self):
        return str(self)

    def __self__(self, A):
        # Should do mat-vec dA
        # TODO: implement
        return A

    def U(self, A):
        # Computes Binv F(Nt) Binv F(Nt-1) ... Binv F(2) Binv(1)
        # where F = exp(A), excluding the µ and h terms.
        #
        # For numerical stability we may need a smarter method.
        # This naive method is at least in principle correct.

        assert (A.shape == self.Spacetime.dims), f"Gauge field shape {A.shape} must match the spacetime dimensions {self.Spacetime.dims}"

        # Rather than incorporate µ ∆t into the exponential, since it is spacetime-constant,
        # we can just pull alll nt terms out and multiply by z at the end.

        # First construct all BinvF(t) for each t

        F = torch.exp(A.reshape(self.Spacetime.nt, self.Spacetime.Lattice.sites))
        # Since F(t) is a diagonal matrix we don't need to expand it and do 'real'
        # matrix multiplication.  Just use the fast simplification for each timeslice instead.
        BinvF = torch.einsum('ij,tj->tij',self.Binverse, F)

        # Then multiply them togther
        U = torch.eye(self.Spacetime.Lattice.sites, dtype=torch.complex128)
        for t in self.Spacetime.t:
            U = torch.matmul(BinvF[t], U)
        return U

    def logdet(self, A):
        one = torch.eye(self.Spacetime.Lattice.sites).to(torch.complex128)
        zU = self.z * self.U(A)
        return torch.log(torch.det(one + zU*self.zh)) + torch.log(torch.det(one + zU/self.zh))

