#!/usr/bin/env python3

import torch

class Hamiltonian:

    def __init__(self, S, massMatrix=None):
        # massMatrix must be symmmetric.

        self.V = S

        # The kinetic piece is p^2/2m
        # but by default set m = 1 for each mode, in which case we get simplifications:
        if massMatrix is None:
            self.M = torch.eye(S.Spacetime.sites)
            self.Minv = torch.eye(S.Spacetime.sites)
            self.T = lambda p: torch.sum(p*p)/2
            self._xdot = lambda p: p
        else:
            self.M = massMatrix.clone()
            self.Minv = torch.linalg.matrix_power(self.M)
            self.T = lambda p: torch.einsum('i,ij,j', p, Minv, p)/2
            self._xdot = lambda p: torch.matmul(Minv, p) # relies on the symmetry of the massMatrix

    def __call__(self, x, p):
        return self.T(p) + self.V(x)

    def velocity(self, p):
        return self._xdot(p)

    def force(self, x):
        grad, = torch.autograd.grad(self.V(x), x)
        return -grad

class MarkovChain:
    
    def __init__(self, H, integrator):
        """
        H is used for accept-reject
        integrator is used for the molecular dynamics and need not use the same H.
        """
        self.H = H
        
        self.refresh_momentum = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(H.V.Spacetime.sites), 
            self.H.M
        ).sample
        
        self.integrator = integrator
        
        # And we do accept/reject sampling
        self.metropolis_hastings = torch.distributions.uniform.Uniform(0,1).sample
        
        
    def step(self, x):
        
        p_i = self.refresh_momentum().reshape(self.H.V.Spacetime.dims).requires_grad_(True)
        x_i = x.clone().requires_grad_(True)
        
        H_i = self.H(x_i,p_i)
        
        x_f, p_f = self.integrator(x_i, p_i)
        
        H_f = self.H(x_f,p_f)
        dH = H_f - H_i

        if torch.exp(-dH.real) > self.metropolis_hastings():
            return x_f
        else:
            return x_i

class LeapFrog:
    
    def __init__(self, H, md_steps, md_time=1):
        self.H = H
        self.md_time  = md_time
        self.md_steps = md_steps
        self.md_dt    = self.md_time / self.md_steps
        
    def integrate(self, x_i, p_i):
        
        # Take an initial half-step of the momentum
        p = p_i + self.H.force(x_i) * self.md_dt / 2
        
        # do the initial position update,
        x = (x_i + self.H.velocity(p) * self.md_dt).clone()
        
        # Now do whole-dt momentum AND position updates
        for md_step in range(self.md_steps-1):
            p = p + self.H.force(x) * self.md_dt
            x = (x + self.H.velocity(p) * self.md_dt).clone()
            
        # Take a final half-step of momentum
        p_f = p + self.H.force(x) * self.md_dt / 2

        return x, p_f
    
    def __call__(self, x_i, p_i):
        return self.integrate(x_i, p_i)
