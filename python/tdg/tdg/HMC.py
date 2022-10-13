#!/usr/bin/env python3

import torch

r'''

HMC is an importance-sampling algorithm. 

'''

class Hamiltonian:
    r"""The HMC *Hamiltonian* for a given action :math:`S` is
    
    .. math::
        \mathcal{H} = \frac{1}{2} p M^{-1} p + S(x)
    
    which has the standard non-relativistic kinetic energy and a potential energy given by the action.
    
    An HMC Hamiltonian serves two purposes:
    
    * The first is to draw momenta and evaluate the starting energy, and to do accept/reject according to the final energy.
      For this purpose Hamiltonians are callable.
    * The second is to help make a proposal to consider in the first place.  To do this we start with a position and momentum pair and
      integrate Hamilton's equations of motion,

    .. math::
            \begin{aligned}
                \frac{dx}{d\tau} &= + \frac{\partial \mathcal{H}}{\partial p}
                &
                \frac{dp}{d\tau} &= - \frac{\partial \mathcal{H}}{\partial x}
            \end{aligned}

    Of course, these two applications are closely related, and in 'normal' circumstances we use the same Hamiltonian for both purposes.
    """

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
        r"""
        The velocity is needed to update the positions in Hamilton's equations of motions.

        .. math::
            \texttt{velocity(p)} = \left.\frac{\partial \mathcal{H}}{\partial p}\right|_p
        """
        return self._xdot(p)

    def force(self, x):
        r"""
        The force is needed to update the momenta in Hamilton's equations of motions.

        .. math::
            \texttt{force(x)} = \left.-\frac{\partial \mathcal{H}}{\partial x}\right|_x
        """
        grad, = torch.autograd.grad(self.V(x), x)
        return -grad

class MarkovChain:
    r"""
    The HMC algorithm for updating an initial configuration :math:`x_i` goes as follows:

    #.  Draw a sample momentum :math:`p_i` from the gaussian distribution given by the kinetic piece of the Hamiltonian.
    #.  Calculate :math:`\mathcal{H}` for the given :math:`x_i` and drawn :math:`p_i`.
    #.  Integrate Hamilton's equations of motion to generate a proposal for a new :math:`x_f` and :math:`p_f`.
    #.  Accept the proposal according to :math:`\exp[-(\Delta\mathcal{H} = \mathcal{H}_f - \mathcal{H}_i)]`
    """
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
        r"""

        Parameters
        ----------
            x:  torch.tensor
                a configuration compatible with the Hamiltonian and integrator.

        Returns
        -------
            torch.tensor:
                a similar configuration; a new configuration if the proposal was accepted, or the original if the proposal is rejected.
                
        """
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
    r"""The LeapFrog integrator integrates Hamilton's equations of motion for a total of :math:`\tau` molecular dynamics time `md_time` in a reversible, symplectic way.
    
    It discretizes :math:`\tau` into `md_steps` steps of :math:`d\tau` and then uses simple finite-differencing.

    One step of :math:`d\tau` integration is accomplished by

    #. updating the momentum by :math:`d\tau/2`
    #. updating the position by :math:`d\tau`
    #. updating the momentum by :math:`d\tau/2`.

    However, if the number of steps is more than 1 the trailing half-step momentum update from one step is combined with the leading half-step momentum update from the next.
    """
 
    def __init__(self, H, md_steps, md_time=1):
        self.H = H
        self.md_time  = md_time
        self.md_steps = md_steps
        self.md_dt    = self.md_time / self.md_steps
        
    def integrate(self, x_i, p_i):
        r"""Integrate an initial position and momentum.

        Parameters
        ----------
            x_i:    torch.tensor
                    a tensor of positions
            p_i:    torch.tensor
                    a tensor of momenta

        Returns
        -------
            x_f:    torch.tensor
                    a tensor of positions,
            p_f:    torch.tensor
                    a tensor of momenta

        """
        
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
