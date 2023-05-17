#!/usr/bin/env python3

import torch
from tdg.h5 import H5able
from collections import deque

import logging
logger = logging.getLogger(__name__)

r'''

HMC is an importance-sampling algorithm. 

'''

class Hamiltonian(H5able):
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
        r"""

        Parameters
        ----------
            x:  torch.tensor
                a configuration compatible with the Hamiltonian
            p:  torch.tensor
                a momentum of the same shape

        Returns
        -------
            torch.tensor:
                :math:`\mathcal{H}` for the given ``x`` and ``p``.
                
        """
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

class MarkovChain(H5able):
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
        
        self.steps = 0
        self.accepted = 0
        self.rejected = 0
        self.dH = deque()
        self.acceptance_probability = deque()
        
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
                
            jacobian:
                The Jacobian of the integration.  Currently hard-coded to 1, but may be needed for other integration schemes.
        """
        p_i = self.refresh_momentum().reshape(*x.shape).requires_grad_(True)
        x_i = x.clone().requires_grad_(True)
        
        H_i = self.H(x_i,p_i)
        
        x_f, p_f = self.integrator(x_i, p_i)
        
        H_f = self.H(x_f,p_f)
        dH = (H_f - H_i).detach()

        if dH.isnan():
            raise ValueError('HMC energy change is NaN.  {H_i=} {H_f=}')

        acceptance_probability = torch.exp(-dH.real).clamp(max=1)
        accept = (acceptance_probability > self.metropolis_hastings())

        self.dH.append(dH)
        self.acceptance_probability.append(acceptance_probability.clone().detach())

        logger.info(f'HMC proposal {"accepted" if accept else "rejected"} with dH={dH.real.cpu().detach().numpy():+} acceptance_probability={acceptance_probability.cpu().detach().numpy()}')

        self.steps += 1
        if accept:
            self.accepted += 1
            return x_f, 1.
        else:
            self.rejected += 1
            return x_i, 1.

class LeapFrog(H5able):
    r"""The LeapFrog integrator integrates Hamilton's equations of motion for a total of :math:`\tau` molecular dynamics time `md_time` in a reversible, symplectic way.
    
    It discretizes :math:`\tau` into `md_steps` steps of :math:`d\tau` and then uses simple finite-differencing.

    One step of :math:`d\tau` integration is accomplished by

    #. updating the coordinates by :math:`d\tau/2`
    #. updating the momenta by :math:`d\tau`
    #. updating the coordinates by :math:`d\tau/2`.

    However, if the number of steps is more than 1 the trailing half-step coordinate update from one step is combined with the leading half-step coordinate update from the next.
    """
 
    def __init__(self, H, md_steps, md_time=1):
        self.H = H
        self.md_time  = md_time
        self.md_steps = md_steps
        self.md_dt    = self.md_time / self.md_steps
        
    def __str__(self):
        return f'LeapFrog(H, md_steps={self.md_steps}, md_time={self.md_time})'

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
        
        # Take an initial half-step of the coordinates
        x = x_i + self.H.velocity(p_i) * self.md_dt / 2
        
        # do the initial momentum update,
        p = p_i + self.H.force(x) * self.md_dt
        
        # Now do whole-dt coordinate AND momentum updates
        for md_step in range(self.md_steps-1):
            x = x + self.H.velocity(p) * self.md_dt
            p = p + self.H.force(x) * self.md_dt
            
        # Take a final half-step of coordinates
        x = x + self.H.velocity(p) * self.md_dt / 2

        return x, p
    
    def __call__(self, x_i, p_i):
        '''
        Forwards to ``integrate``.
        '''
        return self.integrate(x_i, p_i)

class Omelyan(H5able):
    r"""
    The Omelyan integrator is a second-order integrator which integrates Hamilton's equations of motion for a total of :math:`\tau` molecular dynamics time `md_time` in a reversible, symplectic way.

    It discretizes :math:`\tau` into `md_steps` steps of :math:`d\tau` and given :math:`0\leq\zeta\leq 0.5` applies the following integration scheme:

    #. Update the coordinates by :math:`\zeta\;d\tau`,
    #. update the momenta by :math:`\frac{1}{2}\; d\tau`,
    #. update the coordinates by :math:`(1-2\zeta)\;d\tau`,
    #. update the momenta by :math:`\frac{1}{2}\; d\tau`,
    #. update the coordinates by :math:`\zeta\;d\tau`.

    However, if the number of steps is more than 1 the trailing coordinate update from one step is combined with the leading coordinate update from the next.

    If nothing is known about the structure of the potential, the :math:`h^3` errors are minimized when :math:`\zeta \approx 0.193` :cite:`PhysRevE.65.056706`.

    When :math:`\zeta=0` this reproduces the momentum-first leapfrog; when :math:`\zeta=0.5` it reproduces the position-first LeapFrog.
    """

    def __init__(self, H, md_steps, md_time=1, zeta=0.193):
        self.H = H
        self.md_time  = md_time
        self.md_steps = md_steps
        self.md_dt    = self.md_time / self.md_steps
        self.zeta     = zeta

        if (zeta < 0) or (0.5 < zeta):
            raise ValueError("Second-order integrators need 0 <= zeta <= 0.5 for any hope of improvement over LeapFrog.")

    def __str__(self):
        return f'Omelyan(H, md_steps={self.md_steps}, md_time={self.md_time}, zeta={self.zeta})'

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
        # Take an initial zeta-step of the coordinates
        x = x_i + self.H.velocity(p_i) * (self.zeta * self.md_dt)
        
        # do the initial half-step of momentum update,
        p = p_i + self.H.force(x) * (self.md_dt / 2)
        

        # Now do whole-dt coordinate AND momentum updates
        for md_step in range(self.md_steps-1):
            x = x + self.H.velocity(p) * ((1-2*self.zeta) * self.md_dt)
            p = p + self.H.force(x) * (self.md_dt / 2)
            x = x + self.H.velocity(p) * (2 * self.zeta * self.md_dt)
            p = p + self.H.force(x) * (self.md_dt / 2)

        # do the middle coordinate step of (1-2zeta)
        x = x + self.H.velocity(p) * ((1-2 * self.zeta) * self.md_dt)

        # do the final half-step of momentum integration,
        p = p + self.H.force(x) * (self.md_dt / 2)

         # take a final coordinate zeta-step
        x = x + self.H.velocity(p) * (self.zeta * self.md_dt)

        return x, p
    
    def __call__(self, x_i, p_i):
        '''
        Forwards to ``integrate``.
        '''
        return self.integrate(x_i, p_i)

