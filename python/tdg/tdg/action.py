#!/usr/bin/env python3

import torch

from tdg.fermionMatrix import FermionMatrix
from tdg.h5 import H5able

import logging
logger = logging.getLogger(__name__)

class Action(H5able):
    r'''
    Parameters
    ----------
        spacetime:  tdg.Spacetime
            the euclidean spacetime
        potential:  tdg.Potential
            the interaction with which the fermions interact; must be negative-definite for the Hubbard-Stratanovich transformation used to make sense.
        beta:       torch.tensor scalar
            the inverse temperature
        mu:         torch.tensor scalar
            the chemical potential
        h:          torch.tensor
            the spin chemical potential; a triplet
        fermion:    tdg.FermionMatrix
            the fermion matrix corresponding to the desired discretization

    An auxiliary-field spacetime action for fermions interacting via a potential with inverse temperature :math:`\beta`, chemical potential :math:`\mu`, spin chemical potential :math:`\vec{h}`.

    The action is used for importance sampling, since the partition function Z is

    .. math::
        \begin{align}
        Z &= \int DA\; e^{-S}
        &
        S &= \frac{1}{2} \sum_t A_t (-\Delta t V)^{-1} A - \log \det \mathbb{d} + \frac{N_t}{2} \text{tr} \log \left(-2\pi \Delta t V\right)
        \end{align}

    where :math:`\mathbb{d}` is the fermion matrix, the time step :math:`dt = \beta/N_t`, and everything should be understood to be dimensionless.

    The last term is a constant normalization to make :math:`Z` truly equal to the Trotterization of :math:`\text{tr} e^{-\beta H}`.
    '''

    def __init__(self, spacetime, potential, beta, mu=torch.tensor(0, dtype=torch.float), h=torch.tensor([0,0,0], dtype=torch.float), fermion=FermionMatrix):
        self.Spacetime = spacetime
        r'''The :class:`~.Spacetime` on which the action is formulated.'''
        self.Potential = potential
        r'''The :class:`~.Potential` :math:`V` with which the fermions interact.'''

        self.beta = beta
        r''' The dimensionless inverse temperature :math:`\tilde{\beta} = \beta/ML^2`.'''
        self.dt = beta / self.Spacetime.nt
        r''' The temporal discretization :math:`dt = \texttt{beta} / N_t`.'''

        self.mu = mu
        r''' The chemical potential :math:`\tilde{\mu} = \mu ML^2`.'''
        self.h  = h
        r''' The spin chemical potential :math:`\tilde{\vec{h}} = \vec{h} ML^2`.'''
        self.absh = torch.sqrt(torch.einsum('i,i->', self.h, self.h))
        if self.absh == 0.:
            self.hhat = torch.tensor([0,0,1.])
        else:
            self.hhat = self.h / self.absh

        self.V = self.Potential.spatial(self.Spacetime.Lattice)
        r'''The spatial representation of :attr:`Potential` on the ``Spacetime.Lattice``'''
        self.Vinverse = self.Potential.inverse(self.Spacetime.Lattice)
        r'''The inverse of :attr:`Potential` on the ``Spacetime.Lattice``'''

        # Recall that requiring the contact interaction
        # be written as the quadratic nVn induces a term in the Hamiltonian
        # proportional to n itself; a chemical potential equal to - volume * C0/2.
        #
        # Since we work with H-µN-hS this ADDS to the physical chemical potential.
        self.fermion = fermion
        self.FermionMatrix = fermion(self.Spacetime, self.beta, mu=self.mu + self.Spacetime.Lattice.sites * potential.C0/2, h=self.h)
        r'''The fermion matrix that gives the discretization.

        .. note::
            *On the chemical potential:*

            Recall that requiring the contact interaction be written as the quadratic :math:`nVn` induces
            a term in the Hamiltonian proportional to :math:`n` itself, which looks just like a chemical potential term.
            This term comes with a coefficient equal to :math:`-N_x^2 C_0/2`.

            Our sign convention is that the internal energy is :math:`H-\mu N - h\cdot S` so the the signs conspire.
            The fermion matrix is constructed with the 'offset' chemical potential :math:`\mu + N_x^2 C_0/2`.
        '''

        self.normalizing_offset = self.Spacetime.nt / 2 * torch.sum( torch.log(-2*torch.pi*self.dt * self.Potential.eigvals(self.Spacetime.Lattice)))
        r'''
        The A-independent contribution to the action needed to match Z to the Trotterized operator definition.

        .. math::
            \frac{N_t}{2} \text{tr} \log\left(-2\pi \Delta t V\right)
        '''

        self.quenched = torch.distributions.multivariate_normal.MultivariateNormal(
            self.Spacetime.Lattice.vector().flatten(),
            -self.dt * self.V
            ).expand([self.Spacetime.nt])

    def __str__(self):
        return f"Action(β̃={self.beta}, µ̃={self.mu}, h̃={self.h}, {self.Spacetime}, {self.Potential})"

    def __repr__(self):
        return str(self)

    def gauge(self, A):
        r'''
        Parameters
        ----------
            A: torch.tensor
                an auxiliary field configuration

        Returns
        -------
            torch.tensor
                :math:`\frac{1}{2} \sum_t A_t (-\Delta t V)^{-1} A_t`
        '''
        # S_gauge = 1/2 Σ(t) A(t) inverse(- ∆t V) A(t)
        # We can pull the minus sign in the inverse out front.
        return -0.5 * torch.einsum('ta,ab,tb->', A, self.Vinverse.to(A.dtype), A) / self.dt

    def fermionic(self, A):
        r'''
        Parameters
        ----------
            A: torch.tensor
                an auxiliary field configuration

        Returns
        -------
            torch.tensor
                :math:`-\log \det \mathbb{d}`
        '''
        # S_fermionic = - log det d
        return - self.FermionMatrix.logdet(A)

    def __call__(self, A):
        r'''
        Parameters
        ----------
            A: torch.tensor
                an auxiliary field configuration

        Returns
        -------
            torch.tensor
                :math:`S(A) =\texttt{S.gauge}(A) + \texttt{S.fermionic}(A) + \texttt{S.normalizing_offset}`.
        '''
        # S = 1/2 Σ(t) A(t) inverse(- ∆t V) A(t) - log det d + nt/2 tr log(-2π ∆t V )
        return self.gauge(A) + self.fermionic(A) + self.normalizing_offset

    def quenched_sample(self, sample_shape=torch.Size([])):
        r'''
        Provides sample auxiliary fields drawn from the gaussian

        .. math::
            p(A) \propto \exp\left( - \frac{1}{2} \sum_t A_t (-\Delta t V)^{-1} A_t \right)

        Parameters
        ----------
        sample_shape: torch.Size
            See `the torch.distributions interface`_ for details.

        Returns
        -------
            torch.tensor
                Shape is ``[*sample_shape, *spacetime.dims]``.  Called with no arguments the default is to produce just one sample.

        .. _the torch.distributions interface: https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.sample
        '''
        return self.quenched.sample(sample_shape)

    def set_tuning(self, ere):
        spheres = self.Potential.spheres
        radii = [s.r for s in spheres]
        coeff = [s.c for s in spheres]
        self.Tuning = tdg.Tuning(ere, self.Lattice, radii, C=coeff)
        return self

    def projected(self, n, s):
        r'''
        Provides a convenience constructor for actions derived from this one that are needed in the implementation of :class:`~Sector`.
        Shifts the chemical potential and external field, via

        .. math::
            \begin{align}
                \mu &\rightarrow \mu + \frac{2\pi i}{2V+1} \frac{n}{\beta}
                &
                \vec{h} &\rightarrow \vec{h} + \frac{2\pi i}{2V+1} \frac{2s}{\beta} \hat{h}
            \end{align}

        Parameters
        ----------
            n: integer
                Fourier sector for particle number projection
            s: half-integer
                Fourier sector for spin projection

        Returns
        -------
            Action
        '''
        phase = 2j*torch.pi / (2*self.Spacetime.Lattice.sites + 1) / self.beta
        S = Action(self.Spacetime, self.Potential, self.beta,
                      self.mu + n * phase,
                      self.h  + 2*s * phase * self.hhat,
                      self.fermion)
        try:    S.Tuning = self.Tuning
        except: pass
        return S

def _demo(nx = 3, nt=8, beta=1, mu=None, h=None, C0=-5.0,  **kwargs):

    # Why not just use default assignments for mu and h in the _demo definition?
    # This prevents the parameters from being evaluated at import time,
    # which is important as the user might torch.set_default_dtype(torch.float64)
    # while the default is float32 (which would be used at tdg-import time).
    # This prevents annoying import-order issues.
    if mu is None:
        mu = torch.tensor(-2.0)
    if h is None:
        h = torch.tensor([0.+0j,0.+0j,0.+0j])

    logger.info(f'demo {mu=} with dtype {mu.dtype}')
    logger.info(f'demo {h=} with dtype {h.dtype}')

    import tdg
    spacetime = tdg.Spacetime(nt, tdg.Lattice(nx))
    V = tdg.Potential(C0*tdg.LegoSphere([0,0]))
    return Action(spacetime, V, beta, mu, h)
