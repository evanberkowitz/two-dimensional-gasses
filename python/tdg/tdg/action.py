#!/usr/bin/env python3

import torch

from tdg.fermionMatrix import FermionMatrix


class Action:
    r'''
    Parameters
    ----------
        spacetime:  tdg.Spacetime
            the euclidean spacetime
        potential:  tdg.Potential
            the interaction with which the fermions interact; must be negative-definite for the Hubbard-Stratanovich transformation used to make sense.
        beta:       torch.tensor scalar
            the inverse temperature
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
        r'''The spacetime on which the action is formulated.'''
        self.Potential = potential
        r'''The potential :math:`V` with which the fermions interact.'''

        self.beta = beta
        r''' The inverse temperature :math:`\beta`.'''
        self.dt = beta / self.Spacetime.nt
        r''' The temporal discretization :math:`dt = \beta / N_t`.'''

        self.mu = mu
        r''' The chemical potential :math:`\mu`.'''
        self.h  = h
        r''' The spin chemical potential :math:`\vec{h}`.'''

        self.V = self.Potential.spatial(self.Spacetime.Lattice)
        r'''The spatial representation of ``Potential`` on the ``Spacetime.Lattice``'''
        self.Vinverse = self.Potential.inverse(self.Spacetime.Lattice)
        r'''The inverse of ``Potential`` on the ``Spacetime.Lattice``'''

        # Recall that requiring the contact interaction
        # be written as the quadratic nVn induces a term in the Hamiltonian
        # proportional to n itself; a chemical potential equal to - C0/2.
        #
        # Since we work with H-µN-hS this ADDS to the physical chemical potential.
        self.FermionMatrix = fermion(self.Spacetime, self.beta, mu=self.mu + potential.C0/2, h=self.h)
        r'''The fermion matrix that gives the discretization.

        .. note::
            *On the chemical potential:*

            Recall that requiring the contact interaction be written as the quadratic :math:`nVn` induces
            a term in the Hamiltonian proportional to :math:`n` itself, which looks just like a chemical potential term.
            This term comes with a coefficient equal to :math:`-C_0/2`.

            Our sign convention is that the free energy is :math:`H-\mu N - h\cdot S` so the the signs conspire.
            The fermion matrix is constructed with the 'offset' chemical potential :math:`\mu + C_0/2`.
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
        return f"Action(β={self.beta}, µ={self.mu}, h={self.h}, {self.Spacetime}, {self.Potential})"

    def __repr__(self):
        return str(self)

    def __call__(self, A):
        r'''
        Parameters
        ----------
            A: torch.tensor
                an auxiliary field configuration

        Returns
        -------
            torch.tensor
                :math:`S(A)`.
        '''
        # S = 1/2 Σ(t) A(t) inverse(- ∆t V) A(t) - log det d + nt/2 tr log(-2π ∆t V )
        gauss = -0.5 * torch.einsum('ta,ab,tb->', A, self.Vinverse.to(A.dtype), A) / self.dt

        fermionic = - self.FermionMatrix.logdet(A)

        return gauss + fermionic + self.normalizing_offset

    def quenched_sample(self, sample_shape=torch.Size([])):
        r'''
        Provides sample auxiliary fields drawn from the gaussian

        .. math::
            p(A) \propto \exp\left( - \frac{1}{2} \sum_t A_t (-\Delta t V)^{-1} A \right)

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

def _demo(nx = 3, nt=8, beta=1, mu=torch.tensor(-2.0), h=torch.tensor([0,0,0], dtype=torch.complex128), C0=-5.0,  **kwargs):

    import tdg
    spacetime = tdg.Spacetime(nt, tdg.Lattice(nx))
    V = tdg.Potential(C0*tdg.LegoSphere([0,0]))
    return Action(spacetime, V, beta, mu, h)
