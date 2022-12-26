#!/usr/bin/env python3

from functools import cached_property
import torch

class ReducedTwoBodyA1Hamiltonian:
    r'''
    We can project our Hamiltonian to the two-body sector.  If we project to total momentum 0 we need only track the relative coordinate :math:`r`,

    .. math::
        \begin{align}
            H \left|0,r\right\rangle
            &=
                \sum_{r'} 2 \kappa_{rr'} \left|0,r\right\rangle
            +   V_{0,r} \left|0,r\right\rangle
        \end{align}

    and the potential :math:`V = \sum_{\vec{R}} C_{\vec{R}}\mathcal{S}^{\vec{R}}` is a sum of LegoSpheres.

    The primary purpose of studying this two-body sector is to *tune* the Wilson coefficients :math:`C_{\vec{R}}` to produce a spectrum with desired features.
    If we match the two-body sector to the desired two-body scattering amplitudes (via the effective range expansion), we can take the resulting Wilson coefficients
    and use them in the many-body sector.

    We construct H *in momentum space*.  This allows us to make a straightforward projection to the :math:`A_1` sector of the :math:`D_4` lattice symmetry
    (or, more precisely, the little group of :math:`D_4` for zero total-momentum, which is :math:`D_4` itself).

    In momentum space the kinetic piece is diagonal and straightforward, while the LegoSphere :math:`\mathcal{S}^{\vec{R}}` is

    .. math::
        \begin{align}
            \left\langle A_1, n' \middle| \mathcal{S}^{\vec{R}} \middle| A_1, n \right\rangle
            &=
            \frac{1}{\mathcal{N}_{n'}\mathcal{N}_{n}} \frac{1}{\mathcal{V}} \sum_{g'g \in D_4} e^{2\pi i (gn-g'n') \vec{R} / N_x}
            &
            \left|A_1, \vec{n} \right\rangle &= \frac{1}{\mathcal{N}_n} \sum_{g\in D_4} \left| g\vec{n} \right\rangle
        \end{align}

    where :math:`\vec{n}` is a vector of lattice momentum and the normalization depends on the size of the orbit of :math:`\vec{n}` under :math:`D_4`.

    .. note::
        A user should only very rarely need to directly construct a ``ReducedTwoBodyA1Hamiltonian``; but it is integral to the :class:`~.Tuning`.

    Parameters
    ----------
        lattice:        :class:`~.Lattice`
                        The spatial lattice on which the Hamiltonian describes dynamics.
        legoSpheres:    list of :class:`~.LegoSphere`
                        The spheres in the interaction.
    '''

    def __init__(self, lattice, legoSpheres):
        self.Lattice = lattice
        self.LegoSpheres = legoSpheres
        self.spheres = len(self.LegoSpheres)

        self.spherical_operators = []
        r'''
        A list of matrix representations of the LegoSpheres themselves, with no Wilson coefficients.
        '''

        norms = 1/(torch.sqrt(torch.einsum('i,j->ij',self.shellSizes, self.shellSizes))*self.Lattice.nx**2)
        for sphere in self.LegoSpheres:
            expdot = [torch.exp(2j*torch.pi/self.Lattice.nx * torch.einsum('sx,x->s',shell,sphere.r)) for shell in self.shells]
            operator = torch.tensor([[torch.sum(torch.outer(m,torch.conj(n))) for n in expdot] for m in expdot])*norms
            self.spherical_operators+=[operator]

    @cached_property
    def states(self):
        r'''
        A complete basis of :math:`A_1`-projected momentum states for the ``lattice``.
        This basis is *much smaller* than the number of lattice sites; for each orbit there is only one state.
        '''
        return self.Lattice.coordinates[
            torch.where(
                (0 <= self.Lattice.coordinates[:, 0]) &
                (self.Lattice.coordinates[:,0] <= self.Lattice.coordinates[:,1])
            )
        ]

    @cached_property
    def shells(self):
        r'''
        A list of tensors, each tensor contains all the lattice momenta in a single :math:`D_4` orbit.
        The sum of the lengths of all the tensors equals the number of lattice sites.
        '''
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
        r'''
        A tensor of integers which give the size of the orbits.
        '''
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
        r'''
        A diagonal matrix which implements the kinetic energy in the :math:`A_1`-projected momentum basis ``states``.
        '''
        # A diagonal matrix with entries = 2 (reduced mass) * 1/2 * (2π/nx)^2 * n^2
        # Don't bother computing 2 * 1/2 = 1.
        return torch.diag((2*torch.pi/self.Lattice.nx)**2 * torch.einsum('np,np->n', self.states, self.states))

    def potential(self, C):
        r'''
        Parameters
        ----------
            C:  torch.tensor
                Wilson coefficients, one for each LegoSphere.

        Returns
        -------
            torch.tensor
                A dense matrix, the sum of :math:`\sum_{\vec{R}} C_{\vec{R}} \mathcal{S}^{\vec{R}}`
        '''

        V = torch.zeros_like(self.spherical_operators[0])
        for c, o in zip(C, self.spherical_operators):
            V += c * o
        return V

    def operator(self, C):
        r'''
        Parameters
        ----------
            C:  torch.tensor
                Wilson coefficients, one for each LegoSphere.

        Returns
        -------
            torch.tensor
                ``kinetic + potential(C)``
        '''
        return self.kinetic + self.potential(C)

    def eigenenergies(self, C):
        r'''
        Parameters
        ----------
            C:  torch.tensor
                Wilson coefficients, one for each LegoSphere.

        Returns
        -------
            torch.tensor
                A list of the eigenvalues of the :math:`A_1`-projected two-body sector of the Hamiltonian constructed with Wilson coefficients C.
        '''
        return torch.linalg.eigvalsh(self.operator(C))

    def tuning(self, target_energies, start=None, epochs=10000, lr=0.001):
        r'''
        *Tuning* the Hamiltonian solves the inverse problem: which Wilson coefficients do we need to produce
        some energy eigenvalues and a set of LegoSphere operators?

        Parameters
        ----------
            target_energies: torch.tensor
                A list of finite-volume energies.
            start: torch.tensor
                Starting guesses for the Wilson coefficients.

            epochs: int
                How many minimization steps to take.
            lr:     float
                The learning rate for the minimizer.

        Returns
        -------
            torch.tensor
                A list of Wilson coefficients that produce the target energies for the lowest states.

        '''
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
