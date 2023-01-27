#!/usr/bin/env python3

import torch
import tdg
from tdg.h5 import H5able

class FermionMatrix(H5able):
    r'''
    The fermion matrix :math:`\mathbb{d}` which corresponds to

    .. math::
        \begin{align}
            \mathbb{d}
            &=
            \begin{pmatrix}
                    -\mathbb{1}                           & 0                                       & 0                                     & 0             & \cdots
                                                          & 0                                       & -\mathbb{B}^{-1}\mathbb{F}_{N_t}( h )
                \\  +\mathbb{B}^{-1}\mathbb{F}_{1}( h )   & -\mathbb{1}                             & 0                                     & 0             & \cdots
                                                          & 0                                       & 0
                \\  0                                     & +\mathbb{B}^{-1}\mathbb{F}_{2}( h )     & -\mathbb{1}                           & 0             & \cdots
                i                                         & 0                                       & 0
                \\  0                                     & 0                                       & +\mathbb{B}^{-1}\mathbb{F}_{3}( h )   & -\mathbb{1}   & \cdots
                                                          & 0                                       & 0
                \\  \vdots                                & \vdots                                  & \vdots                                & \vdots        & \ddots
                                                          & \vdots                                  & \vdots
                \\  0                                     & 0                                       & 0                                     & 0             & \cdots
                                                          & -\mathbb{1}                             & 0
                \\  0                                     & 0                                       & 0                                     & 0             & \cdots
                                                          & +\mathbb{B}^{-1}\mathbb{F}_{N_t-1}( h ) & -\mathbb{1}
            \end{pmatrix}
        \end{align}

    where the double-struck quantities

    .. math::
        \begin{align}
            \mathbb{F}_t &= \left(e^{A_{t}+\Delta t\mu + \frac{1}{2}\Delta t\vec{h}\cdot\vec{\sigma}}\right)
            &
            \mathbb{B} &= (B = \exp \Delta t \kappa) \otimes \mathbb{1}
        \end{align}

    are space ⊗ spin in size.

    In the language of Ref. :cite:`Wynen:2018ryx` this is called the :math:`\alpha=0` exponential discretization
    because the hopping matrix :math:`\kappa` which encodes the kinetic piece of the Hamiltonian appears inside an exponential,
    and we're in the attractive channel, so the auxiliary field appears without an explicit :math:`i` in :math:`\mathbb{F}`.

    Parameters
    ----------
        spacetime: :class:`~.Spacetime`
        beta:   torch.tensor scalar
                The inverse temperature
        mu:     torch.tensor scalar
                The chemical potential
        h:      torch.tensor triplet
                The external field

    .. todo::
        We should add a method to do fermion matrix-vector multiply, perhaps as ``__call__``.
        The programming puzzle is that when :math:`\vec{h}=0` or :math:`\vec{h}\parallel \hat{z}` we can split :math:`\mathbb{d}` into
        two independent pieces that each operate on spacetime vectors, while for generic :math:`\vec{h}` we need one matrix that operates
        on a spacetime ⊗ spin vector.
    '''

    def __init__(self, spacetime, beta, mu=torch.tensor(0, dtype=torch.float), h=torch.tensor([0,0,0], dtype=torch.float)):
        self.Spacetime = spacetime

        self.beta = beta
        self.dt = beta / self.Spacetime.nt

        self.mu = mu
        self.h  = h
        self.absh = torch.sqrt(torch.einsum('i,i->', self.h, self.h))
        if self.absh == 0.:
            self.hhat = torch.tensor([0,0,1.])
        else:
            self.hhat = self.h / self.absh

        self.z = torch.exp(self.beta * self.mu)
        self.zh = torch.exp(0.5 * self.beta * self.absh)

        if self.absh == 0:
            self.exp_half_beta_h = tdg.PauliMatrix[0]
        else:
            self.exp_half_beta_h = torch.cosh( 0.5 * self.absh * self.beta ) * tdg.PauliMatrix[0]
            for h, sigma in zip(self.hhat, tdg.PauliMatrix[1:]):
                self.exp_half_beta_h += torch.sinh( 0.5 * self.absh * self.beta) * h * sigma

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
        r'''
        .. math::
            U(A) = B^{-1} F_{N_t} B^{-1} F_{N_t-1} \cdots B^{-1} F_{2} B^{-1} F_{1}

        where :math:`F_t = \exp A_t` is a digonal matrix and does not include the chemical potential :math:`\mu`.

        :math:`B`, :math:`F` and :math:`U`  are space × space, with rather than (space ⊗ spin) × (space ⊗ spin).

        Parameters
        ----------
            A:  torch.tensor
                An auxiliary-field configuration

        Returns
        -------
            torch.tensor:
                :math:`U(A)`
        '''
        # Computes Binv F(Nt) Binv F(Nt-1) ... Binv F(2) Binv(1)
        # where F = exp(A), excluding the µ and h terms.
        #
        # For numerical stability we may need a smarter method.
        # This naive method is at least in principle correct.

        assert (A.shape == self.Spacetime.vector().shape), f"Gauge field shape {A.shape} must match the dimensions of a spacetime vector {self.Spacetime.vector().shape}"

        # Rather than incorporate µ ∆t into the exponential, since it is spacetime-constant,
        # we can just pull alll nt terms out and multiply by z at the end.

        # First construct all BinvF(t) for each t

        F = torch.exp(A)
        # Since F(t) is a diagonal matrix we don't need to expand it and do 'real'
        # matrix multiplication.  Just use the fast simplification for each timeslice instead.
        BinvF = torch.einsum('ij,tj->tij',self.Binverse, F)

        # Then multiply them togther
        U = torch.eye(self.Spacetime.Lattice.sites) + 0j
        for t in self.Spacetime.t:
            U = torch.matmul(BinvF[t], U)
        return U

    def UU_tensor(self, A):
        r'''
        .. math::
            \mathbb{U} = \mathbb{B}^{-1} \mathbb{F}_{N_t} \cdots \mathbb{B}^{-1} \mathbb{F}_{2} \mathbb{B}^{-1} \mathbb{F}_{1} 

        where :math:`\mathbb{F}` contains the chemical potential and external field terms.  However, since those terms are space-independent we can commute them with :math:`\mathbb{B}` so that

        .. math::
            \mathbb{U} = \exp\left(\frac{1}{2}\beta h\cdot\sigma\right) \begin{pmatrix} zU & 0 \\ 0 & zU\end{pmatrix}

        where :math:`U` is given by :func:`U` and is (space × space).

        Parameters
        ----------
            A:  torch.tensor
                An auxiliary-field configuration.

        Returns
        -------
            torch.tensor:
                :math:`\mathbb{U}` not as a matrix but with shape `[space, space, spin, spin]`.
                This makes the spin indices broadcastable, which is often useful for future manipulation.

        '''
        zU = self.z * self.U(A)
        return torch.matmul(self.exp_half_beta_h, torch.stack(
            (torch.stack((zU, torch.zeros_like(zU))),
             torch.stack((torch.zeros_like(zU), zU)))
            ).permute(2,3,0,1))

    def UU(self, A):
        r'''
        Parameters
        ----------
            A:  torch.tensor
                An auxiliary-field configuration

        Returns
        -------
            torch.tensor:
                The same as :func:`UU_tensor` but as a true matrix of shape (space ⊗ spin) × (space ⊗ spin), with spin fastest.

        .. todo::
            Leveraging this makes some code much slower.  For example,

            >>> UU = torch.stack(
            ...     tuple(Action.FermionMatrix.UU_tensor(cfg) for cfg in configurations)
            ...     ).permute(0,1,3,2,4).reshape(-1,
            ...                                  2*Action.FermionMatrix.Spacetime.Lattice.sites,
            ...                                  2*Action.FermionMatrix.Spacetime.Lattice.sites)

            is much faster to subsequently manipulate than

            >>> UU = torch.stack(
            ...     tuple(Action.FermionMatrix.UU(cfg) for cfg in configuration))
            
            which is something that ought to be understood / addressed.  Perhaps it is autograd- or computational-graph-related?

            Is it possible to construct this code to distribute over a whole ensemble of auxiliary fields?
            This perhaps should inform the way we code other observables.
        '''
        return self.UU_tensor(A).permute(0,2,1,3).reshape(2*self.Spacetime.Lattice.sites, 2*self.Spacetime.Lattice.sites)

    def logdet(self, A):
        r'''
        The log determinant of the fermion matrix appears in the action.


        Parameters
        ----------
            A:  torch.tensor
                An auxiliary-field configuration.

        Returns
        -------
            torch.tensor:
                :math:`\log \det \mathbb{d}(A)`


        .. note ::

            *On implementation:*

            One may show that :math:`\det \mathbb{d} = \det \mathbb{1} + \mathbb{U}` where all of
            the space-independent pieces of :math:`\mathbb{F}` can be pulled through so that

            .. math::
                \mathbb{U} = \exp\left( \frac{1}{2} \beta \vec{h}\cdot\vec{\sigma} \right) \begin{pmatrix} zU & 0 \\ 0 & zU \end{pmatrix}

            Furthermore, an orthogonal transformation can diagonalize the spin-dependent factor and leave the spin-independent 
            factor alone.  The determinant of :math:`\mathbb{d}` can then be expressed as

            .. math::
                \det \mathbb{d} = \det\left( 1 + e^{+\frac{1}{2}\beta \sqrt{h^2}} z U\right) \det\left( 1 + e^{-\frac{1}{2}\beta \sqrt{h^2}} z U\right)

            where the fugacity :math:`z = \exp(\beta \mu)` and the :math:`\sqrt{h^2}` helps correctly handle complex :math:`h`.
        '''
        one = torch.eye(self.Spacetime.Lattice.sites) + 0j
        zU = self.z * self.U(A)
        return torch.log(torch.det(one + zU*self.zh)) + torch.log(torch.det(one + zU/self.zh))

