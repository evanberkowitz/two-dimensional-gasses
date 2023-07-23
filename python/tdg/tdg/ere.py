#!/usr/bin/env python3

import torch
from tdg.h5 import H5able
from tdg.Luescher import Zeta2D

import logging
logger = logging.getLogger(__name__)

class EffectiveRangeExpansion(H5able):
    r'''
    The *effective range expansion* is an expansion of :math:`\cot\delta` as a function of momentum.
    In 2D the :math:`\ell=0` partial wave scattering amplitude can be expanded

    .. math::
        \cot \delta_0(p) = \frac{2}{\pi}\log pa + r_e^2 p^2 + \cdots

    in the convention of Ref. :cite:`Beane:2022wcn`,
    rather than the hard-disk geometric convention of :cite:`Adhikari:1986a,Adhikari:1986b,Khuri:2008ib,Galea:2017jhe`.

    Because we are in 2D and there is an inescapable log, it is profitable to convert the momentum dependence into
    dependence on dimensionless :math:`x=(pL/2\pi)^2 = \tilde{E} / (2\pi)^2`.
    Then the effective range expansion is

    .. math::
            \begin{align}
            \cot \delta(p)
            = \frac{2}{\pi} \log\frac{2\pi a}{L} \sqrt{x}
            + \left(\frac{2\pi r_e }{L}\right)^2 x
            + \cdots
            \end{align}

    where the dimensionful parameters are normalized by appropriate powers of 2π/L.

    Parameters
    ----------
        parameters: iterable
            ``[a, *analytic]``, where

            * ``a > 0`` is the dimensionless scattering length 2πa/L
            * ``*analytic`` is a sequence of coefficients in the dimensionless expansion
              ``analytic[0] = (2πre/L)^2`` since the effective range multiplies the :math:`x^{(0+1)}` term.
              These coefficients may be of either sign.

              Having not found a convention for the pure numbers on higher order terms,
              all higher-order terms analytic in x have coefficients of 1 for simplicity.

              Presumably we will tune to scattering amplitudes constant in x anyway.
    '''

    def __init__(self, parameters):

        self.parameters = parameters.clone().requires_grad_(True)
        self.a = self.parameters[0]
        '''
        The dimensionless scattering length.
        '''
        assert self.a > 0, "In 2D the scattering length must be positive-definite."

        self.coefficients = self.parameters[1:]
        '''
        The dimensionless shape parameters.  We assume that in the expansion in :math:`x` every term gets a numerical coefficient of 1 × (the dimensionless shape parameter).
        '''

        self.powers = torch.arange(len(self.coefficients)) + 1
        '''
        The powers of :math:`x` that go with the ``coefficients``.
        '''

    def __str__(self):
        return f"ERE(" + (", ".join(f'{p:+.8f}' for p in self.parameters)) + ")"

    def analytic(self, x):
        r"""
        Includes the constant piece = 2/π log(a)

        Parameters
        ----------
            x: float or torch.tensor

        Returns
        -------
            torch.tensor
                :math:`\texttt{analytic(x)} = 2/π \log(a) + \texttt{coefficients} * x^\texttt{powers}`
        """
        return 2/torch.pi * torch.log(self.a) + torch.sum( self.coefficients * x[:,None]**self.powers, axis=1)

    def __call__(self, x):
        r'''
        Evaluates :math:`\cot\delta` for the given :math:`x`.

        Parameters
        ----------
            x: float or torch.tensor

        Returns
        -------
            torch.tensor
                :math:`2/\pi \log \sqrt{x} + \texttt{analytic}(x)`
        '''
        return 2/torch.pi * torch.log(torch.sqrt(x)) + self.analytic(x)

    def target_energies(self, levels, zeta=Zeta2D(), lr = 0.001, epochs=10000):
        r'''
        Given the ERE, we can use the Lüscher quantization condition to find the values of :math:`x`
        that correspond to that ERE.  Then, those :math:`x` can be transformed into dimensionless
        energies.  Those energies will be eigenvalues of the two-body :math:`A_1`-projected Hamiltonian.
        
        Parameters
        ----------
            levels: int
                how many energies to find
            zeta: callable
                the Lüscher zeta function
            lr: float
                the learning rate of the minimizer that solves the inverse problem for :math:`x`.
            epochs: int
                the number of minimization steps
        '''
        # We need to find x that satisfy
        #     self(x) - 2/π log(√x) == zeta(x) / π^2
        # One way to do that is to simply rearrange to get
        #     0 = self(x) - 2/π log(√x) - zeta(x) / π^2,
        # and minimize a loss between 0 and the RHS.
        # However, the logs wind up giving an annoying problem when trying to optimize,
        # since they can yield complex values.  We will instead cancel the log√x by hand
        # to get the constraint
        def constraint(x):
            # equal to self(x) - 2/π log(√x) - zeta(x) / π^2 but circumvents a cancellation of complex values.
            return self.analytic(x) - zeta(x) / torch.pi**2
        # which ought to be tuned to 0.

        # We can tune on the lowest branches of the zeta function.
        X = torch.cat((torch.tensor([-0.5]), 0.5*(zeta.poles[:levels-1] + zeta.poles[1:levels])))
        X.requires_grad_(True)

        # Now optimize:
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            [X],
            lr = lr,
        )

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss(torch.zeros_like(X), constraint(X)).backward()
            optimizer.step()

        # and return the energies
        return (2*torch.pi)**2 * X

def _demo(parameters=None):
    if not parameters:
        # Why not just use parameters=default instead of _demo(parameters=None)?
        # This prevents the parameters from being evaluated at import time,
        # which is important as the user might torch.set_default_dtype(torch.float64)
        # while the default is float32 (which would be used at tdg-import time).
        # This prevents annoying import-order issues.
        parameters = torch.tensor([1.0, 0.0])
    logger.info(f'demo {parameters=} with dtype {parameters.dtype}')
    return EffectiveRangeExpansion(parameters)
