#!/usr/bin/env python3

import torch
from tdg.Luescher import Zeta2D

class EffectiveRangeExpansion:

    def __init__(self, parameters):
        """
            Parameters are normalized by appropriate powers of 2π/L.

            parameters = [a, *analytic]

            a           the dimensionless scattering length, 2πa/L
            analytic    the coefficients in the dimensionless expansion;
                        analytic[1] = (2πre/L)^2 since the effective range multiplies the x^1 term in
                        the finite-volume dimensionless ERE,

            cot δ = 2/π[ log(1/2 a √x) + γ ] + 1/4 re^2 x + ...

            Having not found a convention for the pure numbers on higher order terms,
            all higher-order terms analytic in x have a 1/4 coefficients for simplicity.

            Presumably we will tune to scattering amplitudes constant in x anyway.
        """

        self.a = parameters[0]
        assert self.a > 0, "In 2D the scattering length must be positive-definite."

        self.coefficients = parameters[1:]
        self.powers = torch.arange(len(self.coefficients)) + 1
        self._gamma = torch.distributions.utils.euler_constant

    def analytic(self, x):
        """
        Includes the constant piece = 2/π [ log(a/2) + γ ];

        ere.analytic(x) = 2/π [ log(a/2) + γ ] + 1/4 coefficients * x^powers
        """
        return 2/torch.pi * ( torch.log(0.5 * self.a) + self._gamma ) + 0.25 * torch.sum( self.coefficients * x[:,None]**self.powers, axis=1)

    def __call__(self, x):
        return 2/torch.pi * torch.log(torch.sqrt(x)) + self.analytic(x)

    def target_energies(self, nx, levels, zeta=Zeta2D(), lr = 0.05, epochs=100000):
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
        return (2*torch.pi/nx)**2 * X
