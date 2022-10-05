#!/usr/bin/env python3

import numpy as np
import torch

class Zeta2D(torch.nn.Module):

    def __init__(self, N=200):
        # N defaults to 200 in my Mathematica implementation.
        # Differs from N=2000 by at most 0.006 on torch.linspace(-5.001,20.001,2000)

        super(Zeta2D,self).__init__()

        self.N = N

        max_n = np.ceil((N+1)/2)
        squares = np.arange(-max_n, max_n+1)**2

        # A naive method quadratic in N that finds all the poles and their multiplicities.
        # Mathematica seems to have a more direct method called SquaresR.
        # I'm not sure how much number theory is required for a faster implementation.

        # Here are all possible combinations of squares.
        points = np.array([[i+j for i in squares] for j in squares])
        points, multiplicity = np.unique(points.flat, return_counts=True)

        # Ensure we've regulated so that every included pole has |n| <= N/2
        pm = np.array([[p, m] for p, m in zip(points, multiplicity) if 4*p <= N**2]).transpose()
        points = pm[0]
        multiplicity = pm[1]

        self.poles = torch.from_numpy(points)
        self.multiplicity = torch.from_numpy(multiplicity)

        # With the 2-norm regulation the counterterm is simple.
        self.counterterm = torch.tensor(2*np.pi * np.log(self.N/2))

    def __call__(self, x):
        # This computes S(x) which satisfies the finite-volume formula
        #
        #   cot δ(p) - 2/π log√x = S(x) / π^2
        #
        # when x = (pL/2π)^2 = E N^2 / (2π)^2
        # and E is the dimensionless energy of the dimensionless two-body Schrödinger equation.

        # TODO: there must be a smarter torch way to do this?
        if isinstance(x, (float, complex)):
            return self(torch.tensor([x]))[0]
        if isinstance(x, np.ndarray):
            return self(torch.tensor(x)).to_numpy()
        if isinstance(x, torch.Tensor):
            return torch.sum(self.multiplicity / (self.poles -x[:, None]), dim=1) - self.counterterm
        raise TypeError("Zeta2D must be evaluated on a scalar, numpy array, or torch tensor.")

    def plot(self, ax, x,
             asymptote_color='gray',
             asymptote_alpha=0.5,
             asymptote_linestyle='dashed',
             xlabel=r'$x$',
             ylabel=r'$\cot\delta - \frac{2}{\pi} \log\sqrt{x}$',
             **kwargs):
        # Plot the zeta function evaluated at all values of x.
        # Take care to not plot the illusory sharp jumps at the poles.

        low = min(x)
        high = max(x)

        if low < 0:
            poles = [-float("inf")]
        else:
            poles = []
        poles += [p for p in self.poles if low < p and p < high]

        if 'color' in kwargs:
            color = kwargs['color']
            del kwargs['color']
        else:
            color = next(ax._get_lines.prop_cycler)['color']

        for l, h in zip(poles, poles[1:]):
            X = torch.tensor([xi for xi in x if l < xi and xi < h])
            Z = self(X) / torch.pi**2
            ax.plot(X, Z, color=color, **kwargs)

        for p in poles:
            ax.axvline(p, color=asymptote_color, alpha=asymptote_alpha, linestyle=asymptote_linestyle)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
