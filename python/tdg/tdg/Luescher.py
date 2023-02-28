#!/usr/bin/env python3

from tdg.h5 import H5able
import numpy as np
import torch

class Zeta2D(H5able):
    r'''
    In the continuum finite-volume energy levels may be translated into scattering data via *Lüscher's quantization condition*.

    In two dimensions, in the :math:`A_1` sector the energies can be converted into information about the s-wave phase shift,

    .. math::
        \begin{align}
           \cot \delta(p) - \frac{2}{\pi} \log \frac{pL}{2\pi} &= \frac{1}{\pi^2} S_2\left(\left(\frac{pL}{2\pi}\right)^2\right)
            &
            S_2(x) &= \lim_{N\rightarrow\infty} \sum_{n^2 \leq \left(\frac{N}{2}\right)^2} \frac{1}{n^2-x} - 2\pi \log \frac{N}{2}
        \end{align}


    where :math:`L` is the physical size of the lattice and :math:`p` is the physical momentum.  In dimensionless units,

    .. math::
        \begin{align}
        \cot \delta(p) &= \frac{2}{\pi} \log \sqrt{x} + \frac{1}{\pi^2} S_2(x)
        &
        x &= \left(\frac{pL}{2\pi}\right)^2
        \end{align}

    Parameters
    ----------
    N:
        the cutoff in the sum that defines :math:`S_2`.

    '''

    def __init__(self, N=200):
        # N defaults to 200 in my Mathematica implementation.
        # Differs from N=2000 by at most 0.006 on torch.linspace(-5.001,20.001,2000)

        super(Zeta2D,self).__init__()

        self.N = torch.tensor(N)

        max_n = torch.ceil((self.N+1)/2).int()
        squares = torch.arange(-max_n,max_n+1)**2

        # A naive method quadratic in N that finds all the poles and their multiplicities.
        # Mathematica seems to have a more direct method called SquaresR.
        # I'm not sure how much number theory is required for a faster implementation.
        # A004018 https://oeis.org/A004018 is almost what is needed; but it counts for radii
        # rather than for orbits.

        # Here are all possible combinations of squares.
        points = torch.tensor([[i+j for i in squares] for j in squares])
        points, multiplicity = points.flatten().unique(return_counts=True)

        # Ensure we've regulated so that every included pole has |n| <= N/2
        pm = torch.tensor([[p, m] for p, m in zip(points, multiplicity) if 4*p <= self.N**2]).transpose(0,1)

        self.poles = pm[0]
        self.multiplicity = pm[1]

        # With the 2-norm regulation the counterterm is simple.
        self.counterterm = 2*np.pi * torch.log(self.N/2)

    def __call__(self, x):
        r'''
        Apply the finite-volume S to x.

        Parameters
        ----------
        x:  torch.tensor
            :math:`x = (pL/2\pi)^2`.

        Returns
        -------
        torch.tensor:
            :math:`S(x)`.  :math:`x = E / (2\pi)^2` where :math:`E` is the dimensionless energy of the dimensionless two-body Schrödinger equation :math:`S(x)` goes into the Lüscher quantization condition.
        '''
        # This computes S(x) which satisfies the finite-volume formula
        #
        #   cot δ(p) - 2/π log√x = S(x) / π^2
        #
        # when x = (pL/2π)^2 = E / (2π)^2
        # and E is the dimensionless energy of the dimensionless two-body Schrödinger equation.

        # TODO: there must be a smarter torch way to do this?
        if isinstance(x, (float, complex)):
            return self(torch.tensor([x]))[0]
        if isinstance(x, np.ndarray):
            return self(torch.tensor(x)).cpu().to_numpy()
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
        r"""Plots the zeta function for values of ``x`` on the matplotlib axis ``ax``.
        
        Takes care not to plot the sharp jumps at the poles of the zeta function.

        Parameters
        ----------
            ax: matplotlib.pyplot.axis
                The axis on which to draw
            x:  torch.tensor
                The values to plot along the x-axis
            asymptote_color: 
                A `matplotlib color`_ do draw vertical asymptotes at the poles of the zeta.
            asymptote_linestyle: str
                A matplotlib linestyle for the asymptotes
            xlabel: str
                A label for the x-axis
            ylabel: str
                A label for the y-axis
            kwargs:
                get forwarded to ``ax.plot``

        .. _matplotlib color: https://matplotlib.org/stable/tutorials/colors/colors.html
        """

        low = min(x)
        high = max(x)

        if low < 0:
            poles = [-float("inf")]
        else:
            poles = []
        poles += [p for p in self.poles if low < p and p < high]
        poles = torch.tensor(poles)

        if 'color' in kwargs:
            color = kwargs['color']
            del kwargs['color']
        else:
            color = next(ax._get_lines.prop_cycler)['color']

        for l, h in zip(poles, poles[1:]):
            X = torch.tensor([xi for xi in x if l < xi and xi < h])
            Z = self(X) / torch.pi**2
            ax.plot(X.cpu(), Z.cpu(), color=color, **kwargs)

        for p in poles:
            ax.axvline(p.cpu(), color=asymptote_color, alpha=asymptote_alpha, linestyle=asymptote_linestyle)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
