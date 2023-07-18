#!/usr/bin/env python3

import torch
from itertools import product

import matplotlib.pyplot as plt

class History:
    r'''
    Markov Chains provide a natural time along which measurements vary.
    Both the history and the total histogram are informative.

    Parameters
    ----------
        rows:   int
            Number of rows; they share a common time coordinate.
        histogram: int
            The width of the histogram is one part in ``histogram`` of the full width.
        row_height: float [inches]
            The height of each row.
        width: float [inches]
            The width of the figure.
        kwargs:
            Forwarded to ``matplotlib.pyplot.subplots``
    '''
    def __init__(self, rows=1, histogram=5, row_height=3, width=12, **kwargs):
        self.fig, self.ax = plt.subplots(
            rows, 2, sharey='row',
            squeeze = False,
            gridspec_kw={'width_ratios': [histogram-1, 1], 'wspace': 0},
            figsize = (width, rows*row_height),
            **kwargs
        )
        self.history   = self.ax[:,0]
        self.histogram = self.ax[:,1]

        # The histograms need not be on the same scale.
        # But the hitories should be.
        for h in self.history:
            h.sharex(self.history[0])
        
    def plot(self, data, row=0, x=None, frequency=1, color=None, **kwargs):
        r'''
        Parameters
        ----------
            data:
                A one-dimensional set of data to visualize.
            row: 
                Which row to plot in.
            x:
                If not ``None``, used as the time parameter.
            frequency: int
                Plotting every sample can prove visually overwhelming.
                To reduce the number of points in the temporal history, only plot once per frequency.
            color:
                Forwarded `matplotlib color <https://matplotlib.org/stable/tutorials/colors/colors.html>`_.
        '''
        if isinstance(data, torch.Tensor):
            d = data.clone().detach().cpu().numpy()
        else:
            d = data
        self._plot_history  (d, row=row, x=x, frequency=frequency, color=color, **kwargs)
        self._plot_histogram(d, row=row, color=color, **kwargs)
        
    def _plot_history(self, data, row=0, x=None, label=None, frequency=1, color=None, **kwargs):
        if x is None:
            x = torch.arange(0, len(data), frequency)
        self.ax[row,0].plot(x[::frequency].cpu(), data[::frequency], label=label, color=color)
        
    def _plot_histogram(self, data, row=0, label=None, density=True, alpha=0.5, bins=31, color=None, **kwargs):
        self.ax[row,1].hist(
            data, label=label,
            orientation='horizontal',
            bins=bins, density=density,
            color=color, alpha=alpha,
        )
