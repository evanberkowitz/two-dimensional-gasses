#!/usr/bin/env python3

import numpy as np
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
            rows, 2, sharex='col', sharey='row',
            squeeze = False,
            gridspec_kw={'width_ratios': [histogram-1, 1], 'wspace': 0},
            figsize = (width, rows*row_height),
            **kwargs
        )
        self.history   = self.ax[:,0]
        self.histogram = self.ax[:,1]
        
    def plot(self, data, row=0, x=None, frequency=1, **kwargs):
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
        '''
        self._plot_history  (data, row=row, frequency=frequency, **kwargs)
        self._plot_histogram(data, row=row, **kwargs)
        
    def _plot_history(self, data, row=0, x=None, label=None, frequency=1, **kwargs):
        if x is None:
            x = np.arange(0, len(data), frequency)
        self.ax[row,0].plot(x[::frequency], data[::frequency], label=label)
        
    def _plot_histogram(self, data, row=0, label=None, density=True, alpha=0.5, bins=31, **kwargs):
        self.ax[row,1].hist(
            data, label=label,
            orientation='horizontal',
            bins=bins, density=density,
            alpha=alpha,
        )

class ScatterMatrix:
    r'''
    Different observables are correlated as a function of Markov Chain time, because they are measured on the same state.
    We can visualize the correlations between many different observables in a grid, each panel a two-dimensional projection of the many-dimensional space of observable values.

    Parameters
    ----------
        fields: int
            Number of rows and columns.
        wspace, hspace: float [inches]
            White space between panels.
        kwargs:
            Forwarded to ``matplotlib.pyplot.subplots``
    '''
    def __init__(self,
                 fields=2,
                 wspace=0.05, hspace=0.05,
                 **kwargs
                ):

        width_ratios  = np.ones(fields)
        height_ratios = np.ones(fields)
        self.fig, self.grid = plt.subplots(
            fields, fields,
            gridspec_kw={
                'width_ratios': width_ratios,
                'height_ratios': height_ratios,
                'wspace': wspace, 'hspace': hspace
            },
            **kwargs
        )

        grid = self.grid
        for (i, j) in product(range(fields), range(fields)):
            # Share axes
            if i == 0 and j == 0:
                grid[i,j].sharex(grid[1,0])
            elif i == j:
                grid[i,j].sharex(grid[0,j])
            elif i == 0:
                grid[i,j].sharey(grid[0,1])
            else:
                grid[i,j].sharex(grid[0,j])
                grid[i,j].sharey(grid[i,0])

            # Only allow labels on the left, bottom frames.
            if j != 0:
                [label.set_visible(False) for label in grid[i,j].get_yticklabels()]
            if i != fields-1:
                [label.set_visible(False) for label in grid[i,j].get_xticklabels()]

    def plot(self, data, label=None, density=True, scatter_alpha=0.1, histogram_alpha=0.5, bins=31, **kwargs):
        r'''
        Parameters
        ----------
            data: iterable of length fields
            density: bool
                Should the histograms be normalized?
            scatter_alpha: float
                Transparency of plotted points.
            histogram_alpha: float
                Transparency of the histograms.
            bins: int
                Number of bins in each histogram.
            kwargs:
                Currently ignored.
        '''
        for ((i, y), (j, x)) in product(enumerate(data), enumerate(data)):
            if i != j:
                self.grid[i,j].scatter(x,y,
                                       alpha=scatter_alpha,
                                       edgecolors='none',
                                      )
            else:
                self.grid[i,j].hist(
                    x, label=label,
                    orientation='vertical',
                    bins=bins, density=density,
                    alpha=histogram_alpha,
                )


