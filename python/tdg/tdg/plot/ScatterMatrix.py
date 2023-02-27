#!/usr/bin/env python3

import numpy as np
import torch
from itertools import product

import matplotlib.pyplot as plt

class ScatterMatrix:
    r'''
    Different observables are correlated as a function of Markov Chain time, because they are measured on the same state.
    We can visualize the correlations between many different observables in a grid, each panel a two-dimensional projection of the many-dimensional space of observable values.

    Parameters
    ----------
        fields: int
            Number of rows and columns.
        labels: iterable of strings of length :code:`fields`
            Names for the different axes that will correspond to the plotted fields.
        wspace, hspace: float [inches]
            White space between panels.
        kwargs:
            Forwarded to ``matplotlib.pyplot.subplots``

    .. note::
        If you prefer more whitespace, consider a :class:`ScatterTriangle` over a :class:`~.ScatterMatrix`.
        It has the same interface.
    '''
    def __init__(self,
                 fields=2,
                 labels=None,
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

            # Only allow ticks on the left, bottom frames.
            if j != 0:
                [label.set_visible(False) for label in grid[i,j].get_yticklabels()]
            if i != fields-1:
                [label.set_visible(False) for label in grid[i,j].get_xticklabels()]

            # Labels
            if labels is not None and len(labels) == fields:
                if j == 0:
                    grid[i,j].set_ylabel(labels[i])
                if i == fields-1:
                    grid[i,j].set_xlabel(labels[j])

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
        d = tuple(d.clone().detach().cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data)
        for ((i, y), (j, x)) in product(enumerate(d), enumerate(d)):
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

class ScatterTriangle:
    r'''
    If you prefer more whitespace, consider a ScatterTriangle over a ScatterMatrix.

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
                 labels=None,
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
            if j > i:
                grid[i,j].axis('off')
                continue

            # Share axes
            if i == 0 and j == 0:
                grid[i,j].sharex(grid[1,0])
            elif i == j:
                grid[i,j].sharey(grid[0,j])
            elif i == 0:
                grid[i,j].sharey(grid[0,1])
            else:
                grid[i,j].sharex(grid[0,j])
                grid[i,j].sharey(grid[i,0])

            # Only allow ticks on the left, bottom frames.
            if j != 0:
                [label.set_visible(False) for label in grid[i,j].get_yticklabels()]
            if i < fields-1 or j == fields-1:
                [label.set_visible(False) for label in grid[i,j].get_xticklabels()]
            grid[0,0].set_yticks([])

            # Labels
            if labels is not None and len(labels) == fields:
                if j == 0 and i > 0:
                    grid[i,j].set_ylabel(labels[i])
                if i == fields-1 and j < i:
                    grid[i,j].set_xlabel(labels[j])

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
        d = tuple(d.clone().detach().cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data)
        for ((i, y), (j, x)) in product(enumerate(d), enumerate(d)):
            if j > i:
                continue
            if i != j:
                self.grid[i,j].scatter(x,y,
                                       alpha=scatter_alpha,
                                       edgecolors='none',
                                      )
            else:
                self.grid[i,j].hist(
                    x, label=label,
                    orientation=('vertical' if i ==0 else 'horizontal'),
                    bins=bins, density=density,
                    alpha=histogram_alpha,
                )


