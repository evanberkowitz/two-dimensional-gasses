#!/usr/bin/env python3

import numpy as np
from itertools import product

import matplotlib.pyplot as plt

class History:

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
        
    def plot(self, data, row=0, **kwargs):
        self.plot_history  (data, row=row, **kwargs)
        self.plot_histogram(data, row=row, **kwargs)
        
    def plot_history(self, data, row=0, x=None, label=None, frequency=1, **kwargs):
        if x is None:
            x = np.arange(0, len(data), frequency)
        self.ax[row,0].plot(x, data[::frequency], label=label)
        
    def plot_histogram(self, data, row=0, label=None, density=True, alpha=0.5, bins=31, **kwargs):
        self.ax[row,1].hist(
            data, label=label,
            orientation='horizontal',
            bins=bins, density=density,
            alpha=alpha,
        )

class ScatterMatrix:

    def __init__(self,
                 fields=2, figsize=(12,12),
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
            figsize=figsize,
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


