#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tdg.plot as visualize 

import argparse

parser = argparse.ArgumentParser(description="Demonstrate visualizations in tdg.plot.")
parser.add_argument('--show', action="store_true", default=False, help="show the figures in interactive windows, rather than save them.")

args = parser.parse_args()

uniform = np.random.rand(1000)-0.5
gauss   = np.random.normal(0, 0.25, 1000)

h = visualize.History(1)
h.plot(uniform, label='uniform')
h.plot(gauss,   label='gauss')
h.histogram[0].legend()

if not args.show:
    plt.savefig('plot-history.pdf')

sm = visualize.ScatterMatrix(3, figsize=(6,6))
sm.plot((uniform, gauss,   uniform))
sm.plot((gauss,   uniform, gauss  ))

if not args.show:
    plt.savefig('plot-scatter.pdf')

if args.show:
    plt.show()

