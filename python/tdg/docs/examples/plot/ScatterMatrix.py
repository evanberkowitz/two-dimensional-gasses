#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tdg.plot as visualize

sm = visualize.ScatterMatrix(3, labels=('Uniform', 'Gaussian', 'Two Gaussians'))

# First ensemble
uniform = np.random.rand(1000)-0.5
gauss   = np.random.normal(0, 0.25, 1000)
gauss2  = 2*gauss + np.random.normal(0, 0.125, 1000)
sm.plot((uniform, gauss, gauss2))

# Second ensemble
uniform = np.random.rand(1000)-0.5
gauss   = np.random.normal(0, 0.25, 1000)
gauss2  = 2*gauss + np.random.normal(0, 0.125, 1000)
sm.plot((uniform, gauss, gauss2))

st = visualize.ScatterTriangle(3, labels=('Uniform', 'Gaussian', 'Two Gaussians'))
st.plot((uniform, gauss, gauss2))
