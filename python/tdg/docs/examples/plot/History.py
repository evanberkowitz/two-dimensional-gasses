#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tdg.plot as visualize 

uniform = np.random.rand(1000)-0.5
gauss   = np.random.normal(0, 0.25, 1000)

h1 = visualize.History(1)
h1.plot(uniform, label='uniform')
h1.plot(gauss,   label='gauss')
h1.histogram[0].legend()

h2 = visualize.History(2)
h2.plot(uniform, row=0, label='uniform', color='blue')
h2.plot(gauss,   row=1, label='gauss', color='green')
h2.histogram[0].legend()
h2.histogram[1].legend()

plt.show()
