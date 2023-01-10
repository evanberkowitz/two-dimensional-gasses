*************
Visualization
*************

Visualizing Markov Chain data is a quick way to see if anything interesting is happening or if anything is obviously wrong.
There are a few kinds of common visualization strategies.

The first is to show the value of observables as a function of time.
This can help us see if anything suspicious has happened; if proposals have been repeatedly rejected, or observables don't fluctuate, there may be very long autocorrelations we need to ameliorate in our analysis.

.. autoclass:: tdg.plot.History
   :members:
   :undoc-members:
   :show-inheritance:

For example, if we have uniformly- and normally-distributed observables, we can visualize them on the same trace, or separately.

.. plot:: examples/plot/history.py
   :include-source:

Markov Chain data must often be further analyzed, and correlations in the data can affect uncertainty estimates for post-processed observables.
One way to visualize the correlation between different observables is to show scatter plots of different observables.

.. autoclass:: tdg.plot.ScatterMatrix
   :members:
   :undoc-members:
   :show-inheritance:

For example, consider two ensembles of three datasets, two of which are correlated.

.. plot:: examples/plot/scatter-matrix.py
   :include-source:

The x-axis of each column and y-axis of each row are shared; except for the diagonal, on which we plot the histograms.
We can see the correlation between the last two observables.

