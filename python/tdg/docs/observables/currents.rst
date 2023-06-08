.. _current:

Currents
========

Currents tell you how conserved quantities flow from place to place.

Baryon number current
---------------------

.. autofunction:: tdg.observable.current.current
.. autofunction:: tdg.observable.current.current_squared
.. autofunction:: tdg.observable.current.current_current

We are interested in low-energy current correlations.
We define moments :math:`W`,

.. math::
   W_n(k) = \int d^2r\; e^{-ikr} |r|^n J(r).

where :math:`J(r) = \frac{1}{L^2} \int d^2 x [ J(x,x-r) = j^a(x) j^b(x-r) \delta_{ab} ]` is

.. math::
   \lim_{\Delta x \rightarrow 0} \left(\frac{1}{ML^2 \Delta x}\right)^2 \left[\tilde{J}_r = \texttt{current_current} \right] \rightarrow J(r).


:math:`W_n(k)` has dimensions :math:`[M^{-2} L^{-(4-n)}]`.

So, :math:`M^2 L^{4-n} W_n(k)` is dimensionless.  We can divide by appropriate powers of :math:`2\pi N = k_F^2 L^2` to eliminate :math:`L`.

These have good continuum- and infinite-volume limits.

.. autofunction:: tdg.observable.current.w0_by_kF4
.. autofunction:: tdg.observable.current.w2_by_kF2
.. autofunction:: tdg.observable.current.w4

