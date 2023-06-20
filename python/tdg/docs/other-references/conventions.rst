
Conventions
===========

There are two common conventions for defining the parameters of the effective range expansion.
The more common in nuclear physics, which keeps the effective range expansion simple

.. math::
   
   \cot \delta(k) = \frac{2}{\pi} \log(ka_2) + \sigma_2 k^2 + \mathcal{O}(k^4) \text{ analytic in }k

where :math:`a_2` is the scattering length and :math:`\sqrt{|\sigma_2|}` is the effective range :cite:`Beane:2022wcn`.

The other convention is from hard-disk and square-well scattering.  The scattering length is the radius at which the wavefunction (upon linear extrapolation from the disk's boundary) hits 0.  According to :cite:`Adhikari:1986a,Adhikari:1986b,Khuri:2008ib,Galea:2017jhe` we recover this geometric meaning when

.. math::
   
   \cot \delta(k) = \frac{2}{\pi}\left[\log \frac{ka_{2D}}{2} + \gamma\right] + \frac{1}{4} R_e^2 k^2 + \cdots,

:math:`R_e` has a similar geometric meaning (see (18) in Ref. :cite:`Galea:2017jhe`), and :math:`\gamma=0.5772\ldots` is the `Euler-Mascheroni constant`_.

.. note::
   *We use the nuclear physics convention* and implement it in the :class:`~.EffectiveRangeExpansion`.

.. warning::
   **We typically do not write the subscript on** :math:`a`, **especially in code!  We just assume the nuclear convention.**

To help convert from the geometric convention we provide

.. autoclass:: tdg.conventions.from_geometric
   :members:
   :undoc-members:

.. _Euler-Mascheroni constant: https://en.wikipedia.org/wiki/Euler%27s_constant
