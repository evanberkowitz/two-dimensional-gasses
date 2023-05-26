
Adding a Reference
==================

A reference is a python (sub)module, and it can provide arbitrary data and functions.
The only *requirements* are the bibliographic information.

Bibliographic Information
-------------------------

Each reference must emit a one-line logging info line the first time results from the reference are used.
The logger must emit a full bibtex entry as a debug message the first time results from the reference are used.

See, for example

.. autofunction:: tdg.others.PRA107043314._cite

Each reference must include a ``label`` variable that can be used in figure legends.

Results and Plotting
--------------------

The :ref:`tdg.others <others>` module provides some convenience functions such as :func:`~.others.contact_comparison` and :func:`~.others.energy_comparison`.
To automatically include the results from a custom reference in the figure, you should provide a function with the same name and the signature ``(ax, **kwargs)``.
If your module is included in the ``references`` :func:`~.others.contact_comparison` will call it, forwarding the axis and any kwargs it was given.
For example,

.. autofunction:: tdg.others.PRA107043314.contact_comparison
   :noindex:

requires the user to pass an ``alpha`` parameter to :func:`~.others.contact_comparison` and optionally understands the ``cutoff_variation``.

But other references might only provide data points, or information about the critical temperature.
