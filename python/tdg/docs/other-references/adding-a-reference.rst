
Adding a Reference
==================

A reference is a python (sub)module, and it can provide arbitrary data and functions.
The only *requirements* are the bibliographic information.

Bibliographic Information
-------------------------

Each reference must emit a one-line logging info line the first time results from the reference are used.
The logger must emit a full bibtex entry as a debug message the first time results from the reference are used.

This can be easy if you use

.. autoclass:: tdg.references.citation.Citation

In your reference you can do

.. code::

   from tdg.references.citation import Citation
   citation = Citation(
       'AUTHOR et al., JOURNAL ETC. (YEAR)',
       '''@article{bibtex,
           title={TITLE},
           author={AUTHOR},
           journal={JOURNAL},
           volume={VOL},
           number={N},
           pages={100-200},
           year={YEAR},
           publisher={PUBLISHER}
       }''')
   
   # ... and then inside a piece of code that, if it's running, the authors deserve a citation:
   citation()

Rather than the full bibtex entry you can pass the key of the reference in the master.bib.
Each reference should set the ``label`` argument in each plot.

Results and Plotting
--------------------

The :ref:`tdg.references <references>` module provides some convenience functions such as :func:`~.references.contact_comparison` and :func:`~.references.energy_comparison`.
To automatically include the results from a custom reference in the figure, you should provide a function with the same name and the signature ``(ax, **kwargs)``.
If your module is included in the ``references`` :func:`~.references.contact_comparison` will call it, forwarding the axis and any kwargs it was given.
For example,

.. autofunction:: tdg.references.PRA107043314.contact_comparison
   :noindex:

requires the user to pass an ``alpha`` parameter to :func:`~.references.contact_comparison` and optionally understands the ``cutoff_variation``.

Some references have data that has been superseded by works from the same author, or are far outside the typical parameter range.  The convention in this case is to only plot if the kwarg ``include_all=True``. (For an example see :func:`PRL106110403 <tdg.references.PRL106110403.contact_comparison>`.)

But other references might only provide data points, or information about the critical temperature.
