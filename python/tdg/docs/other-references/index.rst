.. _references:

Comparing to Other References
=============================

.. toctree::
   :maxdepth: 2

   conventions
   PRL106110403
   PRA107043314
   EPJST217153
   PRA103063314
   PRA93023602
   PRA92033603
   adding-a-reference


We provide some utilities for drawing figures with data from the above sources.

.. autodata:: tdg.references.REFERENCES
   :no-value:

Some references have data that has been superseded by works from the same author, or are far outside the typical parameter range.  Pass a truthy ``include_all`` kwarg to include these in comparison figures. (For an example see :func:`PRL106110403 <tdg.references.PRL106110403.contact_comparison>`.)

.. autofunction:: tdg.references.contact_comparison

.. autofunction:: tdg.references.energy_comparison

