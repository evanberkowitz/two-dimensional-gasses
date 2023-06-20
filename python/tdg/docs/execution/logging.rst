Logging
=======

Python has a built-in and widely-used `logging module`_ in the standard library.

Some parts of tdg have it integrated, following `the recommended convention`_ of initializing the module's logger with

.. code-block:: python

   import logging
   logger = logging.getLogger(__name__)

More logging will be added with time, especially as bugs are hunted.

Executables that use the default :class:`tdg.cli.ArgumentParser` can :func:`parse <tdg.cli.log.defaults>` some ``--log`` flags which allow you to format or set the `log level`_.
We have inserted a ``CITE`` level just above the ``WARNING``, which is used to acknowledge other works which you should cite if you use that piece of tdg.


.. _logging module: https://docs.python.org/3/library/logging.html
.. _the recommended convention: https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial
.. _log level: https://docs.python.org/3/library/logging.html#logging-levels
