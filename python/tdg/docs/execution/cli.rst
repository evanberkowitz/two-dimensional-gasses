Command Line Arguments
======================

While writing some sanity checks, examples, and tests, the same `ArgumentParser`_ options kept coming up.
This module organizes them and provides :class:`~.ArgumentParser`, which inherits from the standard library's
object but adds some default flags.

.. automodule:: tdg.cli
   :members:
   :undoc-members:
   :show-inheritance:

We will not provide details in this documentation about how these are implemented, though the source is commented.

.. automodule:: tdg.cli.log
   :members: defaults

.. automodule:: tdg.cli.metadata
   :members: defaults

.. _ArgumentParser: https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
