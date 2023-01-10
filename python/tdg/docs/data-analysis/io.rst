***
I/O
***

`HDF5`_ is a format often used in computational physics and other data-science applications, because of its ability to store huge amounts of structured numerical data.
Many datasets can be stored in a single file, categorized, linked together, and so on.
A variety of python modules leverage HDF5 for input and output; often they rely on `h5py`_ or `PyTables`, pythonic interfaces  interoperabale with `numpy`_, and no native support of python objects.

However, more complex data structures have no native support for H5ing; a variety of choices are possible.
Any python object can be `pickled`_ and stored as a binary blob in HDF5, but the resulting blobs are not usable outside of python.
The `pandas`_ data analysis module can `read_hdf`_ and export `to_hdf`_, but even though the data is written in a usable way, the data layouts are nontrivial to read without pandas.

We aim for a happy medium, by providing a class, :class:`~.H5able`, from which other python classes, which contain a variety of data fields, can inherit to allow them to be easily written to HDF5.
An H5able object will be saved as a `group`_ that contains properties written into groups and `datasets`_, with the same name as the property itself.
If a property is one of a slew of known types then it will be written natively as an H5 field, otherwise it will be pickled.

.. autoclass:: tdg.h5.H5able
   :members:
   :undoc-members:
   :show-inheritance:

The data types that are not pickled are
 - ``H5able``
 - ``int``
 - ``float``
 - ``dict``
 - ``numpy.ndarray``
 - ``torch.tensor``, ``torch.Size``

To provide custom methods for H5ing otherwise-unknown types that cannot be made H5able, a user can write a small strategy.
A strategy is an instance-free class with just static methods ``applies``, ``write``, and ``read``.
Suppose the user had an ``Example`` class; a strategy might look like

.. code-block:: python
   
   class ExampleStrategy(H5Data, name='example'):
        '''
        The name is stored as metadata and then used to look up this strategy
        '''

        @staticmethod
        def applies(value):
            r'''
            Parameters
            ----------
                value: any value at all

            Returns
            -------
                bool: is ``value`` H5able using this interface?
            '''
            return isinstance(value, Example)

        @staticmethod
        def write(group, key, value):
            r'''
            Parameters
            ----------
                group: an H5py group in which to store the Example object in value
                key:   a string to name the object
                value: an Example to store into ``group/key``.
            '''
            group['property'] = value.example_property

        @staticmethod
        def read(group):
            r'''
            Parameters
            ----------
                group: an H5py group

            Returns
            -------
                Example: that was previously written with write.
            '''
            example_property = group['property']
            return Example(example_property)

However, it is probably simplest in most circumstances to just inherit from ``H5able``.
See **tdg/io.py** for the strategies that for ``int``, ``float``, ``dict``, ``numpy.ndarray``, ``torch.tensor``, ``torch.Size``, and ``H5able``.
If the H5able strategy is desired but the class cannot be made to inherit from ``H5able``, just create a new strategy that inherits from ``H5ableStrategy`` and overwrites the ``applies`` method.


.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _h5py: https://docs.h5py.org/en/stable/index.html
.. _PyTables: https://www.pytables.org/
.. _numpy:  https://numpy.org/
.. _pandas: https://pandas.pydata.org/pandas-docs/stable/index.html
.. _read_hdf: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
.. _to_hdf: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html
.. _pickled: https://docs.python.org/3/library/pickle.html
.. _group: https://docs.hdfgroup.org/hdf5/develop/_h5_d_m__u_g.html#subsubsec_data_model_abstract_group
.. _datasets: https://docs.hdfgroup.org/hdf5/develop/_h5_d_m__u_g.html#subsubsec_data_model_abstract_dataset
