from inspect import signature
import tdg.ensemble
from tdg.performance import Timer

import logging
logger = logging.getLogger(__name__)

# To allow users to add observables to GrandCanonical requires some design considerations.
# 
#   https://github.com/evanberkowitz/two-dimensional-gasses/issues/12
# 
# The strategy is to use the descriptor protocol
# 
#   https://docs.python.org/3/howto/descriptor.html
# 
# to hook observables in to the GrandCanonical class.

####
#### Observable interface
####

class Observable:

    def __init_subclass__(cls, name=''):
        # This registers every subclass that inherits from Observable.
        # Upon registration, GrandCanonical gets an attribute with the appropriate name.

        if name == '':
            cls.name = cls.__name__
        else:
            cls.name = name

        cls._logger = (logger.debug if cls.name[0] == '_' else logger.info)
        cls._logger(f'Observable registered: {cls.name}')

        setattr(tdg.ensemble.GrandCanonical, cls.name, cls())
        tdg.ensemble.GrandCanonical._observables.add(cls.name)

    def __set_name__(self, owner, name):
        self.name  = name

    def __get__(self, obj, objtype=None):
        # The __get__ method is the workhorse of the Descriptor protocol.

        # Cache:
        if self.name in obj.__dict__:
            # What's nice about this is that the cache is in the object's dictionary itself,
            # rather than associated with the observable class.  This avoids the issue of a
            # class level cache discussed in https://github.com/evanberkowitz/two-dimensional-gasses/issues/12
            # in that there's no extra reference to the object at all with this strategy.
            # So, when it goes out of scope with no reference, it will be deleted.
            self._logger(f'{self.name} already cached.')
            return obj.__dict__[self.name]

        if objtype is tdg.ensemble.GrandCanonical:
            # Just call the measurement and cache the result.
            with Timer(self._logger, f'Measurement of {self.name}', per=len(obj)):
                obj.__dict__[self.name]= self.measure(obj)
            return obj.__dict__[self.name]

        # It may be possible to further generalize and implement the canonical projections
        # or data analysis like binning and bootstrapping by detecting the class here and
        # giving a different implementation.
        #
        # But for now,
        raise NotImplemented()

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

####
#### The Decorator
####

def observable(func):
    # Now we are ready to decorate a function and turn it into a Descriptor.
    # The primary decision is:
    # 
    #   Does this function need to be a CallableObservable?
    #   Or can it just be an Observable?
    # 
    # We can decide that by counting its arguments.
    sig = signature(func)
    parameters = len(sig.parameters)

    if parameters != 1:
        raise TypeError(f'An @observable must take exactly one argument (the ensemble), not {parameters}.')

    class anonymous(Observable, name=func.__name__):
        
        def measure(self, ensemble):
            return func(ensemble)

    return func # This is a hack to get sphinx to document observables sensibly.


