from inspect import signature
import tdg.analysis

import logging
logger = logging.getLogger(__name__)

# To allow users to add derived quantities to bootstrap objects can reuse the same pattern
# as we used for smuggling observables into GrandCanonical, navigating some design considerations.
# 
#   https://github.com/evanberkowitz/two-dimensional-gasses/issues/12
# 
# The strategy is to use the descriptor protocol
# 
#   https://docs.python.org/3/howto/descriptor.html
# 
# to hook derived quantities into the Bootstrap class.

class DerivedQuantity:

    def __init_subclass__(cls, name=''):
        # This registers every subclass that inherits from DerivedQuantity.
        # Upon registration, GrandCanonical gets an attribute with the appropriate name.

        if name == '':
            cls.name = cls.__name__
        else:
            cls.name = name
        if cls.name[0] == '_':
            logger.debug(f'DerivedQuantity registered: {cls.name}')
        else:
            logger.info(f'DerivedQuantity registered: {cls.name}')
        setattr(tdg.analysis.bootstrap.Bootstrap, cls.name, cls())

    def __set_name__(self, owner, name):
        self.name  = name

    def __get__(self, obj, objtype=None):
        # The __get__ method is the workhorse of the Descriptor protocol.

        # Cache:
        if self.name in obj.__dict__:
            # What's nice about this is that the cache is in the object's dictionary itself,
            # rather than associated with the derived quantity class.  This avoids the issue of a
            # class level cache discussed in https://github.com/evanberkowitz/two-dimensional-gasses/issues/12
            # in that there's no extra reference to the object at all with this strategy.
            # So, when it goes out of scope with no reference, it will be deleted.
            return obj.__dict__[self.name]

        if objtype is tdg.analysis.Bootstrap:
            # Just call the measurement and cache the result.
            result = self.measure(obj)
            obj.__dict__[self.name] = result
            return result

        # It may be possible to further generalize but for now,
        raise NotImplementedError()

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

####
#### The Decorator
####

def derived(func):
    # Now we are ready to decorate a function and turn it into a Descriptor.
    class anonymous(DerivedQuantity, name=func.__name__):
        
        def measure(self, ensemble):
            return func(ensemble)
    
    return func # This is a hack to get sphinx to document derived quantities sensibly.

